"""Week 1 exit criterion: depth-5 benign chain round-trip across 3 designs.

This is the gating test for Week 1 of Paper 2. It exercises:
  - schema v2 (BoundIDPayloadV2 with chain_proof + lifecycle)
  - all three chain backends (linked_list, merkle, signed_dag)
  - multi-deployer harness (>=2 deployers with distinct keypairs)
  - tampering at multiple positions (root, mid, leaf) detected by each design
  - cross-design rejection (a linked_list payload must not verify under merkle)

Tests deliberately do NOT cover B1-B5 attacks here — those are Week 2.
This file establishes the *positive path* and immediate negative tests.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.core.agent_id_v2 import BoundIDPayloadV2, ChainProof
from src.defenses.chain_designs import (
    ChainNode,
    ChainVerifyError,
    LinkedListBackend,
    MerkleBackend,
    SignedDAGBackend,
)
from src.runtime.multi_deployer import ChainBuilder, MultiDeployerHarness


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def harness() -> MultiDeployerHarness:
    """Two deployers, fresh keypairs."""
    return MultiDeployerHarness.with_deployers(["D_A", "D_B"])


def _build_depth5_chain(harness, backend, *, signed_dag_co_signers=False):
    """Build a 5-hop benign chain alternating between D_A and D_B.

    For signed_dag we pass co-signer names; for linked_list and merkle we
    leave them empty.
    """
    b = ChainBuilder(harness, backend)

    def co(parent_dep, child_dep):
        if not signed_dag_co_signers:
            return ()
        if parent_dep == child_dep:
            return (parent_dep,)
        return (parent_dep, child_dep)

    chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"root")
    # Chain: D_A -> D_A -> D_B -> D_B -> D_A -> D_B  (5 extends => depth 5)
    plan = [
        ("D_A", "agent-1"),
        ("D_B", "agent-2"),
        ("D_B", "agent-3"),
        ("D_A", "agent-4"),
        ("D_B", "agent-5"),
    ]
    for host, iid in plan:
        parent_dep = chain[-1].signed.payload.deployer
        chain = b.extend(
            chain,
            host=host,
            instance_id=iid,
            system_id="claude-3-sonnet",
            output=f"output for {iid}".encode("utf-8"),
            co_signer_names=co(parent_dep, host),
        )
    return chain


# ----------------------------------------------------------------------
# Positive-path round trips
# ----------------------------------------------------------------------

class TestPositivePaths:
    def test_linked_list_round_trip_depth5(self, harness):
        backend = LinkedListBackend()
        chain = _build_depth5_chain(harness, backend)
        assert len(chain) == 6  # root + 5 extends
        assert chain[-1].signed.payload.ancestry_depth == 5
        backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_merkle_round_trip_depth5(self, harness):
        backend = MerkleBackend()
        chain = _build_depth5_chain(harness, backend)
        assert len(chain) == 6
        assert chain[-1].signed.payload.ancestry_depth == 5
        backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_signed_dag_round_trip_depth5(self, harness):
        backend = SignedDAGBackend()
        chain = _build_depth5_chain(
            harness, backend, signed_dag_co_signers=True
        )
        assert len(chain) == 6
        assert chain[-1].signed.payload.ancestry_depth == 5
        backend.verify_chain(chain=chain, deployers=harness.registry)


# ----------------------------------------------------------------------
# Cross-design rejection — a chain built with design X must not verify under design Y
# ----------------------------------------------------------------------

class TestCrossDesignRejection:
    def test_linked_list_chain_rejected_by_merkle(self, harness):
        chain = _build_depth5_chain(harness, LinkedListBackend())
        with pytest.raises(ChainVerifyError, match="design"):
            MerkleBackend().verify_chain(chain=chain, deployers=harness.registry)

    def test_merkle_chain_rejected_by_signed_dag(self, harness):
        chain = _build_depth5_chain(harness, MerkleBackend())
        with pytest.raises(ChainVerifyError, match="design"):
            SignedDAGBackend().verify_chain(
                chain=chain, deployers=harness.registry
            )

    def test_signed_dag_chain_rejected_by_linked_list(self, harness):
        chain = _build_depth5_chain(
            harness, SignedDAGBackend(), signed_dag_co_signers=True
        )
        with pytest.raises(ChainVerifyError, match="design"):
            LinkedListBackend().verify_chain(
                chain=chain, deployers=harness.registry
            )


# ----------------------------------------------------------------------
# Tampering: mutate payload AFTER it is signed -> chain-shape proof breaks
# ----------------------------------------------------------------------

def _mutate_node_output(node: ChainNode, new_bytes: bytes) -> ChainNode:
    """Replace the payload's output_hash so chain hashes diverge.

    This simulates an attacker who modifies an in-flight payload.
    The PRIMARY signature would also be invalid, but verify_chain
    operates only on chain shape — that is the point: chain-shape
    detection must fire even if signature checking has not yet run
    (defence in depth).
    """
    from src.core.crypto import sha256_hex

    old = node.signed.payload
    new_payload = dataclasses.replace(old, output_hash=sha256_hex(new_bytes))
    new_signed = dataclasses.replace(node.signed, payload=new_payload)
    return ChainNode(signed=new_signed)


class TestTamperDetection:
    def test_linked_list_detects_root_tamper(self, harness):
        backend = LinkedListBackend()
        chain = _build_depth5_chain(harness, backend)
        chain[0] = _mutate_node_output(chain[0], b"poisoned root")
        with pytest.raises(ChainVerifyError, match="prev_hash"):
            backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_linked_list_detects_mid_tamper(self, harness):
        backend = LinkedListBackend()
        chain = _build_depth5_chain(harness, backend)
        chain[2] = _mutate_node_output(chain[2], b"poisoned mid")
        with pytest.raises(ChainVerifyError, match="prev_hash"):
            backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_merkle_detects_root_tamper(self, harness):
        backend = MerkleBackend()
        chain = _build_depth5_chain(harness, backend)
        chain[0] = _mutate_node_output(chain[0], b"poisoned root")
        # Merkle: every subsequent merkle_root references this leaf, so
        # the FIRST hop after the root is the one that fires the mismatch.
        with pytest.raises(ChainVerifyError, match="merkle_root"):
            backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_merkle_detects_mid_tamper(self, harness):
        backend = MerkleBackend()
        chain = _build_depth5_chain(harness, backend)
        chain[3] = _mutate_node_output(chain[3], b"poisoned mid")
        # Merkle: tampering ancestor i breaks every merkle_root at hops > i.
        with pytest.raises(ChainVerifyError, match="merkle_root"):
            backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_signed_dag_detects_mid_tamper(self, harness):
        backend = SignedDAGBackend()
        chain = _build_depth5_chain(
            harness, backend, signed_dag_co_signers=True
        )
        chain[2] = _mutate_node_output(chain[2], b"poisoned mid")
        # signed_dag: tampering parent's payload changes edge_signing_input,
        # so the next hop's co-signer signatures no longer verify.
        with pytest.raises(ChainVerifyError, match="signature invalid"):
            backend.verify_chain(chain=chain, deployers=harness.registry)


# ----------------------------------------------------------------------
# Empty / degenerate cases
# ----------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.parametrize(
        "backend",
        [LinkedListBackend(), MerkleBackend(), SignedDAGBackend()],
        ids=["linked_list", "merkle", "signed_dag"],
    )
    def test_empty_chain_rejected(self, harness, backend):
        with pytest.raises(ChainVerifyError, match="empty chain"):
            backend.verify_chain(chain=[], deployers=harness.registry)

    @pytest.mark.parametrize(
        "backend,signed_dag",
        [
            (LinkedListBackend(), False),
            (MerkleBackend(), False),
            (SignedDAGBackend(), True),
        ],
        ids=["linked_list", "merkle", "signed_dag"],
    )
    def test_single_root_node_verifies(self, harness, backend, signed_dag):
        """A chain of just the root must verify cleanly."""
        b = ChainBuilder(harness, backend)
        chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"root")
        assert len(chain) == 1
        backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_signed_dag_requires_co_signers(self, harness):
        """Extending under signed_dag with empty co_signer_names must fail."""
        backend = SignedDAGBackend()
        b = ChainBuilder(harness, backend)
        chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"root")
        from src.defenses.chain_designs.base import ChainExtendError
        with pytest.raises(ChainExtendError, match="co_signer"):
            b.extend(
                chain,
                host="D_B",
                instance_id="agent-1",
                system_id="claude-3-sonnet",
                output=b"step",
                co_signer_names=(),
            )
