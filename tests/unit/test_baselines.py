"""Tests for Paper 2 E2 baseline backends and E3 key-compromise ablation.

E2 baselines: no_chain, hash_only, naive_sig
E3 key-compromise: single-deployer compromise across all three primary designs

These tests verify two things:
  1. Baselines are WEAK where expected (attacks bypass verification)
  2. Primary designs RESIST the same attacks (regression check)

This establishes that ASR=0 in the main experiment is due to chain design
strength, not weak attacks.
"""

from __future__ import annotations

import dataclasses
import pytest

from src.core.agent_id_v2 import (
    ChainProof,
    make_chain_proof_linked_list,
    make_chain_proof_merkle,
    make_chain_proof_signed_dag,
)
from src.core.crypto import generate_keypair, sha256_hex, sign
from src.defenses.chain_designs import (
    ChainNode,
    DeployerRegistry,
    HashOnlyBackend,
    LinkedListBackend,
    MerkleBackend,
    NaiveSigBackend,
    NoChainBackend,
    SignedDAGBackend,
)
from src.defenses.chain_designs.base import ChainVerifyError, edge_signing_input
from src.runtime.multi_deployer import ChainBuilder, MultiDeployerHarness


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def harness():
    """Two-deployer harness with known keypairs."""
    return MultiDeployerHarness.with_deployers(["D_A", "D_B"])


def _build_chain(harness, backend, *, depth=4, cross_deployer=False, co_sign=False):
    """Build a benign chain of given depth using the backend."""
    b = ChainBuilder(harness, backend=backend)
    hosts = ["D_A", "D_B"] if cross_deployer else ["D_A"]
    chain = b.start_root(
        hosts[0], "a0", "claude-3-sonnet", b"root-output",
    )
    for i in range(1, depth):
        host = hosts[i % len(hosts)] if cross_deployer else hosts[0]
        co_signers = tuple(harness.names()) if co_sign else ()
        chain = b.extend(
            chain, host=host, instance_id=f"a{i}",
            system_id="claude-3-sonnet", output=f"step-{i}".encode(),
            co_signer_names=co_signers,
        )
    return chain


# ============================================================================
# E2: no_chain baseline — everything bypasses
# ============================================================================


class TestNoChainBaseline:
    """no_chain accepts any chain. All attacks should bypass."""

    def test_benign_chain_accepted(self, harness):
        backend = NoChainBackend()
        chain = _build_chain(harness, backend)
        backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_empty_chain_rejected(self, harness):
        backend = NoChainBackend()
        with pytest.raises(ChainVerifyError, match="empty"):
            backend.verify_chain(chain=[], deployers=harness.registry)

    def test_b1_orphan_bypasses(self, harness):
        """Orphan child with wrong parent_id passes no_chain."""
        backend = NoChainBackend()
        chain = _build_chain(harness, backend)
        # Modify last node's parent_id to nonsense
        old = chain[-1].signed.payload
        mutated = dataclasses.replace(old, parent_id="nonexistent")
        chain[-1] = ChainNode(signed=dataclasses.replace(
            chain[-1].signed, payload=mutated,
        ))
        # Should NOT raise
        backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_b4_ancestor_revision_bypasses(self, harness):
        """Mutated ancestor passes no_chain."""
        backend = NoChainBackend()
        chain = _build_chain(harness, backend)
        old = chain[1].signed.payload
        mutated = dataclasses.replace(old, output_hash=sha256_hex(b"evil"))
        chain[1] = ChainNode(signed=dataclasses.replace(
            chain[1].signed, payload=mutated,
        ))
        backend.verify_chain(chain=chain, deployers=harness.registry)


# ============================================================================
# E2: hash_only baseline — only output_hash presence checked
# ============================================================================


class TestHashOnlyBaseline:
    """hash_only checks output_hash presence only. Chain structure unchecked."""

    def test_benign_accepted(self, harness):
        backend = HashOnlyBackend()
        chain = _build_chain(harness, backend)
        backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_b1_orphan_bypasses(self, harness):
        """Orphan child bypasses because parent_id is not checked."""
        backend = HashOnlyBackend()
        chain = _build_chain(harness, backend)
        old = chain[-1].signed.payload
        mutated = dataclasses.replace(old, parent_id="nonexistent")
        chain[-1] = ChainNode(signed=dataclasses.replace(
            chain[-1].signed, payload=mutated,
        ))
        backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_b2_splice_bypasses(self, harness):
        """Removing a middle node bypasses hash_only."""
        backend = HashOnlyBackend()
        chain = _build_chain(harness, backend, depth=5)
        # Remove node at index 2
        spliced = [chain[0], chain[1], chain[3], chain[4]]
        backend.verify_chain(chain=spliced, deployers=harness.registry)

    def test_empty_output_hash_detected(self, harness):
        """Empty output_hash is the one thing hash_only catches."""
        backend = HashOnlyBackend()
        chain = _build_chain(harness, backend)
        old = chain[1].signed.payload
        mutated = dataclasses.replace(old, output_hash="")
        chain[1] = ChainNode(signed=dataclasses.replace(
            chain[1].signed, payload=mutated,
        ))
        with pytest.raises(ChainVerifyError, match="empty output_hash"):
            backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_design_mismatch_detected(self, harness):
        """Wrong design tag is detected."""
        backend = HashOnlyBackend()
        chain = _build_chain(harness, backend)
        old = chain[0].signed.payload
        mutated = dataclasses.replace(
            old, chain_proof=ChainProof(design="linked_list"),
        )
        chain[0] = ChainNode(signed=dataclasses.replace(
            chain[0].signed, payload=mutated,
        ))
        with pytest.raises(ChainVerifyError, match="design"):
            backend.verify_chain(chain=chain, deployers=harness.registry)


# ============================================================================
# E2: naive_sig baseline — per-node signature only
# ============================================================================


class TestNaiveSigBaseline:
    """naive_sig verifies signatures individually but not chain shape."""

    def test_benign_accepted(self, harness):
        backend = NaiveSigBackend()
        chain = _build_chain(harness, backend)
        backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_b1_orphan_bypasses(self, harness):
        """Orphan child with valid signature but wrong parent_id bypasses.

        We can't easily forge a signature, so instead we test that removing
        a middle node (keeping valid signatures) bypasses the check.
        """
        backend = NaiveSigBackend()
        chain = _build_chain(harness, backend, depth=5)
        # B2 splice: remove middle, keep individually-signed nodes
        spliced = [chain[0], chain[1], chain[3], chain[4]]
        backend.verify_chain(chain=spliced, deployers=harness.registry)

    def test_b4_without_resign_detected(self, harness):
        """Mutated ancestor WITHOUT re-signing should fail (sig check)."""
        backend = NaiveSigBackend()
        chain = _build_chain(harness, backend)
        old = chain[1].signed.payload
        mutated = dataclasses.replace(old, output_hash=sha256_hex(b"evil"))
        chain[1] = ChainNode(signed=dataclasses.replace(
            chain[1].signed, payload=mutated,
        ))
        with pytest.raises(ChainVerifyError, match="signature"):
            backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_b4_with_resign_bypasses(self, harness):
        """Mutated ancestor WITH re-signing bypasses naive_sig.

        This is the key-compromise scenario: attacker has the deployer key,
        re-signs the mutated payload, and naive_sig doesn't check chain shape.
        """
        backend = NaiveSigBackend()
        chain = _build_chain(harness, backend)
        compromised_kp = harness.registry.get_owner("D_A")

        old = chain[1].signed.payload
        mutated = dataclasses.replace(old, output_hash=sha256_hex(b"evil"))
        new_sig = sign(compromised_kp.private, mutated.to_canonical_json())
        chain[1] = ChainNode(signed=dataclasses.replace(
            chain[1].signed, payload=mutated, signature_hex=new_sig,
        ))
        backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_unknown_deployer_detected(self, harness):
        """Node signed by unknown deployer fails."""
        backend = NaiveSigBackend()
        chain = _build_chain(harness, backend)
        old = chain[0].signed.payload
        mutated = dataclasses.replace(old, deployer="unknown_deployer")
        chain[0] = ChainNode(signed=dataclasses.replace(
            chain[0].signed, payload=mutated,
        ))
        with pytest.raises(ChainVerifyError, match="not in registry"):
            backend.verify_chain(chain=chain, deployers=harness.registry)


# ============================================================================
# E2: Contrast — primary designs RESIST what baselines allow
# ============================================================================


class TestPrimaryDesignsResist:
    """Regression: primary designs catch what baselines miss."""

    @pytest.mark.parametrize("BackendCls", [
        LinkedListBackend, MerkleBackend,
    ])
    def test_b1_orphan_detected(self, harness, BackendCls):
        """Primary designs detect orphan parent_id."""
        backend = BackendCls()
        chain = _build_chain(harness, backend)
        old = chain[-1].signed.payload
        mutated = dataclasses.replace(old, parent_id="nonexistent")
        chain[-1] = ChainNode(signed=dataclasses.replace(
            chain[-1].signed, payload=mutated,
        ))
        with pytest.raises(ChainVerifyError):
            backend.verify_chain(chain=chain, deployers=harness.registry)

    @pytest.mark.parametrize("BackendCls", [
        LinkedListBackend, MerkleBackend,
    ])
    def test_b4_without_resign_detected(self, harness, BackendCls):
        """Primary designs detect ancestor mutation without re-signing."""
        backend = BackendCls()
        chain = _build_chain(harness, backend)
        old = chain[1].signed.payload
        mutated = dataclasses.replace(old, output_hash=sha256_hex(b"evil"))
        chain[1] = ChainNode(signed=dataclasses.replace(
            chain[1].signed, payload=mutated,
        ))
        with pytest.raises(ChainVerifyError):
            backend.verify_chain(chain=chain, deployers=harness.registry)


# ============================================================================
# E3: Key-Compromise Ablation
# ============================================================================


class TestKeyCompriseLinkedList:
    """E3: linked_list collapses under single-deployer key compromise."""

    def test_b4_ancestor_revision_with_resign(self, harness):
        """Attacker mutates ancestor, re-signs entire downstream chain.

        With the compromised key, attacker can:
        1. Mutate ancestor's payload
        2. Re-sign the mutated ancestor
        3. Re-compute and re-sign every downstream child's prev_hash
        → linked_list verify_chain passes.
        """
        backend = LinkedListBackend()
        chain = _build_chain(harness, backend, depth=4)
        compromised_kp = harness.registry.get_owner("D_A")

        # Mutate node 1
        old_p = chain[1].signed.payload
        new_p = dataclasses.replace(old_p, output_hash=sha256_hex(b"evil"))
        new_sig = sign(compromised_kp.private, new_p.to_canonical_json())
        chain[1] = ChainNode(signed=dataclasses.replace(
            chain[1].signed, payload=new_p, signature_hex=new_sig,
        ))

        # Re-chain downstream nodes
        for i in range(2, len(chain)):
            parent_p = chain[i - 1].signed.payload
            child_p = chain[i].signed.payload
            new_prev = sha256_hex(parent_p.to_canonical_json())
            child_new = dataclasses.replace(
                child_p,
                chain_proof=make_chain_proof_linked_list(new_prev),
            )
            child_sig = sign(compromised_kp.private, child_new.to_canonical_json())
            chain[i] = ChainNode(signed=dataclasses.replace(
                chain[i].signed, payload=child_new, signature_hex=child_sig,
            ))

        # Should pass — linked_list collapses
        backend.verify_chain(chain=chain, deployers=harness.registry)


class TestKeyCompromiseMerkle:
    """E3: merkle collapses under same-deployer key compromise."""

    def test_b4_ancestor_revision_with_full_recompute(self, harness):
        """Attacker re-computes all Merkle roots + re-signs all downstream.

        More work than linked_list (O(depth^2) hashes) but still feasible
        with the compromised key.
        """
        backend = MerkleBackend()
        chain = _build_chain(harness, backend, depth=4)
        compromised_kp = harness.registry.get_owner("D_A")

        # Mutate node 1
        old_p = chain[1].signed.payload
        new_p = dataclasses.replace(old_p, output_hash=sha256_hex(b"evil"))
        new_sig = sign(compromised_kp.private, new_p.to_canonical_json())
        chain[1] = ChainNode(signed=dataclasses.replace(
            chain[1].signed, payload=new_p, signature_hex=new_sig,
        ))

        # Re-compute merkle roots and re-sign downstream
        for i in range(2, len(chain)):
            ancestors = [n.signed.payload for n in chain[:i]]
            parent = chain[i - 1].signed.payload
            new_proof = backend.build_proof_for_extend(
                parent=parent,
                child_instance_id=chain[i].signed.payload.instance_id,
                ancestors=ancestors,
                deployers=harness.registry,
            )
            child_p = chain[i].signed.payload
            child_new = dataclasses.replace(child_p, chain_proof=new_proof)
            child_sig = sign(compromised_kp.private, child_new.to_canonical_json())
            chain[i] = ChainNode(signed=dataclasses.replace(
                chain[i].signed, payload=child_new, signature_hex=child_sig,
            ))

        # Should pass — merkle collapses for same-deployer chains
        backend.verify_chain(chain=chain, deployers=harness.registry)


class TestKeyCompromiseSignedDAG:
    """E3: signed_dag resists for cross-deployer, collapses for same-deployer."""

    def test_same_deployer_collapses(self, harness):
        """Same-deployer signed_dag chain collapses — attacker has all keys."""
        backend = SignedDAGBackend()
        b = ChainBuilder(harness, backend=backend)
        chain = b.start_root("D_A", "a0", "claude", b"root")
        chain = b.extend(
            chain, host="D_A", instance_id="a1", system_id="claude",
            output=b"step1", co_signer_names=("D_A",),
        )
        chain = b.extend(
            chain, host="D_A", instance_id="a2", system_id="claude",
            output=b"step2", co_signer_names=("D_A",),
        )

        compromised_kp = harness.registry.get_owner("D_A")

        # Mutate root
        old_p = chain[0].signed.payload
        new_p = dataclasses.replace(old_p, output_hash=sha256_hex(b"evil"))
        new_sig = sign(compromised_kp.private, new_p.to_canonical_json())
        chain[0] = ChainNode(signed=dataclasses.replace(
            chain[0].signed, payload=new_p, signature_hex=new_sig,
        ))

        # Re-sign edge sigs and primary sigs for downstream
        for i in range(1, len(chain)):
            parent = chain[i - 1].signed.payload
            child_p = chain[i].signed.payload
            unsigned_proof = ChainProof(design="signed_dag", co_signers=("D_A",))
            child_for_edge = dataclasses.replace(child_p, chain_proof=unsigned_proof)
            edge_sig = sign(
                compromised_kp.private, edge_signing_input(parent, child_for_edge)
            )
            new_proof = make_chain_proof_signed_dag(
                co_signers=("D_A",), signatures=(edge_sig,),
            )
            child_new = dataclasses.replace(child_p, chain_proof=new_proof)
            child_sig = sign(compromised_kp.private, child_new.to_canonical_json())
            chain[i] = ChainNode(signed=dataclasses.replace(
                chain[i].signed, payload=child_new, signature_hex=child_sig,
            ))

        backend.verify_chain(chain=chain, deployers=harness.registry)

    def test_cross_deployer_resists(self, harness):
        """Cross-deployer signed_dag resists — attacker lacks D_B's key.

        Attacker compromises D_A, mutates a D_A-hosted node, but cannot
        forge D_B's co-signature on the edge. Verification must fail.
        """
        backend = SignedDAGBackend()
        b = ChainBuilder(harness, backend=backend)
        chain = b.start_root("D_A", "a0", "claude", b"root")
        chain = b.extend(
            chain, host="D_B", instance_id="a1", system_id="claude",
            output=b"step1", co_signer_names=("D_A", "D_B"),
        )

        compromised_kp = harness.registry.get_owner("D_A")

        # Mutate root (D_A's node)
        old_p = chain[0].signed.payload
        new_p = dataclasses.replace(old_p, output_hash=sha256_hex(b"evil"))
        new_sig = sign(compromised_kp.private, new_p.to_canonical_json())
        chain[0] = ChainNode(signed=dataclasses.replace(
            chain[0].signed, payload=new_p, signature_hex=new_sig,
        ))

        # Try to forge edge sigs — can sign as D_A but not D_B.
        child_p = chain[1].signed.payload
        unsigned_proof = ChainProof(
            design="signed_dag", co_signers=("D_A", "D_B")
        )
        child_for_edge = dataclasses.replace(child_p, chain_proof=unsigned_proof)
        d_a_edge_sig = sign(
            compromised_kp.private, edge_signing_input(new_p, child_for_edge)
        )
        # Use D_A's sig for D_B's slot (forged — will fail).
        forged_proof = make_chain_proof_signed_dag(
            co_signers=("D_A", "D_B"),
            signatures=(d_a_edge_sig, d_a_edge_sig),
        )
        child_new = dataclasses.replace(child_p, chain_proof=forged_proof)
        # Can't re-sign the primary sig (D_B's key needed)
        # Keep original sig — it won't match the new payload anyway
        chain[1] = ChainNode(signed=dataclasses.replace(
            chain[1].signed, payload=child_new,
        ))

        with pytest.raises(ChainVerifyError):
            backend.verify_chain(chain=chain, deployers=harness.registry)


# ============================================================================
# E2 + E3 Summary Matrix (parametric)
# ============================================================================


class TestBaselineVsPrimaryMatrix:
    """Parametric test: same attack, baseline vs primary → different outcomes."""

    @pytest.mark.parametrize(
        "BackendCls,expect_bypass",
        [
            (NoChainBackend, True),
            (HashOnlyBackend, True),
            (LinkedListBackend, False),
            (MerkleBackend, False),
        ],
    )
    def test_b4_ancestor_revision_no_resign(self, harness, BackendCls, expect_bypass):
        """B4 ancestor revision WITHOUT re-signing.

        Baselines: bypass. Primary designs: detect.
        """
        backend = BackendCls()
        chain = _build_chain(harness, backend, depth=4)
        old = chain[1].signed.payload
        mutated = dataclasses.replace(old, output_hash=sha256_hex(b"evil"))
        chain[1] = ChainNode(signed=dataclasses.replace(
            chain[1].signed, payload=mutated,
        ))

        if expect_bypass:
            backend.verify_chain(chain=chain, deployers=harness.registry)
        else:
            with pytest.raises(ChainVerifyError):
                backend.verify_chain(chain=chain, deployers=harness.registry)
