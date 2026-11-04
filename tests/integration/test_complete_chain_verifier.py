"""Regression tests for the composed receipt + ancestry verifier."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from src.core.crypto import sign
from src.core.agent_id_v2 import make_chain_proof_signed_dag
from src.defenses.chain_designs import (
    ChainNode,
    ChainVerifyError,
    LinkedListBackend,
    SignedDAGBackend,
)
from src.defenses.complete_chain_verifier import verify_complete_chain
from src.runtime.multi_deployer import ChainBuilder, MultiDeployerHarness


def _chain_and_outputs():
    harness = MultiDeployerHarness.with_deployers(["D_A", "D_B"])
    backend = LinkedListBackend()
    builder = ChainBuilder(harness, backend)
    chain = builder.start_root("D_A", "agent-a", "system", b"root")
    chain = builder.extend(
        chain,
        host="D_B",
        instance_id="agent-b",
        system_id="system",
        output=b"child",
    )
    return harness, backend, chain, [b"root", b"child"]


def test_complete_verifier_accepts_valid_receipts_and_ancestry():
    harness, backend, chain, outputs = _chain_and_outputs()
    verify_complete_chain(
        chain=chain, outputs=outputs, backend=backend, deployers=harness.registry
    )


def test_complete_verifier_rejects_invalid_primary_signature():
    harness, backend, chain, outputs = _chain_and_outputs()
    chain[-1] = ChainNode(
        signed=dataclasses.replace(chain[-1].signed, signature_hex="00" * 64)
    )
    with pytest.raises(ChainVerifyError, match="primary signature invalid"):
        verify_complete_chain(
            chain=chain, outputs=outputs, backend=backend, deployers=harness.registry
        )


def test_complete_verifier_rejects_unknown_key_id_and_revoked_key():
    harness, backend, chain, outputs = _chain_and_outputs()
    unknown_key = ChainNode(
        signed=dataclasses.replace(chain[-1].signed, key_id="00" * 8)
    )
    with pytest.raises(ChainVerifyError, match="key_id does not match"):
        verify_complete_chain(
            chain=[chain[0], unknown_key],
            outputs=outputs,
            backend=backend,
            deployers=harness.registry,
        )

    harness.registry.revoke_key(chain[-1].signed.key_id)
    with pytest.raises(ChainVerifyError, match="signer key is revoked"):
        verify_complete_chain(
            chain=chain, outputs=outputs, backend=backend, deployers=harness.registry
        )


def test_complete_verifier_rejects_released_byte_substitution():
    harness, backend, chain, outputs = _chain_and_outputs()
    outputs[-1] = b"attacker-controlled bytes"
    with pytest.raises(ChainVerifyError, match="output hash mismatch"):
        verify_complete_chain(
            chain=chain, outputs=outputs, backend=backend, deployers=harness.registry
        )


def test_complete_verifier_rejects_replayed_nonce_even_when_resigned():
    harness, backend, chain, outputs = _chain_and_outputs()
    root_nonce = chain[0].signed.payload.nonce
    child = chain[1].signed
    replayed_payload = dataclasses.replace(child.payload, nonce=root_nonce)
    keypair = harness.registry.get_owner("D_B")
    chain[1] = ChainNode(
        signed=dataclasses.replace(
            child,
            payload=replayed_payload,
            signature_hex=sign(keypair.private, replayed_payload.to_canonical_json()),
        )
    )
    with pytest.raises(ChainVerifyError, match="reuses a nonce"):
        verify_complete_chain(
            chain=chain, outputs=outputs, backend=backend, deployers=harness.registry
        )


def test_complete_verifier_rejects_a_second_presentation_with_persistent_nonce_store():
    """A verifier session must consume nonces only after a successful chain."""
    from src.core.verifier import InMemoryNonceStore

    harness, backend, chain, outputs = _chain_and_outputs()
    nonces = InMemoryNonceStore()
    verify_complete_chain(
        chain=chain,
        outputs=outputs,
        backend=backend,
        deployers=harness.registry,
        nonces=nonces,
    )
    with pytest.raises(ChainVerifyError, match="already consumed"):
        verify_complete_chain(
            chain=chain,
            outputs=outputs,
            backend=backend,
            deployers=harness.registry,
            nonces=nonces,
        )


def _signed_dag_chain():
    harness = MultiDeployerHarness.with_deployers(["D_A", "D_B"])
    backend = SignedDAGBackend()
    builder = ChainBuilder(harness, backend)
    chain = builder.start_root("D_A", "agent-a", "system-a", b"root")
    chain = builder.extend(
        chain,
        host="D_B",
        instance_id="agent-b",
        system_id="system-b",
        output=b"child",
        co_signer_names=("D_A", "D_B"),
    )
    return harness, backend, chain


@pytest.mark.parametrize(
    "field,value",
    [
        ("system_id", "attacker-system"),
        ("output_hash", "00" * 32),
        ("nonce", "11" * 16),
        ("binding_scope", "per_turn"),
    ],
)
def test_signed_dag_edge_binds_security_relevant_child_fields(field, value):
    """Even a compromised child key cannot alter a parent-authorized edge."""
    harness, backend, chain = _signed_dag_chain()
    child = chain[1].signed
    mutated_payload = dataclasses.replace(child.payload, **{field: value})
    child_kp = harness.registry.get_owner("D_B")
    chain[1] = ChainNode(
        signed=dataclasses.replace(
            child,
            payload=mutated_payload,
            signature_hex=sign(child_kp.private, mutated_payload.to_canonical_json()),
        )
    )
    with pytest.raises(ChainVerifyError, match="signature invalid"):
        backend.verify_chain(chain=chain, deployers=harness.registry)


def test_signed_dag_edge_binds_the_agreed_co_signer_set():
    harness, backend, chain = _signed_dag_chain()
    child = chain[1].signed
    changed_policy = dataclasses.replace(
        child.payload,
        chain_proof=make_chain_proof_signed_dag(
            co_signers=("D_A", "D_B", "D_A"),
            signatures=child.payload.chain_proof.signatures + ("00" * 64,),
        ),
    )
    child_kp = harness.registry.get_owner("D_B")
    chain[1] = ChainNode(
        signed=dataclasses.replace(
            child,
            payload=changed_policy,
            signature_hex=sign(child_kp.private, changed_policy.to_canonical_json()),
        )
    )
    with pytest.raises(ChainVerifyError, match="signature invalid"):
        backend.verify_chain(chain=chain, deployers=harness.registry)
