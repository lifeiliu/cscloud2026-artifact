"""Focused E12 receipt-policy checks; the live protocol path is run by E12."""

from __future__ import annotations

from dataclasses import replace

from src.core.agent_id_v2 import ChainProof, make_chain_proof_signed_dag, make_lifecycle_new
from src.core.crypto import generate_keypair, sign
from src.defenses.chain_designs import ChainNode, ChainVerifyError, LinkedListBackend, SignedDAGBackend
from src.defenses.chain_designs.base import edge_signing_input
from src.defenses.complete_chain_verifier import verify_complete_chain
from src.runtime.multi_deployer import ChainBuilder, MultiDeployerHarness


def test_e12_child_key_only_claim_accepts_for_linked_list_and_rejects_for_signed_dag() -> None:
    """The E12 live attack has the same policy distinction as its protocol proof."""
    parent_key, child_key = generate_keypair(), generate_keypair()
    harness = MultiDeployerHarness({"deployer_parent": parent_key, "deployer_child": child_key})

    linked = ChainBuilder(harness, LinkedListBackend())
    linked_root = linked.start_root("deployer_parent", "parent-linked", "e12", b"parent")
    linked_claim = linked.extend(
        linked_root,
        host="deployer_child",
        instance_id="child-linked-claim",
        system_id="e12",
        output=b"unapproved child claim",
    )
    verify_complete_chain(
        chain=linked_claim,
        outputs=(b"parent", b"unapproved child claim"),
        backend=LinkedListBackend(),
        deployers=harness.registry,
    )

    dag = ChainBuilder(harness, SignedDAGBackend())
    dag_root = dag.start_root("deployer_parent", "parent-dag", "e12", b"parent")
    child = harness.get_deployer("deployer_child")
    provisional = child.build_payload(
        instance_id="child-dag-claim",
        system_id="e12",
        output_bytes=b"unapproved child claim",
        binding_scope="per_session",
        chain_proof=ChainProof(design="signed_dag", co_signers=("deployer_parent", "deployer_child")),
        lifecycle=make_lifecycle_new(),
        parent_id=dag_root[-1].signed.payload.instance_id,
        ancestry_depth=1,
    )
    child_signature = sign(child.keypair.private, edge_signing_input(dag_root[-1].signed.payload, provisional))
    forged = child.sign_payload(
        replace(
            provisional,
            chain_proof=make_chain_proof_signed_dag(
                ("deployer_parent", "deployer_child"), ("00" * 64, child_signature)
            ),
        )
    )
    try:
        verify_complete_chain(
            chain=(dag_root[-1], ChainNode(forged)),
            outputs=(b"parent", b"unapproved child claim"),
            backend=SignedDAGBackend(),
            deployers=harness.registry,
        )
    except ChainVerifyError as error:
        assert "deployer_parent" in str(error)
    else:  # pragma: no cover - a passing forged edge is a safety regression.
        raise AssertionError("child-only Signed-DAG claim was accepted")
