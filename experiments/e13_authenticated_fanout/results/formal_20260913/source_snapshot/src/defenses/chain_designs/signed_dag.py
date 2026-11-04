"""SignedDAGBackend — deployer cross-signatures on each edge.

For a hop where parent's deployer = D_A and child's deployer = D_B (possibly
the same): both D_A and D_B (and any additional configured co-signers) sign a
domain-separated canonical edge manifest. It binds the parent payload digest
and the complete child payload except the edge-signature list itself. Each
signature is checked by the verifier against the deployer's public key in the
`DeployerRegistry`.

Properties:
  - B3 cross-chain merge: defended. An attacker cannot graft a node from
    chain X into chain Y without forging the destination chain's deployer
    signatures on the new edge.
  - B1 orphan-child injection: defended. The orphan would need a forged
    co-signer signature whose key the verifier doesn't possess.
  - B2 chain-break splice: defended. Splicing requires forging signatures
    on the new (false) edge.
  - B4 ancestor revision: PARTIALLY defended. An old ancestor can be
    rewritten only if all subsequent edges are re-signed by their original
    co-signers — equivalent to compromising those deployers. We treat this
    as defended under the Paper 1 threat model (deployer keys are
    out-of-scope-to-compromise).

Cost: highest of the three designs. Each hop carries N signatures where N is
the co-signer count (typically 2 for cross-deployer, 1 for same-deployer);
verification is N Ed25519 verifies per hop.

Co-signer convention:
  - For same-deployer hops, co_signer_names = (parent.deployer,).
  - For cross-deployer hops, co_signer_names = (parent.deployer, child.deployer).
  - Other policies (e.g. always include a third-party witness) are accepted
    as long as build_proof_for_extend and verify_chain agree on the list.
"""

from __future__ import annotations

from typing import Sequence

from ...core.agent_id_v2 import (
    BoundIDPayloadV2,
    ChainProof,
    make_chain_proof_signed_dag,
)
from ...core.crypto import sign, verify
from .base import (
    ChainBackend,
    ChainExtendError,
    ChainNode,
    ChainVerifyError,
    DeployerRegistry,
    edge_signing_input,
)


class SignedDAGBackend(ChainBackend):
    design_name = "signed_dag"

    # ---------- Build ----------

    def build_proof_for_root(self) -> ChainProof:
        # Root has no incoming edge: empty co_signers / signatures.
        return make_chain_proof_signed_dag(co_signers=(), signatures=())

    def build_proof_for_extend(
        self,
        *,
        parent: BoundIDPayloadV2,
        child_instance_id: str,
        ancestors: Sequence[BoundIDPayloadV2],  # unused — only parent matters
        deployers: DeployerRegistry,
        co_signer_names: Sequence[str] = (),
        child_payload: BoundIDPayloadV2 | None = None,
    ) -> ChainProof:
        if not co_signer_names:
            raise ChainExtendError(
                "signed_dag requires at least one co_signer "
                "(typically the parent's deployer)"
            )

        if child_payload is None:
            raise ChainExtendError(
                "signed_dag requires a provisional child_payload so that "
                "the co-signed edge binds the full child transcript"
            )
        if child_payload.instance_id != child_instance_id:
            raise ChainExtendError(
                "child_payload.instance_id must match child_instance_id"
            )
        if child_payload.parent_id != parent.instance_id:
            raise ChainExtendError(
                "child_payload.parent_id must name the supplied parent"
            )
        if child_payload.ancestry_depth != parent.ancestry_depth + 1:
            raise ChainExtendError(
                "child_payload.ancestry_depth must extend the supplied parent"
            )
        if child_payload.chain_proof.design != self.design_name:
            raise ChainExtendError(
                "child_payload must carry a signed_dag chain proof"
            )
        if tuple(child_payload.chain_proof.co_signers) != tuple(co_signer_names):
            raise ChainExtendError(
                "child_payload co_signers must match the requested co-signers"
            )

        message = edge_signing_input(parent, child_payload)
        sigs: list[str] = []
        for name in co_signer_names:
            kp = deployers.get_owner(name)
            if kp is None:
                raise ChainExtendError(
                    f"co_signer {name!r} is not a registered owner "
                    f"in this DeployerRegistry"
                )
            sigs.append(sign(kp.private, message))

        return make_chain_proof_signed_dag(
            co_signers=tuple(co_signer_names),
            signatures=tuple(sigs),
        )

    # ---------- Verify ----------

    def verify_chain(
        self,
        *,
        chain: Sequence[ChainNode],
        deployers: DeployerRegistry,
    ) -> None:
        if not chain:
            raise ChainVerifyError("empty chain")

        # Root checks.
        root = chain[0].signed.payload
        if root.chain_proof.design != self.design_name:
            raise ChainVerifyError(
                f"root chain_proof.design={root.chain_proof.design!r}, "
                f"backend={self.design_name!r}"
            )
        if root.parent_id is not None:
            raise ChainVerifyError("root must have parent_id=None")
        if root.ancestry_depth != 0:
            raise ChainVerifyError(
                f"root ancestry_depth={root.ancestry_depth}, expected 0"
            )
        if root.chain_proof.co_signers or root.chain_proof.signatures:
            raise ChainVerifyError(
                "root must have empty co_signers and signatures"
            )

        # Per-hop checks.
        for i in range(1, len(chain)):
            child = chain[i].signed.payload
            parent = chain[i - 1].signed.payload

            if child.chain_proof.design != self.design_name:
                raise ChainVerifyError(
                    f"hop {i} chain_proof.design={child.chain_proof.design!r}, "
                    f"backend={self.design_name!r}"
                )
            if child.parent_id != parent.instance_id:
                raise ChainVerifyError(f"hop {i} parent_id mismatch")
            if child.ancestry_depth != parent.ancestry_depth + 1:
                raise ChainVerifyError(
                    f"hop {i} ancestry_depth={child.ancestry_depth}, "
                    f"expected {parent.ancestry_depth + 1}"
                )

            co_signers = child.chain_proof.co_signers
            signatures = child.chain_proof.signatures
            if not co_signers:
                raise ChainVerifyError(
                    f"hop {i} has no co_signers; signed_dag requires >=1"
                )
            if len(co_signers) != len(signatures):
                raise ChainVerifyError(
                    f"hop {i} co_signers ({len(co_signers)}) != "
                    f"signatures ({len(signatures)})"
                )

            # Policy: parent.deployer MUST be among co_signers (otherwise the
            # parent's deployer never authorised this delegation).
            if parent.deployer not in co_signers:
                raise ChainVerifyError(
                    f"hop {i} parent's deployer {parent.deployer!r} "
                    f"is not in co_signers {co_signers}"
                )

            # For cross-deployer hops, child.deployer MUST also be among
            # co_signers (the child's deployer must agree to host the agent).
            if (
                child.deployer != parent.deployer
                and child.deployer not in co_signers
            ):
                raise ChainVerifyError(
                    f"hop {i} cross-deployer hop missing child deployer "
                    f"{child.deployer!r} in co_signers {co_signers}"
                )

            message = edge_signing_input(parent, child)
            for name, sig_hex in zip(co_signers, signatures):
                pub = deployers.get_public(name)
                if pub is None:
                    raise ChainVerifyError(
                        f"hop {i} co_signer {name!r} has no registered "
                        f"public key"
                    )
                if not verify(pub, message, sig_hex):
                    raise ChainVerifyError(
                        f"hop {i} co_signer {name!r} signature invalid"
                    )
