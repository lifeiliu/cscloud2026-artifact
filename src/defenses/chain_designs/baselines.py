"""Baseline chain backends for Paper 2 §6 ablation study (E2).

Three deliberately weak backends that demonstrate what happens WITHOUT
the chain integrity mechanisms tested in the main experiment:

  no_chain   — Accept any chain unconditionally. No cryptographic checks.
               Expected ASR ≈ 100% for B1–B4 (attacks trivially bypass).
  hash_only  — Verify only output_hash consistency (each node's output_hash
               is present and non-empty). No prev_hash, no merkle, no sigs.
               Expected: B1 (orphan) and B4 (ancestor revision on non-hash
               fields) should succeed; B2 splice may partially succeed.
  naive_sig  — Verify each node's primary Ed25519 signature individually,
               but do NOT check inter-node hash continuity. This models the
               common "just sign each payload" deployment pattern.
               Expected: attacks that tamper with unsigned chain-shape
               relations (B1 parent_id mismatch, B2 splice, B3 graft)
               succeed because the verifier doesn't check chain coherence.

These baselines answer the reviewer question: "Are your attacks actually
meaningful, or is ASR=0 because the attacks are weak?" By showing high ASR
on unprotected systems, we establish that the chain designs are the defence,
not an artifact of weak attacks.

Anti-pattern note: these backends are intentionally insecure. Do not use
them outside the ablation study.
"""

from __future__ import annotations

from typing import Sequence

from ...core.agent_id_v2 import (
    BoundIDPayloadV2,
    ChainProof,
)
from ...core.crypto import sha256_hex, verify
from .base import (
    ChainBackend,
    ChainNode,
    ChainVerifyError,
    DeployerRegistry,
)


# ============================================================================
# no_chain — accept everything
# ============================================================================

class NoChainBackend(ChainBackend):
    """No verification at all. Accepts any chain unconditionally.

    This is the "no defence" baseline. The only check is that the chain
    is non-empty, so the experiment harness doesn't degenerate.
    """

    design_name = "no_chain"

    def build_proof_for_root(self) -> ChainProof:
        return ChainProof(design="no_chain")

    def build_proof_for_extend(
        self,
        *,
        parent: BoundIDPayloadV2,
        child_instance_id: str,
        ancestors: Sequence[BoundIDPayloadV2],
        deployers: DeployerRegistry,
        co_signer_names: Sequence[str] = (),
    ) -> ChainProof:
        return ChainProof(design="no_chain")

    def verify_chain(
        self,
        *,
        chain: Sequence[ChainNode],
        deployers: DeployerRegistry,
    ) -> None:
        if not chain:
            raise ChainVerifyError("empty chain")
        # Accept everything — no structural checks.


# ============================================================================
# hash_only — verify output_hash presence but no chain continuity
# ============================================================================

class HashOnlyBackend(ChainBackend):
    """Verify only that each node has a non-empty output_hash.

    Does NOT check:
      - prev_hash continuity (B1, B2 bypass)
      - parent_id / ancestry_depth consistency (B1, B2, B3 bypass)
      - merkle roots (B4 bypass on non-hash fields)
      - signatures on edges (B3 bypass)

    Does check:
      - output_hash is present and non-empty on every node
      - chain_proof.design matches
    """

    design_name = "hash_only"

    def build_proof_for_root(self) -> ChainProof:
        return ChainProof(design="hash_only")

    def build_proof_for_extend(
        self,
        *,
        parent: BoundIDPayloadV2,
        child_instance_id: str,
        ancestors: Sequence[BoundIDPayloadV2],
        deployers: DeployerRegistry,
        co_signer_names: Sequence[str] = (),
    ) -> ChainProof:
        # Store prev_hash for benign chain building, but verify_chain
        # won't check it — that's the point of this weak baseline.
        prev_hash = sha256_hex(parent.to_canonical_json())
        return ChainProof(design="hash_only", prev_hash=prev_hash)

    def verify_chain(
        self,
        *,
        chain: Sequence[ChainNode],
        deployers: DeployerRegistry,
    ) -> None:
        if not chain:
            raise ChainVerifyError("empty chain")

        for i, node in enumerate(chain):
            payload = node.signed.payload
            if payload.chain_proof.design != self.design_name:
                raise ChainVerifyError(
                    f"hop {i} chain_proof.design={payload.chain_proof.design!r}, "
                    f"backend={self.design_name!r}"
                )
            if not payload.output_hash:
                raise ChainVerifyError(
                    f"hop {i} has empty output_hash"
                )
        # No parent_id, ancestry_depth, or prev_hash checks.


# ============================================================================
# naive_sig — verify per-node signatures but no chain-shape integrity
# ============================================================================

class NaiveSigBackend(ChainBackend):
    """Verify each node's primary Ed25519 signature, but skip chain checks.

    This models the common deployment pattern where each agent payload is
    individually signed, but there is no chain-of-custody verification
    between nodes. The verifier trusts that if each node was signed by a
    known deployer, the chain is legitimate.

    Does check:
      - Each node's primary signature is valid against its deployer's public key
      - chain_proof.design matches

    Does NOT check:
      - prev_hash continuity
      - parent_id / ancestry_depth consistency
      - Cross-deployer co-signatures
      - Merkle roots

    Expected vulnerabilities:
      - B1: Orphan child injection — a node can claim any parent_id
      - B2: Chain-break splice — removing middle nodes goes undetected
      - B3: Cross-chain merge — grafting nodes across chains works
      - B4: Ancestor revision — if the attacker can re-sign, revision works
    """

    design_name = "naive_sig"

    def build_proof_for_root(self) -> ChainProof:
        return ChainProof(design="naive_sig")

    def build_proof_for_extend(
        self,
        *,
        parent: BoundIDPayloadV2,
        child_instance_id: str,
        ancestors: Sequence[BoundIDPayloadV2],
        deployers: DeployerRegistry,
        co_signer_names: Sequence[str] = (),
    ) -> ChainProof:
        return ChainProof(design="naive_sig")

    def verify_chain(
        self,
        *,
        chain: Sequence[ChainNode],
        deployers: DeployerRegistry,
    ) -> None:
        if not chain:
            raise ChainVerifyError("empty chain")

        for i, node in enumerate(chain):
            payload = node.signed.payload
            if payload.chain_proof.design != self.design_name:
                raise ChainVerifyError(
                    f"hop {i} chain_proof.design={payload.chain_proof.design!r}, "
                    f"backend={self.design_name!r}"
                )

            # Verify primary signature against deployer's public key
            pub = deployers.get_public(payload.deployer)
            if pub is None:
                raise ChainVerifyError(
                    f"hop {i} deployer {payload.deployer!r} not in registry"
                )
            if not verify(pub, payload.to_canonical_json(), node.signed.signature_hex):
                raise ChainVerifyError(
                    f"hop {i} primary signature verification failed "
                    f"for deployer {payload.deployer!r}"
                )
        # No chain-shape checks: parent_id, ancestry_depth, prev_hash all unchecked.
