"""LinkedListBackend — per-hop signature over the parent's canonical payload.

Properties:
  - Verification cost: O(depth) — each node hashed once, sig checked once.
  - B1 orphan-child injection: defended by `prev_hash` consistency check
    (an orphan has no preceding node to hash against).
  - B2 chain-break splice: defended by `prev_hash` continuity — a spliced
    chain breaks the hash chain at the splice point.
  - B4 ancestor revision: defended under the non-compromise model
    (prev_hash mismatch at the next hop); NOT defended under key
    compromise (attacker can re-sign downstream nodes).

The chain_proof for this design carries `prev_hash` only:
    chain_proof.prev_hash == sha256_hex(parent.to_canonical_json())  if non-root
    chain_proof.prev_hash is None                                    if root
"""

from __future__ import annotations

from typing import Optional, Sequence

from ...core.agent_id_v2 import (
    BoundIDPayloadV2,
    ChainProof,
    make_chain_proof_linked_list,
)
from ...core.crypto import sha256_hex
from .base import (
    ChainBackend,
    ChainNode,
    ChainVerifyError,
    DeployerRegistry,
)


class LinkedListBackend(ChainBackend):
    design_name = "linked_list"

    # ---------- Build ----------

    def build_proof_for_root(self) -> ChainProof:
        return make_chain_proof_linked_list(prev_hash=None)

    def build_proof_for_extend(
        self,
        *,
        parent: BoundIDPayloadV2,
        child_instance_id: str,  # unused for linked_list; signature is on parent payload via primary sig
        ancestors: Sequence[BoundIDPayloadV2],  # unused; only `parent` matters
        deployers: DeployerRegistry,  # unused; primary deployer signature handles authentication
        co_signer_names: Sequence[str] = (),  # must be empty
        child_payload: Optional[BoundIDPayloadV2] = None,  # unused
    ) -> ChainProof:
        if co_signer_names:
            raise ValueError(
                "linked_list does not use co-signers; "
                "use signed_dag for cross-deployer hops"
            )
        prev_hash = sha256_hex(parent.to_canonical_json())
        return make_chain_proof_linked_list(prev_hash=prev_hash)

    # ---------- Verify ----------

    def verify_chain(
        self,
        *,
        chain: Sequence[ChainNode],
        deployers: DeployerRegistry,  # unused for linked_list
    ) -> None:
        if not chain:
            raise ChainVerifyError("empty chain")

        # Root node: prev_hash MUST be None and design MUST match.
        root = chain[0].signed.payload
        if root.chain_proof.design != self.design_name:
            raise ChainVerifyError(
                f"root chain_proof.design={root.chain_proof.design!r}, "
                f"backend={self.design_name!r}"
            )
        if root.chain_proof.prev_hash is not None:
            raise ChainVerifyError("root must have prev_hash=None")
        if root.parent_id is not None:
            raise ChainVerifyError("root must have parent_id=None")
        if root.ancestry_depth != 0:
            raise ChainVerifyError(
                f"root ancestry_depth={root.ancestry_depth}, expected 0"
            )

        # Each subsequent hop must hash-link to its predecessor.
        for i in range(1, len(chain)):
            child = chain[i].signed.payload
            parent = chain[i - 1].signed.payload

            if child.chain_proof.design != self.design_name:
                raise ChainVerifyError(
                    f"hop {i} chain_proof.design={child.chain_proof.design!r}, "
                    f"backend={self.design_name!r}"
                )
            if child.parent_id != parent.instance_id:
                raise ChainVerifyError(
                    f"hop {i} parent_id={child.parent_id!r} does not match "
                    f"presented parent instance_id={parent.instance_id!r}"
                )
            if child.ancestry_depth != parent.ancestry_depth + 1:
                raise ChainVerifyError(
                    f"hop {i} ancestry_depth={child.ancestry_depth}, "
                    f"expected {parent.ancestry_depth + 1}"
                )

            expected = sha256_hex(parent.to_canonical_json())
            if child.chain_proof.prev_hash != expected:
                raise ChainVerifyError(
                    f"hop {i} prev_hash mismatch: "
                    f"got {child.chain_proof.prev_hash!r}, expected {expected!r}"
                )
