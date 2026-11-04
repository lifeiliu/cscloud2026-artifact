"""MerkleBackend — root hash over the ordered ancestor list.

Each non-root node carries `merkle_root = merkle_root_of(ancestors_root_to_parent)`.
Verification: for each hop i>=1, the verifier independently computes the
Merkle root of the *presented* ancestors and checks it equals the stored
`chain_proof.merkle_root`.

Properties:
  - Verification cost: O(n) hashes per hop in the simplest implementation
    (we recompute the root for each i). With proof paths this can be O(log n)
    per hop, but we ship the simple path now and optimise in Week 6 if the
    perf baseline exceeds the 30%-of-LLM-turn-latency target.
  - B4 ancestor revision: defended. If any old ancestor is mutated, every
    subsequent merkle_root in the chain becomes invalid simultaneously.
  - B2 chain-break splice: defended. A spliced chain produces a different
    root than the legitimate one.
  - B1 orphan-child injection: defended same as linked_list — the orphan's
    merkle_root references no real parent so it cannot match.

Tree shape: balanced binary, leaves left-to-right in ancestor order
(root-most ancestor at leaf 0). Odd numbers of leaves at any level are
handled by carrying the unpaired leaf up to the next level (Bitcoin-style).
"""

from __future__ import annotations

from typing import Optional, Sequence

from ...core.agent_id_v2 import (
    BoundIDPayloadV2,
    ChainProof,
    make_chain_proof_merkle,
)
from ...core.crypto import sha256_hex
from .base import (
    ChainBackend,
    ChainNode,
    ChainVerifyError,
    DeployerRegistry,
)


def _merkle_root(leaves: Sequence[bytes]) -> str:
    """Compute the SHA-256 Merkle root over `leaves`, returned as hex.

    leaves: ordered iterable of the *raw bytes* of each leaf (we hash here).
    Empty input -> empty string sentinel (root nodes use this).

    Promotion rule for odd levels: the unpaired item is carried up unchanged
    (NOT duplicated like Bitcoin) — this avoids the CVE-2012-2459-style
    second-preimage by keeping the level shape unambiguous.
    """
    if not leaves:
        return ""

    # Hash leaves once.
    layer: list[str] = [sha256_hex(b) for b in leaves]

    while len(layer) > 1:
        next_layer: list[str] = []
        for i in range(0, len(layer) - 1, 2):
            combined = (layer[i] + layer[i + 1]).encode("utf-8")
            next_layer.append(sha256_hex(combined))
        if len(layer) % 2 == 1:
            # Odd: carry the last item unchanged.
            next_layer.append(layer[-1])
        layer = next_layer

    return layer[0]


class MerkleBackend(ChainBackend):
    design_name = "merkle"

    # ---------- Build ----------

    def build_proof_for_root(self) -> ChainProof:
        # Root has no ancestors. Empty merkle_root sentinel "".
        return make_chain_proof_merkle(merkle_root="")

    def build_proof_for_extend(
        self,
        *,
        parent: BoundIDPayloadV2,
        child_instance_id: str,
        ancestors: Sequence[BoundIDPayloadV2],
        deployers: DeployerRegistry,
        co_signer_names: Sequence[str] = (),
        child_payload: Optional[BoundIDPayloadV2] = None,  # unused
    ) -> ChainProof:
        if co_signer_names:
            raise ValueError(
                "merkle does not use co-signers; "
                "use signed_dag for cross-deployer hops"
            )
        # ancestors must be ordered root..parent (inclusive of parent).
        if not ancestors:
            raise ValueError(
                "non-root extend requires ancestors (at minimum, the parent)"
            )
        if ancestors[-1] is not parent and ancestors[-1] != parent:
            raise ValueError(
                "ancestors[-1] must be `parent` (root..parent ordering)"
            )

        leaves = [a.to_canonical_json() for a in ancestors]
        root_hex = _merkle_root(leaves)
        return make_chain_proof_merkle(merkle_root=root_hex)

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
        if root.chain_proof.merkle_root != "":
            raise ChainVerifyError(
                f"root merkle_root must be empty, got "
                f"{root.chain_proof.merkle_root!r}"
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
                raise ChainVerifyError(
                    f"hop {i} parent_id mismatch"
                )
            if child.ancestry_depth != parent.ancestry_depth + 1:
                raise ChainVerifyError(
                    f"hop {i} ancestry_depth={child.ancestry_depth}, "
                    f"expected {parent.ancestry_depth + 1}"
                )

            # Recompute the Merkle root over presented ancestors[0..i-1]
            ancestor_leaves = [
                chain[j].signed.payload.to_canonical_json()
                for j in range(i)
            ]
            expected = _merkle_root(ancestor_leaves)
            if child.chain_proof.merkle_root != expected:
                raise ChainVerifyError(
                    f"hop {i} merkle_root mismatch: "
                    f"got {child.chain_proof.merkle_root!r}, "
                    f"expected {expected!r}"
                )
