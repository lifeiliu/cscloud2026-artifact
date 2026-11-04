"""Lifecycle invariant verifier — Paper 2 §4.

Verifies that the `LifecycleRecord` attached to each ChainNode is
semantically consistent with the node's identity fields and its parent.

This is a *composable* layer alongside `ChainBackend.verify_chain()`:
  backend.verify_chain(chain=chain, deployers=registry)   # chain shape
  LifecycleVerifier().verify(chain=chain)                  # lifecycle

Five lifecycle events, each with per-node invariants:

  L1 new         — root node; default lifecycle fields
  L2 reload      — same instance, same deployer, same model; reload_count >= 1
  L3 branch      — new instance, same model; branch_index >= 1
  L4 composite   — >= 2 constituents; each must be a known instance
  L5 fine_tune   — base_system_id != system_id; provenance to base in chain

Error messages include the invariant ID (e.g. "L2c") and field values so the
Week 5 red-team LLM can read them and adapt its attacks.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core.agent_id_v2 import BoundIDPayloadV2, LifecycleRecord
from .chain_designs.base import ChainBackend, ChainNode, ChainVerifyError, DeployerRegistry


# ---------- Errors ----------

class LifecycleVerifyError(Exception):
    """Raised when a lifecycle invariant is violated.

    Attributes:
        invariant_id: the specific invariant that failed (e.g. "L2c", "L5b")
        node_index:   position in the chain where the violation occurred
    """

    def __init__(self, invariant_id: str, node_index: int, message: str) -> None:
        self.invariant_id = invariant_id
        self.node_index = node_index
        super().__init__(f"{invariant_id}: hop {node_index} — {message}")


# ---------- Verifier ----------

class LifecycleVerifier:
    """Verifies lifecycle invariants across a presented chain.

    Composable with `ChainBackend.verify_chain()`. Use `verify_full()` below
    to run both in one call.

    For composite events, pass auxiliary constituent chains via
    `auxiliary_chains` so the verifier can validate that named constituents
    exist.
    """

    def verify(
        self,
        chain: Sequence[ChainNode],
        *,
        auxiliary_chains: Sequence[Sequence[ChainNode]] = (),
    ) -> None:
        """Verify all lifecycle invariants. Raises on first violation."""
        if not chain:
            raise LifecycleVerifyError("L0", 0, "empty chain")

        # Auxiliary chains are fully trusted (already verified separately).
        aux_ids: set[str] = set()
        aux_system_ids: set[str] = set()
        for aux in auxiliary_chains:
            for node in aux:
                p = node.signed.payload
                aux_ids.add(p.instance_id)
                aux_system_ids.add(p.system_id)

        # Incremental sets: only contain nodes 0..i-1 when checking node i.
        known_ids: set[str] = set(aux_ids)
        chain_system_ids: set[str] = set(aux_system_ids)
        seen_nonces: set[str] = set()

        for i, node in enumerate(chain):
            payload = node.signed.payload
            lc = payload.lifecycle
            parent: Optional[BoundIDPayloadV2] = (
                chain[i - 1].signed.payload if i > 0 else None
            )

            # L6: nonce uniqueness within the chain
            if payload.nonce in seen_nonces:
                raise LifecycleVerifyError(
                    "L6a", i,
                    f"duplicate nonce {payload.nonce!r} "
                    f"(already used by an earlier node)"
                )
            seen_nonces.add(payload.nonce)

            # L5-temporal: timestamp monotonicity
            if parent is not None:
                if payload.created_at < parent.created_at:
                    raise LifecycleVerifyError(
                        "L5t", i,
                        f"created_at {payload.created_at!r} is before "
                        f"parent's created_at {parent.created_at!r}"
                    )

            if lc.event == "new":
                self._verify_new(payload, i)
            elif lc.event == "reload":
                if parent is None:
                    raise LifecycleVerifyError(
                        "L2b", i, "reload must have a parent (cannot be root)"
                    )
                self._verify_reload(payload, i, parent)
            elif lc.event == "branch":
                if parent is None:
                    raise LifecycleVerifyError(
                        "L3b", i, "branch must have a parent (cannot be root)"
                    )
                self._verify_branch(payload, i, parent)
            elif lc.event == "composite":
                self._verify_composite(payload, i, known_ids)
            elif lc.event == "fine_tune":
                self._verify_fine_tune(payload, i, chain_system_ids)
            else:
                raise LifecycleVerifyError(
                    "L0", i, f"unknown lifecycle event {lc.event!r}"
                )

            # Add current node AFTER its own checks pass (incremental build).
            known_ids.add(payload.instance_id)
            chain_system_ids.add(payload.system_id)

    # ---- Per-event checks ----

    def _verify_new(self, p: BoundIDPayloadV2, idx: int) -> None:
        """L1: new event — fields must be at defaults."""
        lc = p.lifecycle
        if idx == 0:
            # Root node: structural constraints
            if p.parent_id is not None:
                raise LifecycleVerifyError(
                    "L1b", idx, f"new root must have parent_id=None, got {p.parent_id!r}"
                )
            if p.ancestry_depth != 0:
                raise LifecycleVerifyError(
                    "L1a", idx, f"new root must have ancestry_depth=0, got {p.ancestry_depth}"
                )
        # Event-specific field defaults (apply to all new nodes)
        if lc.reload_count != 0:
            raise LifecycleVerifyError(
                "L1c", idx, f"new must have reload_count=0, got {lc.reload_count}"
            )
        if lc.branch_index != 0:
            raise LifecycleVerifyError(
                "L1d", idx, f"new must have branch_index=0, got {lc.branch_index}"
            )
        if lc.constituents:
            raise LifecycleVerifyError(
                "L1e", idx,
                f"new must have no constituents, got {lc.constituents!r}"
            )
        if lc.base_system_id is not None:
            raise LifecycleVerifyError(
                "L1f", idx,
                f"new must have base_system_id=None, got {lc.base_system_id!r}"
            )

    def _verify_reload(
        self, p: BoundIDPayloadV2, idx: int, parent: BoundIDPayloadV2
    ) -> None:
        """L2: reload — same identity, incremented reload_count."""
        lc = p.lifecycle
        if lc.reload_count < 1:
            raise LifecycleVerifyError(
                "L2a", idx,
                f"reload must have reload_count>=1, got {lc.reload_count}"
            )
        if p.instance_id != parent.instance_id:
            raise LifecycleVerifyError(
                "L2c", idx,
                f"reload must preserve instance_id, "
                f"expected {parent.instance_id!r}, got {p.instance_id!r}"
            )
        if p.system_id != parent.system_id:
            raise LifecycleVerifyError(
                "L2d", idx,
                f"reload must preserve system_id, "
                f"expected {parent.system_id!r}, got {p.system_id!r}"
            )
        if p.deployer != parent.deployer:
            raise LifecycleVerifyError(
                "L2e", idx,
                f"reload must preserve deployer, "
                f"expected {parent.deployer!r}, got {p.deployer!r}"
            )
        if lc.branch_index != 0:
            raise LifecycleVerifyError(
                "L2f", idx,
                f"reload must have branch_index=0, got {lc.branch_index}"
            )

    def _verify_branch(
        self, p: BoundIDPayloadV2, idx: int, parent: BoundIDPayloadV2
    ) -> None:
        """L3: branch — new instance, same model family."""
        lc = p.lifecycle
        if lc.branch_index < 1:
            raise LifecycleVerifyError(
                "L3a", idx,
                f"branch must have branch_index>=1, got {lc.branch_index}"
            )
        if p.instance_id == parent.instance_id:
            raise LifecycleVerifyError(
                "L3c", idx,
                f"branch must have a new instance_id, "
                f"got same as parent {parent.instance_id!r}"
            )
        if p.system_id != parent.system_id:
            raise LifecycleVerifyError(
                "L3d", idx,
                f"branch must preserve system_id, "
                f"expected {parent.system_id!r}, got {p.system_id!r}"
            )

    def _verify_composite(
        self, p: BoundIDPayloadV2, idx: int, known_ids: set[str]
    ) -> None:
        """L4: composite — merge of >= 2 known constituents."""
        lc = p.lifecycle
        if len(lc.constituents) < 2:
            raise LifecycleVerifyError(
                "L4a", idx,
                f"composite must have >=2 constituents, got {len(lc.constituents)}"
            )
        for cid in lc.constituents:
            if cid not in known_ids:
                raise LifecycleVerifyError(
                    "L4b", idx,
                    f"composite constituent {cid!r} not found in "
                    f"presented chain or auxiliaries"
                )

    def _verify_fine_tune(
        self, p: BoundIDPayloadV2, idx: int, chain_system_ids: set[str]
    ) -> None:
        """L5: fine_tune — provenance to base model in chain."""
        lc = p.lifecycle
        if not lc.base_system_id:
            raise LifecycleVerifyError(
                "L5a", idx, "fine_tune must have non-empty base_system_id"
            )
        if lc.base_system_id == p.system_id:
            raise LifecycleVerifyError(
                "L5b", idx,
                f"fine_tune base_system_id must differ from system_id, "
                f"both are {p.system_id!r}"
            )
        if lc.base_system_id not in chain_system_ids:
            raise LifecycleVerifyError(
                "L5c", idx,
                f"fine_tune base_system_id {lc.base_system_id!r} not found "
                f"as any node's system_id in the chain"
            )


# ---------- Convenience: combined verification ----------

def verify_full(
    chain: Sequence[ChainNode],
    backend: ChainBackend,
    deployers: DeployerRegistry,
    *,
    auxiliary_chains: Sequence[Sequence[ChainNode]] = (),
) -> None:
    """Run both chain-shape and lifecycle verification.

    Calls backend.verify_chain() first (structural), then
    LifecycleVerifier.verify() (semantic). Either may raise.
    """
    backend.verify_chain(chain=chain, deployers=deployers)
    LifecycleVerifier().verify(chain=chain, auxiliary_chains=auxiliary_chains)
