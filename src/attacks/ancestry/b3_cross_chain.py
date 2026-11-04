"""B3 — Cross-chain merge.

Threat: attacker has access to a node N from a legitimate chain X belonging
to user_X, and grafts N into chain Y belonging to user_Y. The forged Y'
chain claims user_Y's authority extends through N's downstream actions.

Defence surface:
  - linked_list : weak. If N's payload was already signed, attacker can
                  only attach N at a position where prev_hash matches —
                  unlikely but not impossible if attacker controls timing.
                  Detection relies on parent_id mismatch.
  - merkle      : same — N's stored `merkle_root` references X's ancestors,
                  which the verifier sees as inconsistent with Y's presented
                  ancestors at that hop.
  - signed_dag  : strongest. The edge from Y's prefix into N requires N's
                  deployer to co-sign FOR Y's parent (different
                  edge_signing_input from any edge co-signed for X).
                  Without that co-signature, the graft is rejected.

This is the attack where signed_dag earns its overhead.

Attack variants:
  graft_at_leaf — replace the last node in chain Y with a node from chain X.
  graft_at_mid  — replace a middle node in chain Y with a node from chain X.
  append_alien  — append a node from chain X as a new leaf on chain Y,
                  patching parent_id to point at Y's current leaf.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

from ...defenses.chain_designs.base import ChainNode


def attack_b3_cross_chain_merge(
    chain_y: Sequence[ChainNode],
    chain_x: Sequence[ChainNode],
    *,
    variant: Literal["graft_at_leaf", "graft_at_mid", "append_alien"] = "graft_at_leaf",
    graft_index_x: int = -1,
) -> list[ChainNode]:
    """Graft a node from chain_x into chain_y.

    Args:
        chain_y: the victim chain. Must have length >= 2.
        chain_x: the source chain (different deployer / lineage). Must be non-empty.
        variant: graft strategy.
        graft_index_x: which node from chain_x to graft (default: last).

    Returns:
        Mutated chain_y with a foreign node. All three designs should detect
        this, with signed_dag catching it most robustly.
    """
    if len(chain_y) < 2:
        raise ValueError("B3 requires chain_y length >= 2")
    if not chain_x:
        raise ValueError("B3 requires chain_x to be non-empty")

    y = list(chain_y)
    x_node = chain_x[graft_index_x]

    if variant == "graft_at_leaf":
        # Replace Y's leaf with X's node, patching parent_id to look connected.
        parent_payload = y[-2].signed.payload
        alien_payload = dataclasses.replace(
            x_node.signed.payload,
            parent_id=parent_payload.instance_id,
            ancestry_depth=parent_payload.ancestry_depth + 1,
        )
        y[-1] = ChainNode(
            signed=dataclasses.replace(x_node.signed, payload=alien_payload)
        )

    elif variant == "graft_at_mid":
        mid = len(y) // 2
        if mid == 0:
            mid = 1  # don't replace root
        parent_payload = y[mid - 1].signed.payload
        alien_payload = dataclasses.replace(
            x_node.signed.payload,
            parent_id=parent_payload.instance_id,
            ancestry_depth=parent_payload.ancestry_depth + 1,
        )
        y[mid] = ChainNode(
            signed=dataclasses.replace(x_node.signed, payload=alien_payload)
        )
        # Also patch the node AFTER the graft point so parent_id links correctly
        if mid + 1 < len(y):
            next_payload = y[mid + 1].signed.payload
            patched = dataclasses.replace(
                next_payload,
                parent_id=alien_payload.instance_id,
            )
            y[mid + 1] = ChainNode(
                signed=dataclasses.replace(y[mid + 1].signed, payload=patched)
            )

    elif variant == "append_alien":
        # Append X's node as a new leaf on Y, patching parent_id.
        leaf_payload = y[-1].signed.payload
        alien_payload = dataclasses.replace(
            x_node.signed.payload,
            parent_id=leaf_payload.instance_id,
            ancestry_depth=leaf_payload.ancestry_depth + 1,
        )
        y.append(ChainNode(
            signed=dataclasses.replace(x_node.signed, payload=alien_payload)
        ))

    else:
        raise ValueError(f"unknown B3 variant: {variant!r}")

    return y
