"""B1 — Orphan-child injection.

Threat: attacker presents a child node whose `parent_id` references an
instance_id that was never issued. A naive verifier that checks each hop in
isolation (signature ok, expiry ok) but does not validate parent existence
would accept this.

Defence surface:
  - linked_list : `prev_hash` must equal SHA-256 of an actually-presented parent.
                  An orphan presents no real parent, so prev_hash either points
                  at a fabricated payload (signature fails) or a missing one
                  (verify_chain detects parent_id mismatch with chain[i-1]).
  - merkle      : same — `merkle_root` over a fabricated ancestor list does
                  not match what the verifier recomputes from the *presented*
                  ancestors. If the attacker presents ONLY the orphan as a
                  single-hop "chain", ancestry_depth=1 contradicts the
                  presented depth (which would be 1 with no actual ancestor
                  preceding it -> mismatch).
  - signed_dag  : the attacker has no parent deployer's signature on the
                  fabricated edge.

Attack variants:
  fabricated_parent — replace the root with a completely fabricated payload.
                      The child's chain_proof still references the ORIGINAL
                      root, so the hash / merkle / signature link breaks.
  missing_parent   — drop the root node entirely, present child-only chain
                      with ancestry_depth=1. Verifier should catch the
                      length mismatch or root-check failure.
  ghost_id         — keep the chain structure but mutate the root's
                      instance_id to a random value. The child's parent_id
                      no longer matches chain[0].instance_id.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Literal, Sequence

from ...core.agent_id_v2 import BoundIDPayloadV2
from ...core.crypto import sha256_hex
from ...defenses.chain_designs.base import ChainNode, SignedBoundIDV2


def attack_b1_orphan_injection(
    chain: Sequence[ChainNode],
    *,
    variant: Literal["fabricated_parent", "missing_parent", "ghost_id"] = "fabricated_parent",
) -> list[ChainNode]:
    """Produce a mutated chain that injects an orphan child.

    Args:
        chain: a benign chain of length >= 2 (root + at least one child).
        variant: which orphan-injection strategy to use.

    Returns:
        Mutated chain ready to feed to verify_chain. The attack SHOULD fail
        against all three chain designs.
    """
    if len(chain) < 2:
        raise ValueError("B1 requires chain length >= 2 (root + child)")

    mutated = list(chain)

    if variant == "fabricated_parent":
        # Replace root with a fabricated payload. The child at index 1 still
        # has chain_proof referencing the ORIGINAL root's hash/merkle/sigs.
        old_root = chain[0].signed.payload
        fabricated = dataclasses.replace(
            old_root,
            instance_id=f"fabricated-{uuid.uuid4().hex[:8]}",
            output_hash=sha256_hex(b"fabricated output"),
        )
        mutated[0] = ChainNode(
            signed=dataclasses.replace(
                chain[0].signed,
                payload=fabricated,
            )
        )

    elif variant == "missing_parent":
        # Drop the root entirely. Present chain[1:] as if ancestry_depth
        # starts at 0. But chain[1]'s payload has ancestry_depth=1 and
        # parent_id pointing at the (now absent) root.
        mutated = list(chain[1:])

    elif variant == "ghost_id":
        # Keep the root payload intact except mutate its instance_id.
        # The child's parent_id still references the ORIGINAL instance_id,
        # creating a parent_id mismatch at hop 1.
        old_root = chain[0].signed.payload
        ghosted = dataclasses.replace(
            old_root,
            instance_id=f"ghost-{uuid.uuid4().hex[:8]}",
        )
        mutated[0] = ChainNode(
            signed=dataclasses.replace(
                chain[0].signed,
                payload=ghosted,
            )
        )

    else:
        raise ValueError(f"unknown B1 variant: {variant!r}")

    return mutated
