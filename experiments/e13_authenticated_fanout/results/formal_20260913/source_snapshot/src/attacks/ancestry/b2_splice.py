"""B2 — Chain-break splice.

Threat: attacker removes a middle node from an N-hop chain and rewires the
endpoints, presenting an (N-1)-hop chain that elides the missing node. If
the verifier only checks signatures + expiry per node, the splice goes
undetected.

Defence surface:
  - linked_list : the spliced child's `prev_hash` was computed against the
                  removed parent; it cannot match the new (presented) parent.
  - merkle      : root recomputed over the presented ancestor list differs
                  from the stored `merkle_root` (which referenced the elided
                  ancestor).
  - signed_dag  : the spliced edge was never co-signed by the new parent's
                  deployer; signature verification on the rewired edge fails.

Attack variants:
  root_adjacent  — remove index 1 (the first non-root node).
  middle         — remove the node at len(chain)//2.
  leaf_adjacent  — remove index -2 (the node just before the leaf).

In all variants the attacker fixes `ancestry_depth` and `parent_id` on the
nodes after the splice point to look consistent, but CANNOT fix chain_proof
(that's what the defence checks).
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

from ...defenses.chain_designs.base import ChainNode


def attack_b2_chain_break_splice(
    chain: Sequence[ChainNode],
    *,
    variant: Literal["root_adjacent", "middle", "leaf_adjacent"] = "middle",
) -> list[ChainNode]:
    """Remove one middle node and patch parent_id / ancestry_depth.

    The chain_proof fields (prev_hash, merkle_root, co-signer sigs) are
    left untouched — the attacker does not have the deployer's private key
    to re-sign them. This is the realistic scenario.

    Args:
        chain: a benign chain of length >= 3 (need at least one true middle node).
        variant: which node to remove.

    Returns:
        Spliced chain with one fewer node. All three designs should detect this.
    """
    if len(chain) < 3:
        raise ValueError("B2 requires chain length >= 3 (need a middle node to remove)")

    chain = list(chain)

    # Pick the index to remove
    if variant == "root_adjacent":
        remove_idx = 1
    elif variant == "middle":
        remove_idx = len(chain) // 2
    elif variant == "leaf_adjacent":
        remove_idx = len(chain) - 2
    else:
        raise ValueError(f"unknown B2 variant: {variant!r}")

    # Remove the target node
    spliced = chain[:remove_idx] + chain[remove_idx + 1:]

    # Patch parent_id and ancestry_depth on all nodes after the splice point
    # so they LOOK consistent. The chain_proof fields are left broken.
    for i in range(remove_idx, len(spliced)):
        old_payload = spliced[i].signed.payload
        new_parent_id = (
            spliced[i - 1].signed.payload.instance_id if i > 0 else None
        )
        new_depth = i  # root=0, next=1, ...

        patched = dataclasses.replace(
            old_payload,
            parent_id=new_parent_id,
            ancestry_depth=new_depth,
        )
        spliced[i] = ChainNode(
            signed=dataclasses.replace(
                spliced[i].signed,
                payload=patched,
            )
        )

    return spliced
