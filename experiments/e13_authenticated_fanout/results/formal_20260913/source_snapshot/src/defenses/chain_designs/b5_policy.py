"""B5 Sibling-Fanout Policy Defense.

Defends against B5 (sibling fanout) by enforcing a signed max_children
constraint. The parent's payload declares the maximum number of children
it permits. The verifier maintains a per-parent child count and rejects
any child that would exceed the limit.

This is a POLICY-level defense, not a cryptographic one: the max_children
field is signed (so it cannot be tampered with), but enforcement requires
the verifier to track state across chain presentations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .base import ChainNode, ChainVerifyError


class SiblingFanoutChecker:
    """Stateful checker that enforces max_children across presented chains."""

    def __init__(self):
        self._children_count: dict[str, int] = defaultdict(int)

    def check_chain(self, chain: Sequence[ChainNode]) -> None:
        """Check B5 policy for all nodes in a presented chain."""
        for i in range(1, len(chain)):
            child = chain[i].signed.payload
            parent = chain[i - 1].signed.payload

            if parent.max_children is not None:
                self._children_count[parent.instance_id] += 1
                if self._children_count[parent.instance_id] > parent.max_children:
                    raise ChainVerifyError(
                        f"B5 fanout violation: parent {parent.instance_id!r} "
                        f"has {self._children_count[parent.instance_id]} children, "
                        f"max_children={parent.max_children}"
                    )

    def reset(self):
        """Reset child counters (e.g., between test runs)."""
        self._children_count.clear()
