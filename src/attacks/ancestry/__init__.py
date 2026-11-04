"""Paper 2 — Ancestry-chain attack family (B1–B5).

Each attack is a callable that takes a benign chain plus context and returns
an *adversarially mutated* chain. Attacks must NOT modify the verifier or
backend; they only manipulate the wire-format chain that a verifier will see.

Week 2 will fill in concrete implementations and run 1,000 trials per
(design × attack) cell. Week 1 stubs are minimal so the test harness can
import them.

Attack catalogue (kickoff §6):
  B1 orphan-child injection — child whose parent_id points at a never-created node
  B2 chain-break splice     — remove a middle node, relink endpoints directly
  B3 cross-chain merge      — graft a node from chain X into chain Y
  B4 ancestor revision      — modify an ancestor that has been extended
  B5 sibling fanout         — create many legitimate siblings to mask one malicious
"""

from .b1_orphan import attack_b1_orphan_injection
from .b2_splice import attack_b2_chain_break_splice
from .b3_cross_chain import attack_b3_cross_chain_merge
from .b4_ancestor_revision import attack_b4_ancestor_revision
from .b5_sibling_fanout import (
    attack_b5_sibling_fanout,
    evaluate_fanout_detection,
    FanoutDetector,
)


__all__ = [
    "attack_b1_orphan_injection",
    "attack_b2_chain_break_splice",
    "attack_b3_cross_chain_merge",
    "attack_b4_ancestor_revision",
    "attack_b5_sibling_fanout",
    "evaluate_fanout_detection",
    "FanoutDetector",
]
