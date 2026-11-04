"""B5 — Sibling fanout obfuscation.

Threat: rather than break crypto, attacker creates a large number of
legitimate sibling instances (all properly signed) under one parent, hiding
ONE malicious sibling among them. Detection here is anomaly-detection-shaped,
not crypto-shaped — we measure how easily a verifier can flag the malicious
sibling under different fanout-rate thresholds.

Defence surface (all three designs equally — this is NOT a crypto property):
  - The verifier must implement a layered check above the chain crypto:
      1. Per-deployer fanout-rate threshold (e.g. > N siblings/hour from
         the same parent triggers review).
      2. Anomaly score on output content distribution across siblings.
  - This experiment quantifies the false-negative vs. false-positive
    trade-off of such a layer.

Attack model:
  - Attacker calls a legitimate deployer to spawn N benign children from
    the same parent, then spawns 1 malicious child (e.g. with a poisoned
    output_hash or altered certifications).
  - All children have VALID signatures and chain_proofs (the deployer
    issued them). The malicious one differs only in content.
  - Detection must come from the fanout-rate layer, not from crypto.

Variants:
  threshold_only    — pure fanout-rate threshold. Returns whether the
                      malicious child falls above or below the threshold.
  content_anomaly   — additionally checks if any sibling's output_hash
                      is an outlier relative to the set.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Literal, Optional, Sequence

from ...core.crypto import sha256_hex
from ...defenses.chain_designs.base import ChainNode


# ---------- Fanout detector (anomaly layer ABOVE the chain crypto) ----------

class FanoutDetector:
    """Simple fanout-rate anomaly detector.

    Tracks the number of children spawned per parent_id. Flags any parent
    whose child count exceeds `threshold` as suspicious.
    """

    def __init__(self, threshold: int = 50) -> None:
        self.threshold = threshold
        self._counts: dict[str, int] = {}

    def record_child(self, parent_id: str) -> None:
        self._counts[parent_id] = self._counts.get(parent_id, 0) + 1

    def is_flagged(self, parent_id: str) -> bool:
        return self._counts.get(parent_id, 0) > self.threshold

    def child_count(self, parent_id: str) -> int:
        return self._counts.get(parent_id, 0)


def attack_b5_sibling_fanout(
    parent_chain: Sequence[ChainNode],
    *,
    n_benign_siblings: int = 50,
    n_malicious_siblings: int = 1,
    malicious_marker: str = "malicious",
) -> tuple[list[ChainNode], list[ChainNode]]:
    """Spawn benign + malicious siblings from the same parent.

    All siblings are structurally valid (they would pass chain crypto checks
    if independently presented). Detection relies on the fanout detector.

    Args:
        parent_chain: the benign chain ending at the parent node.
        n_benign_siblings: how many benign children to spawn.
        n_malicious_siblings: how many malicious children to plant.
        malicious_marker: string tag embedded in malicious output_hash for
                          ground-truth identification.

    Returns:
        (benign_children, malicious_children) — each is a list of ChainNode.
        The caller runs a FanoutDetector over all children to measure detection.
    """
    if len(parent_chain) < 1:
        raise ValueError("B5 requires at least a parent chain of length 1")

    parent = parent_chain[-1].signed
    parent_payload = parent.payload

    def _make_sibling(suffix: str, output_tag: bytes) -> ChainNode:
        """Create a structurally valid sibling node (chain_proof is copied from
        a template — in the real attack the deployer would compute a fresh one,
        but for ASR testing the structure is what matters)."""
        child_payload = dataclasses.replace(
            parent_payload,
            instance_id=f"sibling-{suffix}",
            parent_id=parent_payload.instance_id,
            ancestry_depth=parent_payload.ancestry_depth + 1,
            output_hash=sha256_hex(output_tag),
            nonce=uuid.uuid4().hex,
        )
        return ChainNode(
            signed=dataclasses.replace(parent, payload=child_payload)
        )

    benign = [
        _make_sibling(f"benign-{i}", f"benign-output-{i}".encode())
        for i in range(n_benign_siblings)
    ]
    malicious = [
        _make_sibling(f"{malicious_marker}-{i}", f"{malicious_marker}-output-{i}".encode())
        for i in range(n_malicious_siblings)
    ]

    return benign, malicious


def evaluate_fanout_detection(
    benign: list[ChainNode],
    malicious: list[ChainNode],
    *,
    threshold: int = 50,
) -> dict[str, float]:
    """Run a fanout-rate detection evaluation over a set of siblings.

    Returns metrics:
      - total_children: total count of benign + malicious siblings.
      - threshold: the configured fanout threshold.
      - parent_flagged: whether the parent was flagged (True if total > threshold).
      - true_positive_rate: fraction of malicious siblings that are
        detectable (= 1.0 if parent is flagged, since all children of a
        flagged parent are under review).
      - false_positive_rate: fraction of benign siblings flagged. If
        parent is flagged, all benign are also flagged (100% FPR wrt
        the per-child perspective). This is the cost of threshold-only.
    """
    total = len(benign) + len(malicious)
    parent_flagged = total > threshold

    return {
        "total_children": total,
        "threshold": threshold,
        "parent_flagged": parent_flagged,
        "true_positive_rate": 1.0 if parent_flagged else 0.0,
        "false_positive_rate": 1.0 if parent_flagged else 0.0,
        "n_benign": len(benign),
        "n_malicious": len(malicious),
    }
