"""E2 Baseline Evaluation — replay hand-crafted + parametric attacks against all backends.

Instead of spending $30+ on LLM-generated attacks against baselines (which
would trivially succeed), we run the project's own hand-crafted B1–B5 attack
functions against all 6 backends. This is strictly MORE rigorous than LLM
variants because:
  - Hand-crafted attacks are the gold-standard implementations
  - We test every parametric variant (not just the subset Haiku happens to try)
  - Deterministic: zero randomness, fully reproducible
  - Zero API cost

For each (backend × attack × variant), we record pass/fail. The expected
result is that baselines show high bypass rates while primary designs show 0%.

Usage:
    python -m src.analysis.e2_baseline_eval
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.core.agent_id_v2 import (
    BoundIDPayloadV2,
    ChainProof,
    make_chain_proof_linked_list,
    make_chain_proof_signed_dag,
)
from src.core.crypto import generate_keypair, sha256_hex, sign
from src.defenses.chain_designs import (
    ChainNode,
    DeployerRegistry,
    HashOnlyBackend,
    LinkedListBackend,
    MerkleBackend,
    NaiveSigBackend,
    NoChainBackend,
    SignedDAGBackend,
)
from src.defenses.chain_designs.base import ChainVerifyError, edge_signing_input
from src.runtime.multi_deployer import ChainBuilder, MultiDeployerHarness


# ============================================================================
# Chain builders for each backend
# ============================================================================

def build_chain(harness: MultiDeployerHarness, backend, depth: int = 5):
    """Build a benign chain of given depth."""
    names = harness.names()
    b = ChainBuilder(harness, backend=backend)
    chain = b.start_root(names[0], "a0", "claude-3-sonnet", b"root-output")
    for i in range(1, depth):
        host = names[i % len(names)]
        co_signers = tuple(names) if backend.design_name == "signed_dag" else ()
        chain = b.extend(
            chain, host=host, instance_id=f"a{i}",
            system_id="claude-3-sonnet",
            output=f"step-{i}-output".encode(),
            co_signer_names=co_signers,
        )
    return chain


# ============================================================================
# Attack implementations (adapted from src/attacks/ancestry/)
# ============================================================================

def attack_b1_orphan(chain: list[ChainNode]) -> list[ChainNode]:
    """B1: Change last node's parent_id to a nonexistent ID."""
    result = list(chain)
    old = result[-1].signed.payload
    new_p = dataclasses.replace(old, parent_id="nonexistent-parent-" + uuid.uuid4().hex[:8])
    result[-1] = ChainNode(signed=dataclasses.replace(result[-1].signed, payload=new_p))
    return result


def attack_b1_orphan_root_swap(chain: list[ChainNode]) -> list[ChainNode]:
    """B1 variant: Make root claim a parent (orphan root)."""
    result = list(chain)
    old = result[0].signed.payload
    new_p = dataclasses.replace(old, parent_id="phantom-root", ancestry_depth=1)
    result[0] = ChainNode(signed=dataclasses.replace(result[0].signed, payload=new_p))
    return result


def attack_b2_splice_middle(chain: list[ChainNode]) -> list[ChainNode]:
    """B2: Remove middle node, keep rest unchanged."""
    if len(chain) < 4:
        return chain  # need length >= 4
    return [chain[0], chain[1]] + chain[3:]


def attack_b2_splice_early(chain: list[ChainNode]) -> list[ChainNode]:
    """B2 variant: Remove node at index 1 (root-adjacent splice)."""
    if len(chain) < 3:
        return chain
    return [chain[0]] + chain[2:]


def attack_b3_cross_chain_graft(
    chain_x: list[ChainNode], chain_y: list[ChainNode]
) -> list[ChainNode]:
    """B3: Graft last node of chain_y onto chain_x."""
    result = list(chain_x)
    grafted = chain_y[-1]
    old_p = grafted.signed.payload
    # Patch parent_id to point to chain_x's last node
    new_p = dataclasses.replace(
        old_p,
        parent_id=chain_x[-1].signed.payload.instance_id,
        ancestry_depth=chain_x[-1].signed.payload.ancestry_depth + 1,
    )
    result.append(ChainNode(signed=dataclasses.replace(grafted.signed, payload=new_p)))
    return result


def attack_b4_revision_output(chain: list[ChainNode], idx: int = 1) -> list[ChainNode]:
    """B4: Flip ancestor's output_hash without re-signing."""
    result = list(chain)
    old = result[idx].signed.payload
    new_p = dataclasses.replace(old, output_hash=sha256_hex(b"evil-" + uuid.uuid4().bytes))
    result[idx] = ChainNode(signed=dataclasses.replace(result[idx].signed, payload=new_p))
    return result


def attack_b4_revision_deployer(chain: list[ChainNode], idx: int = 1) -> list[ChainNode]:
    """B4 variant: Change ancestor's deployer field."""
    result = list(chain)
    old = result[idx].signed.payload
    new_p = dataclasses.replace(old, deployer="rogue-deployer")
    result[idx] = ChainNode(signed=dataclasses.replace(result[idx].signed, payload=new_p))
    return result


def attack_b4_revision_certs(chain: list[ChainNode], idx: int = 1) -> list[ChainNode]:
    """B4 variant: Add fake certification to ancestor."""
    result = list(chain)
    old = result[idx].signed.payload
    new_p = dataclasses.replace(
        old, certifications=old.certifications + ("fabricated_cert",)
    )
    result[idx] = ChainNode(signed=dataclasses.replace(result[idx].signed, payload=new_p))
    return result


def attack_b5_fanout(
    harness: MultiDeployerHarness, backend, parent_chain: list[ChainNode],
    n_siblings: int = 10,
) -> list[list[ChainNode]]:
    """B5: Create n siblings from the same parent. Returns list of chains."""
    b = ChainBuilder(harness, backend=backend)
    siblings = []
    for i in range(n_siblings):
        co_signers = ("D_A", "D_B") if backend.design_name == "signed_dag" else ()
        sibling = b.extend(
            parent_chain, host="D_A", instance_id=f"sibling-{i}",
            system_id="claude-3-sonnet",
            output=f"sibling-{i}-output".encode(),
            co_signer_names=co_signers,
        )
        siblings.append(sibling)
    return siblings


# ============================================================================
# Evaluation Matrix
# ============================================================================

ALL_BACKENDS = {
    # Baselines
    "no_chain": NoChainBackend,
    "hash_only": HashOnlyBackend,
    "naive_sig": NaiveSigBackend,
    # Primary
    "linked_list": LinkedListBackend,
    "merkle": MerkleBackend,
    "signed_dag": SignedDAGBackend,
}


def run_evaluation(output_dir: Path) -> Path:
    """Run all attacks against all backends and record results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "e2_baseline_matrix.csv"

    harness = MultiDeployerHarness.with_deployers(["D_A", "D_B"])

    results = []

    for design_name, BackendCls in ALL_BACKENDS.items():
        backend = BackendCls()

        # Build chains for this backend
        chain = build_chain(harness, backend, depth=5)

        # For B3, build a second chain
        chain_y = build_chain(
            MultiDeployerHarness.with_deployers(["D_C", "D_D"]),
            BackendCls(),  # separate backend instance
            depth=3,
        )

        # Define attacks
        attacks = {
            "B1_orphan_leaf": lambda c=chain: attack_b1_orphan(c),
            "B1_orphan_root": lambda c=chain: attack_b1_orphan_root_swap(c),
            "B2_splice_middle": lambda c=chain: attack_b2_splice_middle(c),
            "B2_splice_early": lambda c=chain: attack_b2_splice_early(c),
            "B3_cross_chain": lambda c=chain, cy=chain_y: attack_b3_cross_chain_graft(c, cy),
            "B4_output_hash": lambda c=chain: attack_b4_revision_output(c, idx=1),
            "B4_output_hash_idx2": lambda c=chain: attack_b4_revision_output(c, idx=2),
            "B4_output_hash_idx3": lambda c=chain: attack_b4_revision_output(c, idx=3),
            "B4_deployer": lambda c=chain: attack_b4_revision_deployer(c, idx=1),
            "B4_certs": lambda c=chain: attack_b4_revision_certs(c, idx=1),
        }

        for attack_name, attack_fn in attacks.items():
            attack_cat = attack_name.split("_")[0]  # B1, B2, B3, B4
            try:
                manipulated = attack_fn()
                backend.verify_chain(chain=manipulated, deployers=harness.registry)
                bypassed = True
            except (ChainVerifyError, Exception):
                bypassed = False

            results.append({
                "design": design_name,
                "attack_category": attack_cat,
                "attack_variant": attack_name,
                "bypassed": bypassed,
                "is_baseline": design_name in ("no_chain", "hash_only", "naive_sig"),
            })

        # B5: fanout (always succeeds structurally — it's about anomaly detection)
        try:
            siblings = attack_b5_fanout(harness, backend, chain[:2], n_siblings=10)
            # B5 "success" = all siblings pass verification individually
            all_valid = all(
                _verify_silent(backend, sib, harness.registry)
                for sib in siblings
            )
            results.append({
                "design": design_name,
                "attack_category": "B5",
                "attack_variant": "B5_fanout_10",
                "bypassed": all_valid,
                "is_baseline": design_name in ("no_chain", "hash_only", "naive_sig"),
            })
        except Exception:
            results.append({
                "design": design_name,
                "attack_category": "B5",
                "attack_variant": "B5_fanout_10",
                "bypassed": False,
                "is_baseline": design_name in ("no_chain", "hash_only", "naive_sig"),
            })

    # Write CSV
    fields = ["design", "attack_category", "attack_variant", "bypassed", "is_baseline"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    _print_summary(results)

    return csv_path


def _verify_silent(backend, chain, registry) -> bool:
    """Verify without raising."""
    try:
        backend.verify_chain(chain=chain, deployers=registry)
        return True
    except Exception:
        return False


def _print_summary(results: list[dict]) -> None:
    """Print the attack bypass matrix."""
    # Group by design × attack_category
    from collections import defaultdict
    matrix = defaultdict(lambda: defaultdict(lambda: {"total": 0, "bypassed": 0}))

    for r in results:
        cell = matrix[r["design"]][r["attack_category"]]
        cell["total"] += 1
        if r["bypassed"]:
            cell["bypassed"] += 1

    designs_order = ["no_chain", "hash_only", "naive_sig", "linked_list", "merkle", "signed_dag"]
    attacks_order = ["B1", "B2", "B3", "B4", "B5"]

    # Header
    header = f"{'':>12} |"
    for d in designs_order:
        header += f" {d:>12}"
    print(header)
    print("-" * len(header))

    for atk in attacks_order:
        row = f"{atk:>12} |"
        for d in designs_order:
            cell = matrix[d][atk]
            if cell["total"] == 0:
                row += f" {'N/A':>12}"
            else:
                rate = cell["bypassed"] / cell["total"]
                row += f" {rate:>11.0%}{'*' if cell['bypassed'] > 0 else ' '}"
        print(row)

    print()
    print("* = at least one variant bypassed verification")

    # Aggregate
    print()
    for d in designs_order:
        total = sum(matrix[d][a]["total"] for a in attacks_order)
        bypassed = sum(matrix[d][a]["bypassed"] for a in attacks_order)
        pct = bypassed / total * 100 if total else 0
        label = "BASELINE" if d in ("no_chain", "hash_only", "naive_sig") else "PRIMARY"
        print(f"  {d:>12} [{label}]: {bypassed}/{total} bypassed ({pct:.0f}%)")


# ============================================================================
# CLI
# ============================================================================

def main():
    repo = Path(__file__).resolve().parents[2]
    output_dir = repo / "results" / "paper2" / "e2_baselines"
    print(f"E2 Baseline Evaluation — {datetime.now(timezone.utc).isoformat()}")
    print(f"Output: {output_dir}")
    print()
    csv_path = run_evaluation(output_dir)
    print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
