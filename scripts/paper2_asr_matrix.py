"""Paper 2 — Fixed-adversary ASR matrix: 3 designs × 5 attacks × 1,000 trials.

Exit criterion for Week 2: all 15 cells have ASR + Wilson 95% CIs.
H1 target: all three designs achieve ASR ≤ 1% on B1–B5 (Wilson upper bound ≤ 0.014).

Outputs:
  - results/paper2/asr_matrix.csv         — per-cell summary
  - results/paper2/asr_matrix_detail.csv  — per-trial detail
  - results/paper2/asr_matrix_summary.txt — human-readable summary

Usage:
  python scripts/paper2_asr_matrix.py [--trials 1000] [--designs linked_list merkle signed_dag]

``--verification-mode complete`` additionally validates the primary receipt
signature, key identifier, expiry, output bytes, and nonce predicates. Only an
explicit fail-closed ``ChainVerifyError`` is counted as a rejected attack;
unexpected exceptions fail the experiment rather than being silently counted.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

# Project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.stats import CellResult, wilson_ci
from src.attacks.ancestry import (
    attack_b1_orphan_injection,
    attack_b2_chain_break_splice,
    attack_b3_cross_chain_merge,
    attack_b4_ancestor_revision,
    attack_b5_sibling_fanout,
    evaluate_fanout_detection,
)
from src.defenses.chain_designs import (
    ChainNode,
    ChainVerifyError,
    LinkedListBackend,
    MerkleBackend,
    SignedDAGBackend,
    DeployerRegistry,
)
from src.defenses.complete_chain_verifier import verify_complete_chain
from src.runtime.multi_deployer import ChainBuilder, MultiDeployerHarness

logger = logging.getLogger(__name__)

# ---------- Chain-building helpers ----------

CHAIN_DEPTH = 5  # matches Week 1 round-trip test

DESIGNS = {
    "linked_list": LinkedListBackend,
    "merkle": MerkleBackend,
    "signed_dag": SignedDAGBackend,
}


def build_benign_chain(
    harness: MultiDeployerHarness,
    backend_cls: type,
    depth: int = CHAIN_DEPTH,
) -> list[ChainNode]:
    """Build a benign chain of `depth` hops alternating D_A/D_B."""
    backend = backend_cls()
    is_dag = backend.design_name == "signed_dag"
    b = ChainBuilder(harness, backend)
    chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"root-output")

    hosts = ["D_A", "D_B"]
    for i in range(1, depth + 1):
        host = hosts[i % 2]
        parent_dep = chain[-1].signed.payload.deployer

        if is_dag:
            if parent_dep == host:
                co = (parent_dep,)
            else:
                co = (parent_dep, host)
        else:
            co = ()

        chain = b.extend(
            chain,
            host=host,
            instance_id=f"agent-{i}",
            system_id="claude-3-sonnet",
            output=f"output-{i}".encode(),
            co_signer_names=co,
        )
    return chain


# ---------- Attack runners ----------

B1_VARIANTS = ["fabricated_parent", "missing_parent", "ghost_id"]
B2_VARIANTS = ["root_adjacent", "middle", "leaf_adjacent"]
B3_VARIANTS = ["graft_at_leaf", "graft_at_mid", "append_alien"]
B4_FIELDS = ["output_hash", "certifications", "deployer", "parent_id"]
B5_THRESHOLDS = [10, 50, 100, 500]
VerificationMode = Literal["ancestry", "complete"]


def _outputs_for_chain(chain: Sequence[ChainNode]) -> list[bytes]:
    """Reconstruct deterministic bytes emitted by ``build_benign_chain``."""
    outputs: list[bytes] = []
    for node in chain:
        payload = node.signed.payload
        instance_id = payload.instance_id
        if payload.ancestry_depth == 0:
            outputs.append(b"root-output")
        elif instance_id.startswith("agent-") and instance_id[6:].isdigit():
            outputs.append(f"output-{instance_id[6:]}".encode())
        else:
            raise ValueError(
                f"cannot reconstruct deterministic output for {instance_id!r}"
            )
    return outputs


def _accepted(
    chain: Sequence[ChainNode],
    backend,
    registry: DeployerRegistry,
    verification_mode: VerificationMode,
) -> bool:
    """Return whether the selected verifier accepts an adversarial chain."""
    try:
        if verification_mode == "complete":
            verify_complete_chain(
                chain=chain,
                outputs=_outputs_for_chain(chain),
                backend=backend,
                deployers=registry,
            )
        else:
            backend.verify_chain(chain=chain, deployers=registry)
        return True
    except ChainVerifyError:
        return False


def run_b1_trial(
    chain: list[ChainNode],
    backend,
    registry: DeployerRegistry,
    rng_idx: int,
    verification_mode: VerificationMode,
) -> bool:
    """Run one B1 trial. Returns True if the attack SUCCEEDS (bypass)."""
    variant = B1_VARIANTS[rng_idx % len(B1_VARIANTS)]
    mutated = attack_b1_orphan_injection(chain, variant=variant)
    return _accepted(mutated, backend, registry, verification_mode)


def run_b2_trial(
    chain: list[ChainNode],
    backend,
    registry: DeployerRegistry,
    rng_idx: int,
    verification_mode: VerificationMode,
) -> bool:
    """Run one B2 trial. Returns True if the attack SUCCEEDS."""
    variant = B2_VARIANTS[rng_idx % len(B2_VARIANTS)]
    mutated = attack_b2_chain_break_splice(chain, variant=variant)
    return _accepted(mutated, backend, registry, verification_mode)


def run_b3_trial(
    chain_y: list[ChainNode],
    chain_x: list[ChainNode],
    backend,
    registry: DeployerRegistry,
    rng_idx: int,
    verification_mode: VerificationMode,
) -> bool:
    """Run one B3 trial. Returns True if the attack SUCCEEDS."""
    variant = B3_VARIANTS[rng_idx % len(B3_VARIANTS)]
    mutated = attack_b3_cross_chain_merge(chain_y, chain_x, variant=variant)
    return _accepted(mutated, backend, registry, verification_mode)


def run_b4_trial(
    chain: list[ChainNode],
    backend,
    registry: DeployerRegistry,
    rng_idx: int,
    verification_mode: VerificationMode,
) -> bool:
    """Run one B4 trial. Returns True if the attack SUCCEEDS."""
    field = B4_FIELDS[rng_idx % len(B4_FIELDS)]
    # Revise an internal ancestor (not root, not leaf)
    max_internal = len(chain) - 2
    revise_idx = (rng_idx % max_internal) + 1 if max_internal > 0 else 1
    mutated = attack_b4_ancestor_revision(chain, revise_index=revise_idx, field=field)
    return _accepted(mutated, backend, registry, verification_mode)


def run_b5_trial(
    chain: list[ChainNode],
    rng_idx: int,
) -> bool:
    """Run one B5 trial. Returns True if the attack SUCCEEDS (malicious child NOT detected)."""
    threshold = B5_THRESHOLDS[rng_idx % len(B5_THRESHOLDS)]
    # Spawn n_benign = threshold (so we're right at the edge) + 1 malicious
    n_benign = threshold
    benign, malicious = attack_b5_sibling_fanout(
        chain, n_benign_siblings=n_benign, n_malicious_siblings=1,
    )
    result = evaluate_fanout_detection(benign, malicious, threshold=threshold)
    # Attack succeeds if the malicious child is NOT detected
    return not result["parent_flagged"]


# ---------- Matrix runner ----------

def run_cell(
    design_name: str,
    attack_name: str,
    n_trials: int,
    harness: MultiDeployerHarness,
    verification_mode: VerificationMode = "ancestry",
) -> CellResult:
    """Run one (design × attack) cell and return the ASR + Wilson CI."""
    if verification_mode == "complete" and attack_name == "B5":
        raise ValueError(
            "B5 is a separate stateful fanout-detector experiment; it does not "
            "exercise the composed receipt--chain verifier"
        )
    backend_cls = DESIGNS[design_name]
    backend = backend_cls()
    registry = harness.registry

    successes = 0
    for trial_idx in range(n_trials):
        chain = build_benign_chain(harness, backend_cls)

        if attack_name == "B1":
            success = run_b1_trial(chain, backend, registry, trial_idx, verification_mode)
        elif attack_name == "B2":
            success = run_b2_trial(chain, backend, registry, trial_idx, verification_mode)
        elif attack_name == "B3":
            # Build a second independent chain for cross-chain graft
            chain_x = build_benign_chain(harness, backend_cls)
            success = run_b3_trial(
                chain, chain_x, backend, registry, trial_idx, verification_mode
            )
        elif attack_name == "B4":
            success = run_b4_trial(chain, backend, registry, trial_idx, verification_mode)
        elif attack_name == "B5":
            success = run_b5_trial(chain, trial_idx)
        else:
            raise ValueError(f"unknown attack: {attack_name}")

        if success:
            successes += 1

    ci = wilson_ci(successes, n_trials)
    return CellResult(
        design=design_name,
        attack=attack_name,
        n_trials=n_trials,
        n_successes=successes,
        asr=ci.point,
        ci_low=ci.low,
        ci_high=ci.high,
    )


def run_matrix(
    designs: Sequence[str],
    attacks: Sequence[str],
    n_trials: int,
    output_dir: Path,
    verification_mode: VerificationMode = "ancestry",
) -> list[CellResult]:
    """Run the full ASR matrix and write results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    harness = MultiDeployerHarness.with_deployers(["D_A", "D_B"])

    results: list[CellResult] = []
    total_cells = len(designs) * len(attacks)
    cell_num = 0

    for design in designs:
        for attack in attacks:
            cell_num += 1
            t0 = time.monotonic()
            logger.info(
                "[%d/%d] Running %s × %s (%d trials)...",
                cell_num, total_cells, design, attack, n_trials,
            )
            result = run_cell(
                design, attack, n_trials, harness, verification_mode=verification_mode
            )
            elapsed = time.monotonic() - t0
            logger.info(
                "  → ASR=%s  (%.1fs)",
                result.ci_str, elapsed,
            )
            results.append(result)

    # Write CSV
    csv_path = output_dir / "asr_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "design", "attack", "n_trials", "n_successes",
            "asr", "ci_low", "ci_high",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "design": r.design,
                "attack": r.attack,
                "n_trials": r.n_trials,
                "n_successes": r.n_successes,
                "asr": f"{r.asr:.6f}",
                "ci_low": f"{r.ci_low:.6f}",
                "ci_high": f"{r.ci_high:.6f}",
            })

    # Write human-readable summary
    _write_summary(results, output_dir)

    logger.info("Results written to %s", csv_path)
    return results


def _write_summary(results: list[CellResult], output_dir: Path) -> None:
    """Write a formatted summary table."""
    summary_path = output_dir / "asr_matrix_summary.txt"

    # Build the heatmap grid
    designs = sorted(set(r.design for r in results))
    attacks = sorted(set(r.attack for r in results))
    lookup = {(r.design, r.attack): r for r in results}

    lines = [
        "Paper 2 — Fixed-Adversary ASR Matrix (H1)",
        "=" * 70,
        "",
        f"{'':20s} " + "  ".join(f"{a:>12s}" for a in attacks),
        "-" * (20 + 14 * len(attacks)),
    ]

    h1_pass = True
    for d in designs:
        row_parts = []
        for a in attacks:
            r = lookup.get((d, a))
            if r:
                row_parts.append(f"{r.asr:>6.4f} ±{r.ci_high - r.ci_low:>5.4f}")
                if r.ci_high > 0.014:
                    h1_pass = False
            else:
                row_parts.append(f"{'N/A':>12s}")
        lines.append(f"{d:20s} " + "  ".join(row_parts))

    lines.append("-" * (20 + 14 * len(attacks)))
    lines.append("")
    lines.append(f"H1 (ASR ≤ 1%, Wilson upper ≤ 0.014): {'PASS' if h1_pass else 'FAIL'}")
    lines.append("")

    # Per-cell detail
    lines.append("Per-cell detail:")
    for r in results:
        status = "✓" if r.ci_high <= 0.014 else "✗"
        lines.append(
            f"  {status} {r.design:15s} × {r.attack:3s}: "
            f"ASR={r.asr:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}] "
            f"({r.n_successes}/{r.n_trials})"
        )

    summary_path.write_text("\n".join(lines) + "\n")


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper 2 — Fixed-adversary ASR matrix",
    )
    parser.add_argument(
        "--trials", type=int, default=1000,
        help="Trials per cell (default: 1000)",
    )
    parser.add_argument(
        "--designs", nargs="+",
        choices=list(DESIGNS.keys()),
        default=list(DESIGNS.keys()),
        help="Which chain designs to test",
    )
    parser.add_argument(
        "--attacks", nargs="+",
        choices=["B1", "B2", "B3", "B4", "B5"],
        default=["B1", "B2", "B3", "B4", "B5"],
        help="Which attacks to run",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "paper2",
    )
    parser.add_argument(
        "--verification-mode",
        choices=["ancestry", "complete"],
        default="ancestry",
        help="Use only chain predicates or the composed receipt--chain verifier",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    run_matrix(
        designs=args.designs,
        attacks=args.attacks,
        n_trials=args.trials,
        output_dir=args.output_dir,
        verification_mode=args.verification_mode,
    )


if __name__ == "__main__":
    main()
