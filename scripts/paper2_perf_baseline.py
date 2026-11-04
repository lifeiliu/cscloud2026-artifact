"""Per-hop verify latency baseline for Paper 2's three chain designs.

Run:
    python scripts/paper2_perf_baseline.py [--depth N] [--trials T]

Outputs:
    - Console summary (mean / p50 / p95 / p99 in microseconds)
    - CSV at results/paper2/perf_baseline.csv

This is the §6.5 performance anchor for the paper. It does NOT include
LLM inference time; that ratio is filled in once Week 4's lifecycle runtime
plus a stub LLM model loader provides a reference number.

Methodology notes:
  - Each trial builds a fresh chain (so node bytes are not cached) and times
    `verify_chain` end-to-end. We report both per-CHAIN and per-HOP latency.
  - Wall-clock via `time.perf_counter_ns`. Disable PGO/JIT noise by warming
    up 100 trials before the timed run.
  - Fixed PRNG seed for the keypair generation order would be ideal but
    Ed25519 keys derive from os.urandom; we accept this and measure across
    1000 trials to amortise.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

# Permit ``python scripts/paper2_perf_baseline.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.defenses.chain_designs import (
    LinkedListBackend,
    MerkleBackend,
    SignedDAGBackend,
)
from src.runtime.multi_deployer import ChainBuilder, MultiDeployerHarness


def build_chain(harness, backend, depth: int, *, dag_co_signers: bool):
    """Build a depth-N chain alternating between two deployers."""
    b = ChainBuilder(harness, backend)
    chain = b.start_root("D_A", "agent-0", "claude-3-sonnet", b"root")
    for i in range(1, depth + 1):
        host = "D_B" if i % 2 == 1 else "D_A"
        parent_dep = chain[-1].signed.payload.deployer
        if dag_co_signers:
            co = (parent_dep,) if parent_dep == host else (parent_dep, host)
        else:
            co = ()
        chain = b.extend(
            chain,
            host=host,
            instance_id=f"agent-{i}",
            system_id="claude-3-sonnet",
            output=f"step {i}".encode("utf-8"),
            co_signer_names=co,
        )
    return chain


def percentile(sorted_values, p: float) -> float:
    """Linear-interpolation percentile. p in [0, 1]."""
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    return float(sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f))


def run_design(name: str, backend, harness, *, depth: int, trials: int, warmup: int):
    """Time `trials` builds + verifies. Returns dict of stats in microseconds."""
    dag_cs = name == "signed_dag"

    # Warmup
    for _ in range(warmup):
        chain = build_chain(harness, backend, depth, dag_co_signers=dag_cs)
        backend.verify_chain(chain=chain, deployers=harness.registry)

    chain_us: list[float] = []
    for _ in range(trials):
        chain = build_chain(harness, backend, depth, dag_co_signers=dag_cs)
        t0 = time.perf_counter_ns()
        backend.verify_chain(chain=chain, deployers=harness.registry)
        t1 = time.perf_counter_ns()
        chain_us.append((t1 - t0) / 1000.0)

    chain_us_sorted = sorted(chain_us)
    n_hops = depth  # number of edges (root has no incoming edge)
    per_hop = [v / n_hops for v in chain_us]
    per_hop_sorted = sorted(per_hop)

    return {
        "design": name,
        "depth": depth,
        "trials": trials,
        "chain_mean_us": statistics.mean(chain_us),
        "chain_p50_us": percentile(chain_us_sorted, 0.50),
        "chain_p95_us": percentile(chain_us_sorted, 0.95),
        "chain_p99_us": percentile(chain_us_sorted, 0.99),
        "per_hop_mean_us": statistics.mean(per_hop),
        "per_hop_p50_us": percentile(per_hop_sorted, 0.50),
        "per_hop_p95_us": percentile(per_hop_sorted, 0.95),
        "per_hop_p99_us": percentile(per_hop_sorted, 0.99),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/paper2/perf_baseline.csv"),
    )
    args = parser.parse_args()

    harness = MultiDeployerHarness.with_deployers(["D_A", "D_B"])
    designs = [
        ("linked_list", LinkedListBackend()),
        ("merkle", MerkleBackend()),
        ("signed_dag", SignedDAGBackend()),
    ]

    rows = []
    for name, backend in designs:
        rows.append(
            run_design(
                name,
                backend,
                harness,
                depth=args.depth,
                trials=args.trials,
                warmup=args.warmup,
            )
        )

    # Console summary
    print(
        f"\nWeek 1 perf baseline — depth={args.depth}, trials={args.trials}, "
        f"warmup={args.warmup}"
    )
    print("-" * 78)
    print(
        f"{'design':12} {'chain_mean':>11} {'chain_p95':>11} "
        f"{'per_hop_mean':>13} {'per_hop_p95':>13}"
    )
    print(
        f"{'':12} {'(us)':>11} {'(us)':>11} "
        f"{'(us)':>13} {'(us)':>13}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['design']:12} {r['chain_mean_us']:>11.2f} "
            f"{r['chain_p95_us']:>11.2f} "
            f"{r['per_hop_mean_us']:>13.2f} "
            f"{r['per_hop_p95_us']:>13.2f}"
        )

    # CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
