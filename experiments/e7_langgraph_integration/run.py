"""E7: LangGraph integration experiment for ancestry attestation.

This experiment executes a real LangGraph ``StateGraph`` with three
deterministic agent nodes: Planner -> Researcher -> Writer.  Each node invokes
the deployer-side signing hook after producing its output.  Deterministic node
outputs intentionally isolate protocol overhead from model-inference latency;
this is a framework integration experiment, not an LLM-throughput benchmark.

Run from the repository root:

    python -m experiments.e7_langgraph_integration.run --trials 1000

The command writes ``results.json`` next to this file.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.defenses.chain_designs import ChainNode, LinkedListBackend
from src.defenses.complete_chain_verifier import verify_complete_chain
from src.core.verifier import InMemoryNonceStore
from src.runtime.multi_deployer import ChainBuilder, MultiDeployerHarness


@dataclass(frozen=True)
class AttestationRecord:
    """Timing and provenance record for one executed LangGraph node."""

    agent: str
    deployer: str
    ancestry_depth: int
    sign_latency_ms: float


class PipelineState(TypedDict):
    """State that flows through the actual LangGraph execution."""

    chain: list[ChainNode]
    records: list[AttestationRecord]
    final_output: str


_NODE_OUTPUTS = {
    "planner": "Plan: research the topic, summarize findings, and write a report.",
    "researcher": "Findings: delegation provenance requires authenticated parent-child edges.",
    "writer": "Report: deployer-side attestations preserve delegation ancestry for audit.",
}
_NODE_DEPLOYERS = {
    "planner": "platform_A",
    "researcher": "platform_A",
    "writer": "platform_B",
}


class LangGraphAttestationPipeline:
    """A three-node StateGraph whose nodes trigger deployer-side attestation."""

    def __init__(self) -> None:
        self.harness = MultiDeployerHarness.with_deployers(["platform_A", "platform_B"])
        self.backend = LinkedListBackend()
        self.builder = ChainBuilder(self.harness, self.backend)
        # The benchmark uses a persistent verifier session.  It is deliberately
        # process-local: results do not claim distributed replay protection.
        self.nonces = InMemoryNonceStore()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(PipelineState)
        workflow.add_node("planner", lambda state: self._attest(state, "planner"))
        workflow.add_node("researcher", lambda state: self._attest(state, "researcher"))
        workflow.add_node("writer", lambda state: self._attest(state, "writer"))
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "writer")
        workflow.add_edge("writer", END)
        return workflow.compile()

    def _attest(self, state: PipelineState, agent: str) -> dict[str, object]:
        """Execute after a graph node produces output and append its proof."""

        output = _NODE_OUTPUTS[agent]
        deployer = _NODE_DEPLOYERS[agent]
        chain = state["chain"]
        t0 = time.perf_counter_ns()
        if chain:
            signed_chain = self.builder.extend(
                chain,
                host=deployer,
                instance_id=agent,
                system_id="langgraph-node",
                output=output.encode("utf-8"),
            )
        else:
            signed_chain = self.builder.start_root(
                deployer,
                agent,
                "langgraph-node",
                output.encode("utf-8"),
            )
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000
        payload = signed_chain[-1].signed.payload
        record = AttestationRecord(
            agent=agent,
            deployer=deployer,
            ancestry_depth=payload.ancestry_depth,
            sign_latency_ms=latency_ms,
        )
        return {
            "chain": signed_chain,
            "records": [*state["records"], record],
            "final_output": output,
        }

    def run_once(self) -> dict[str, object]:
        """Execute all graph nodes and verify the resulting full chain."""

        state = self.graph.invoke({"chain": [], "records": [], "final_output": ""})
        chain = state["chain"]
        t0 = time.perf_counter_ns()
        outputs = [
            _NODE_OUTPUTS[record.agent].encode("utf-8")
            for record in state["records"]
        ]
        verify_complete_chain(
            chain=chain,
            outputs=outputs,
            backend=self.backend,
            deployers=self.harness.registry,
            nonces=self.nonces,
        )
        verify_latency_ms = (time.perf_counter_ns() - t0) / 1_000_000
        return {
            "chain_verified": True,
            "chain_nodes": len(chain),
            "records": state["records"],
            "verify_latency_ms": verify_latency_ms,
            "final_output": state["final_output"],
        }


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("cannot summarize an empty measurement")
    return {
        "mean": round(sum(ordered) / n, 4),
        "p50": round(ordered[n // 2], 4),
        "p95": round(ordered[int(n * 0.95)], 4),
        "p99": round(ordered[int(n * 0.99)], 4),
    }


def run_benchmark(n_trials: int = 1000, warmup: int = 100) -> dict[str, object]:
    """Benchmark the signing hook and full-chain verification after warm-up."""

    if n_trials <= 0 or warmup < 0:
        raise ValueError("n_trials must be positive and warmup must be non-negative")
    pipeline = LangGraphAttestationPipeline()
    for _ in range(warmup):
        pipeline.run_once()

    per_node_sign_ms: list[float] = []
    total_sign_ms: list[float] = []
    chain_verify_ms: list[float] = []
    sample: dict[str, object] | None = None
    for _ in range(n_trials):
        result = pipeline.run_once()
        records = result["records"]
        assert isinstance(records, list)
        sign_times = [record.sign_latency_ms for record in records]
        per_node_sign_ms.extend(sign_times)
        total_sign_ms.append(sum(sign_times))
        chain_verify_ms.append(float(result["verify_latency_ms"]))
        sample = result

    assert sample is not None
    sample_records = sample["records"]
    assert isinstance(sample_records, list)
    return {
        "framework": "langgraph",
        "framework_version": version("langgraph"),
        "scenario": "Planner -> Researcher -> Writer",
        "design": "linked_list",
        "node_outputs": "deterministic; no LLM invocation",
        "n_trials": n_trials,
        "warmup": warmup,
        "all_chains_verified": True,
        "per_node_sign_ms": _stats(per_node_sign_ms),
        "total_sign_overhead_per_task_ms": _stats(total_sign_ms),
        "full_chain_verify_ms": _stats(chain_verify_ms),
        "sample": {
            "chain_nodes": sample["chain_nodes"],
            "records": [asdict(record) for record in sample_records],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("results.json"),
    )
    args = parser.parse_args()
    results = run_benchmark(args.trials, args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    print(
        "LangGraph integration: "
        f"{results['n_trials']} executions, all chains verified; "
        f"per-node signing mean={results['per_node_sign_ms']['mean']:.4f} ms; "
        f"full-chain verification mean={results['full_chain_verify_ms']['mean']:.4f} ms."
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
