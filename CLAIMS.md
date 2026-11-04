# Paper result-to-artifact map (CSCloud 2026)

Paths below are relative to the extracted repository root. Recorded results are
immutable evidence; use fresh output directories for all reproductions.

| Paper claim | Exact evidence | Scope / expected recorded result |
|---|---|---|
| Sec. VI-B/VI-C: B1–B4 ancestry conformance | `artifacts/current/asr_20260830_063346/asr_matrix.csv` | 3 designs × 4 classes × 1,000 checks = 12,000; historical B5 rows excluded |
| Sec. VI-C: complete-receipt conformance | `artifacts/current/complete_receipt_asr_20260830_064451/asr_matrix.csv` | A separate 12,000 checks; all specified invalid inputs rejected |
| Sec. VI-B / Fig. 2: controlled child compromise | `artifacts/current/e12_mcp_a2a_llm/results.json`, `security_attack` | Child-only linked claim accepted; invalid parent co-signature rejected. Other key-compromise entries are construction analysis, not measured attack rates |
| Sec. VI-D: authenticated fanout (E13) | `experiments/e13_authenticated_fanout/results/formal_20260913/{configuration.json,events.jsonl,summary.json,outcomes.csv}` | 360 sequences; 3,420 distinct candidates in each arm; 2,340 within-limit accepts and 1,080 overflow rejects; 1,233 state-preserving serial rejections |
| Sec. VI-D: races and persistence | Same E13 event log | 18 four-process race groups with one acceptance each; 9 fresh-process replay groups and 9 interrupted-transaction checks |
| Sec. VI-E / Table III: live MCP/A2A/LLM workflow (E12) | `artifacts/current/e12_mcp_a2a_llm/results.json` | 30 tasks/condition; means 770.8 / 826.2 / 813.1 ms for no evidence / LL / S-DAG; attested paths all accepted |
| Sec. VI-E: three-zone AWS deployment (E11--AWS) | `artifacts/current/e11_aws_20260831/rerun_01/results.json` | 1,000 valid paths; mean issuance 3.651 ms and complete verification 5.015 ms; routing metadata sanitized |
| Sec. VI-E: ancestry-only cost | `artifacts/current/perf_20260830_064113.csv` | Depth 5, 1,000 trials; mean 24.3 / 72.7 / 499.2 microseconds per hop for LL / Merkle / S-DAG |
| Sec. VI-E: LangGraph integration | `artifacts/current/langgraph_20260830_064130.json` | 1,000 complete LL checks; mean signing 0.512 ms/task, verification 0.832 ms/task |

The accounting script checks all distributed file hashes and the main numerical
claims. E13 replay additionally re-verifies the signed requests and state history.

```sh
python tools/verify_artifact.py
python scripts/paper2_asr_matrix.py --trials 1000 --verification-mode ancestry \
  --attacks B1 B2 B3 B4 --output-dir artifacts/reproduced/ancestry_new
python scripts/paper2_asr_matrix.py --trials 1000 --verification-mode complete \
  --attacks B1 B2 B3 B4 --output-dir artifacts/reproduced/complete_new
python scripts/paper2_perf_baseline.py --depth 5 --trials 1000 --warmup 100 \
  --out artifacts/reproduced/perf_new.csv
python -m experiments.e7_langgraph_integration.run --trials 1000 --warmup 100 \
  --out artifacts/reproduced/langgraph_new.json
python -B experiments/e13_authenticated_fanout/results/formal_20260913/source_snapshot/e13/run.py replay \
  --source experiments/e13_authenticated_fanout/results/formal_20260913 \
  --out artifacts/reproduced/e13_replay_new
```

See `README.md` for fresh E13 execution and live MCP/A2A/Ollama reproduction.
See `infra/e11_aws/README.md` for provisioning a separate cloud deployment.
Local E10 and container E11 recorded results support the deployment harness;
their measurements are not substituted for the AWS result.

E13's frozen executable source hashes remain original. E12's runner and the E10
signer / E11 coordinator hashes also match the recorded environment fields.
Cloud latency uses trial-level records; LangGraph and local ancestry latency
are available as recorded aggregates. No claim of full per-trial timing logs is
made for the latter two. Model and cloud latencies exclude E13 admission and do
not establish its latency or throughput.
