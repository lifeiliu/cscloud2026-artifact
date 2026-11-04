# CSCloud 2026 reproducibility artifact

Artifact for *Attesting Outputs and Delegation Ancestry in Multi-Agent AI
Systems* (IEEE CSCloud 2026). It contains the runnable sources and recorded
results behind the paper's experiments: B1--B4 structural matrices, E10--E12
deployment harnesses, E11--AWS, and E13 authenticated fanout admission.
`CLAIMS.md` maps the paper's experimental claims to exact files and commands.

## Quick start

Use Python 3.11. Run these commands from the extracted package root:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python tools/verify_artifact.py
python -B -m pytest -q -p no:cacheprovider tests
```

The integrity and result accounting check uses only the Python standard library.
The base test suite uses the dependencies in `requirements.txt`. No cloud account,
GPU, model endpoint, or network service is needed for these checks or E13 replay.

## E13: authenticated fanout admission

The recorded run is in
`experiments/e13_authenticated_fanout/results/formal_20260913/`.
Replay its original signed requests using the frozen executable sources:

```sh
python -B experiments/e13_authenticated_fanout/results/formal_20260913/source_snapshot/e13/run.py replay \
  --source experiments/e13_authenticated_fanout/results/formal_20260913 \
  --out artifacts/reproduced/e13_replay_new
```

For focused tests and a fresh execution, run from the E13 directory:

```sh
cd experiments/e13_authenticated_fanout
python -B -m pytest -q -p no:cacheprovider tests
python -B run.py run --seeds 17 29 43 --repeats 10 --out results/formal_new
```

Always choose a new output directory; do not run with Python `-O`.
Seeds control candidate ordering; keys and nonces use fresh OS randomness.
Validity checks use the recorded fixture clock, not the current wall clock.

The formal run contains 360 sequences and 3,420 distinct candidates. The
cryptographic control accepts all 3,420; admission accepts 2,340 within-limit
candidates and rejects 1,080 over-limit candidates, across all three designs.
These are 6,840 paired calls. The 7,173 log records also include grouped race,
restart, and crash events; they are not 7,173 independent attacks or verifier calls.
Replay checks serial verdicts and state, linearizes races to their recorded
winners, and checks crash records as unchanged committed state. A fresh `run`
executes new process races and killed-writer checks.

The quota counts distinct child receipts per immutable signed parent receipt
within one shared local SQLite store. Reissued parent receipts establish new
scopes. This does not limit prior execution, unobserved children, malicious
intent, or totals across independent stores. LL, Merkle, and Signed-DAG use the
same admission policy; this experiment does not establish superiority of one.

## Other paper results

| Evidence | Recorded files under `artifacts/current/` | Scope |
|---|---|---|
| Ancestry matrix | `asr_20260830_063346/asr_matrix.csv` | Use B1–B4 only: 12,000 checks; historical B5 rows are excluded |
| Complete-receipt matrix | `complete_receipt_asr_20260830_064451/asr_matrix.csv` | 12,000 B1–B4 checks |
| Ancestry timing | `perf_20260830_064113.csv` | Does not include E13 admission |
| LangGraph | `langgraph_20260830_064130.json` | Local framework integration |
| E10 / E11 | `multiprocess_cross_deployer_20260830/`, `container_cross_deployer_20260830/` | Process/container boundaries |
| E11 AWS | `e11_aws_20260831/rerun_01/results.json` | Three hosts; sanitized routing metadata; 1,000 trials |
| E12 | `e12_mcp_a2a_llm/results.json` | 30 tasks per condition; local MCP/A2A/Ollama services |

Reproduce the two structural matrices from the package root:

```sh
python scripts/paper2_asr_matrix.py --trials 1000 --verification-mode ancestry \
  --attacks B1 B2 B3 B4 --output-dir artifacts/reproduced/ancestry_new
python scripts/paper2_asr_matrix.py --trials 1000 --verification-mode complete \
  --attacks B1 B2 B3 B4 --output-dir artifacts/reproduced/complete_new
```

E12 needs a local Ollama service and the MCP dependency:

```sh
pip install 'mcp>=1.27,<2'
ollama pull llama3.1:latest
python -m experiments.e12_mcp_a2a_llm.run run --trials 30 --warmup 10 \
  --model llama3.1:latest --out artifacts/reproduced/e12_new/results.json
```

The model tag is mutable; compare the downloaded model digest with the recorded
configuration. Exact latency and model text need not repeat on another system.
The pinned LangGraph dependency is included in `requirements.txt`.
See `infra/e11_aws/README.md` for cloud reproduction using your own account and
`experiments/e11_container_cross_deployer/compose.yaml` for Docker reproduction.
Neither cloud nor model experiments are run by the quick checks.

E10 and container E11 are supporting deployment harnesses for the AWS experiment.
Unrelated historical experiment entry points are excluded. Shared library code,
its regression tests and E13's complete frozen dependency snapshot are retained.
The old B5 unsigned threshold rows in the ancestry CSV remain historical records;
they are excluded from E13 and the 24,000 structural checks. No historical
exploratory LLM result aggregates are distributed.

## Integrity and sanitization

`MANIFEST.json` binds every distributed file except itself. `ANONYMIZATION.json`
records local path redactions. No author manuscript, Git history, credentials,
private signing keys, cloud state, operational topology, or review correspondence
is distributed. E10/E11 public copies retain trial measurements while removing
environment-routing metadata.

E13 `events.jsonl` is byte-identical to the original signed corpus (SHA-256
`6792c70c7b1ccb8d5b7df84b8ed98eff43bc2cc8346aee6f5878f4c9c6206a64`).
All executable files in its snapshot retain their original hashes. One snapshot
validation document has its local repository path redacted. The original
`source_hashes.json` is retained as a historical record; use
`public_snapshot_hashes.json` for the distributed snapshot, together with the
documented redaction. The upstream 486-file manifest is omitted because it also
describes SQLite stores omitted here. Replay reconstructs fresh stores.

`VALIDATION.json` records checks actually run on this package; it distinguishes
new package validation from original experimental measurements.
