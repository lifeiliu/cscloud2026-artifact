# E13: authenticated fanout admission

This experiment repairs the evidentiary gap in the historical B5 threshold
test. It adds an opt-in admission API that calls the existing complete
receipt/ancestry verifier, then enforces the authenticated parent receipt's
`max_children` using one transactional SQLite database. The original verifier,
old B5 checker, historical results and manuscript are not overwritten.

## Defined scope

- Count distinct child receipts under an immutable signed parent receipt.
  A fresh parent receipt, including one reissued under the same instance ID,
  is a fresh scope. This is not an agent-wide or globally distributed quota.
- Full verified public-key fingerprint plus canonical payload identifies a
  receipt. A signed null limit is unrestricted, zero admits no children, and
  other limits must be nonnegative integers (booleans are not integers here).
- Every call validates signatures, trusted/locally revoked keys, time, exact
  outputs and ancestry using `verify_complete_chain`.
- Identical historical ancestors may be reused. A terminal receipt may be
  submitted once. Different receipts reusing one signer's nonce are rejected.
  All receipt identities, distinct edges and terminal submissions are written
  in a single transaction, and failed calls leave no partial writes.
- Concurrency and restart support apply to processes sharing the same local
  SQLite file and static trusted registry. Do not place the database on a
  network filesystem or interpret independent databases as global enforcement.
- `expected_leaf` can pin a target receipt identity. Without it, admission
  verifies the presented path, not the caller's unspecified intended leaf.
- No claim of detecting malicious intent, preventing tool/model execution,
  counting unobserved children, production latency, or distributed revocation.

The fixture clock is fixed at `2026-09-13T12:00:00+00:00` for replayable validity
checks. Keys and nonces are fresh OS randomness. Seeds govern candidate order,
not key generation. Only public keys and signed requests are exported.

## Files

- `verifier.py`: actual full-validation-plus-state admission API.
- `fixtures.py`: properly signed LL/Merkle/S-DAG receipts and public wire codec.
- `scenarios.py`: authentication, policy and state failure sequences.
- `run.py`: boundary/control matrix, process races, killed-writer checks and replay.
- `tests/test_verifier.py`: focused regression checks.
- `refine-logs/EXPERIMENT_PLAN.md`: claims, expected outcomes, run budget and limits.

## Run from this directory

Use Python 3.11 with the existing repository dependencies. Recorded environment
uses cryptography 48.0.1, SQLite 3.45.1 and pytest 8.1.1; see each configuration
file for the actual environment. No GPU, LLM endpoint, cloud account or network
access is needed.

```bash
python3 -B -m pytest -q -p no:cacheprovider tests
python3 -B run.py run --seeds 17 --repeats 1 --out results/smoke_new
python3 -B run.py run --seeds 17 29 43 --repeats 10 --out results/formal_new
python3 -B run.py replay --source results/formal_new --out results/replay_new
```

Output directories must not already exist. Unexpected exceptions abort the run
and produce a failed summary; they are never counted as successful defenses.
Assertions are part of the experiment oracle: do not run with Python `-O`.

## Replay and provenance

Each run writes `configuration.json`, `source_hashes.json`, `source_snapshot/`,
`events.jsonl`, `summary.json`, and per-sequence SQLite stores. Events include
signed inputs, public keys, local revocations, expected and actual rejection
codes, and logical state digests before/after each operation. Private keys are
not recorded. The snapshot contains the exact dependency `src/` tree and E13
implementation used at run start.

For replay independent of later repository edits, invoke the archived runner:

```bash
python3 -B results/formal_new/source_snapshot/e13/run.py replay \
  --source results/formal_new --out results/archived_replay_new
```

Replay reconstructs exact serial verdicts and committed state in fresh stores.
Races are replayed using their observed winning commit first; it does not
recreate OS scheduling. Killed-writer records are checked as unchanged committed
state. Fresh concurrent/crash execution is performed by `run`, separately.

The boundary block has 360 sequences and 3,420 signed candidate presentations:
2,340 within-limit candidates and 1,080 overflow candidates in the policy arm.
The same inputs also go to a stateless cryptographic-only control, for 6,840
paired calls. Report observed results from `summary.json`, not these planned
counts. Do not add these to the old 3,000 B5 threshold trials or interpret a
zero-mismatch conformance run as an attack-distribution confidence interval.
