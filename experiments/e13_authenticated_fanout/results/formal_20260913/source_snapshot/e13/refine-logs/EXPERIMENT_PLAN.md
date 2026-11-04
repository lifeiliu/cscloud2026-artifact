# E13 authenticated fanout experiment plan

Date: 2026-09-13. Scope frozen before formal execution.

## Claim map

| Claim | Minimum evidence | Blocks |
|---|---|---|
| C1: For an immutable authenticated parent receipt, a local verifier admits at most its signed number of distinct child receipts. | Valid under-limit receipts accepted; valid overflow rejected only by the policy-enabled arm; tampered evidence rejected without state mutation. | 1, 2 |
| C2: Reusable historical ancestry, terminal replay, nonce consistency, and quotas compose transactionally across processes sharing one local persistent database. | No double counting, rejected operations leave no writes, last-slot and duplicate races have exactly one winner, restart and killed-writer recovery preserve state. | 3, 4 |

Anti-claim: successes merely count a constructed list above a configured threshold. Every new observation must run existing full receipt/ancestry checks. Parent limits come from authenticated payloads. No LLM, GPU, model-quality or production-global-policy claim.

## Semantics frozen before runs

- Unit: distinct child **receipts** under an immutable parent **receipt**, not all runtime agent instances or unobserved actions. Full verified public-key hash plus canonical payload identifies each receipt. Reissuing a parent creates a new scope; signer authorization of policy updates is outside this experiment.
- Revalidate all signatures, key admission/revocation, time, outputs and ancestry on every call, using the existing complete verifier.
- Historical ancestors may be presented again with exactly the same signer/nonce/payload. A signer reusing a nonce for a different receipt is rejected.
- A terminal receipt may be submitted once to this store; observing it later as an ancestor is allowed. First terminal submission of a previously observed ancestor is allowed, without charging existing edges twice.
- Edge registration, nonce-to-receipt registration and terminal submission commit in one `BEGIN IMMEDIATE` transaction; any failure rolls back the entire request.
- Signed `max_children`: null = unrestricted; nonnegative integer = limit; zero = no admitted children. Booleans, negative and noninteger limits rejected.
- This adds a new opt-in admission API. It does not silently change the old verifier's all-record single-use nonce semantics.
- One local SQLite file shared by processes, static trusted public registry. Independent databases, distributed consensus, key rotation and disk-loss recovery are excluded.
- Fixed fixture clock is recorded to permit offline replay of signed test receipts. Timing is not a production latency benchmark.

## Five must-run blocks

| Block | Setup | Metrics and success criterion | Failure interpretation | Paper target |
|---|---|---|---|---|
| 1. Authenticated boundary | LL/Merkle/S-DAG; limits 0,1,5,20; seeds 17,29,43; 10 fresh sequences/cell; limit+3 properly signed candidate children, shuffled order. | Exactly limit accepted and three `fanout_limit` rejections per sequence; no wrong decisions; final distinct-edge count = limit. | Quota or fixture failure. | Separate B5 results row/table |
| 2. Mechanism control | Same signed input, existing complete cryptographic verifier without quota state. | All candidates, including overflow, pass; paired arm differs only by policy admission. This arm is explicitly stateless, not an alternative replay policy. | If overflow fails crypto, the fixture does not isolate B5. | Mechanism attribution |
| 3. Composition/failure | Tampered signed limit, invalid primary/ancestry, output substitution, invalid policy, terminal replay, conflicting nonce, shared ancestors, same instance IDs across parents, revoked ancestor, late multi-hop quota failure. | Expected reason and full logical state equality on all failed requests; legitimate requests after failures still pass. | Authentication/state composition failure. | Reproducibility appendix/artifact |
| 4. Persistence/concurrency | Fresh worker processes, public keys only; restart, four processes competing for last slot, four replaying same terminal, worker killed with an uncommitted transaction. Three seeds/design for formal run. | Exact admission counts, no partial state, restart/killed-writer recovery matches committed snapshot. | Cross-process/atomicity claim unsupported. | Supporting state guarantee |
| 5. Independent replay/provenance | Export signed requests with public keys, predeclared expected outcomes, actual reason codes and before/after state digests; rerun corpus in fresh databases. | All judgments and state hashes reproduce; source hashes and versions retained. Unexpected errors abort, never count as rejections. | Evidence not ready. | Artifact manifest |

## Milestones and budget

| Stage | Gate | Compute estimate |
|---|---|---|
| Focused tests | All invariants, including failure cases, pass before formal experiment | CPU seconds to a few minutes |
| Smoke | One seed, one sequence/cell; all five blocks pass | CPU minutes |
| Formal | Three seeds, ten sequences/cell; zero mismatches, explicit abort accounting | CPU minutes; 0 GPU hours |
| Replay and handoff | Replay all logged serial judgments; retain concurrency/crash evidence separately | CPU minutes |

Boundary block has 360 sequences, 3,420 signed candidate presentations, and 6,840 paired verifier calls. Expected policy arm: 2,340 legitimate accepts and 1,080 quota rejections. These are coverage counts, not independent attacks sampled from an unknown distribution; no ASR confidence interval or cryptographic-security probability will be reported.

Additional adversarial schedules are conformance cases. Seed controls candidate order; cryptographic keys/nonces use OS randomness. Full signed public corpus enables exact replay without retaining private keys. Do not overwrite or relabel the old 3,000 B5 threshold trials.

## Risks

- Parent/ancestor nonce reuse conflicts with legacy single-use policy: explicitly implement the separate terminal-event model above.
- Invalid inputs poison quota: verify crypto first, then stage all state in one transaction.
- Baseline missing a relevant check: both arms call the identical existing complete verifier.
- Concurrency flakes: synchronize worker starts, require exact counts; unexpected process/database errors abort.
- Scope inflation: count receipts observed by one shared store, not globally spawned agents or malicious-child detection.

Nice-to-have (not required): production latency, distributed quota service, live LLM/AWS, updated paper figures. Keep the paper unchanged until the new evidence is available and separately reviewed.
