"""Adaptive Adversary Loop — Paper 2 §5.4 / §7.

Runs 100 rounds of LLM-driven adversary (default: Haiku 3.5) against
a FIXED defender (no defender patches between rounds). The adversary
reads the chain design source code, the corpus of prior successful
attacks, and prior round outcomes, then proposes 20 attack variants.

Protocol (kickoff §7, pre-reg §3.1):
  For each chain design D in {linked_list, merkle, signed_dag}:
    For each attack A in {B1, B2, B3, B4, B5}:
      corpus = {hand-crafted seed attacks from Week 2}
      for round in 0..99:
        attacker reads: source(D), corpus, prior outcomes
        attacker proposes 20 variants (chain-manipulating Python snippets)
        execute each against verify_chain; ASR_round = successes / 20
        successful variants append to corpus
      emit: ASR_round[0..99] for design D, attack A

Output:
  - results/paper2/adaptive/asr_curves.csv          — per-round ASR
  - results/paper2/adaptive/api_logs/*.jsonl         — full API logs
  - results/paper2/adaptive/cost_summary.txt         — budget accounting

Key differences from the LLM red-team harness (src/redteam/llm_redteam.py):
  - This targets CHAIN DESIGNS (B1–B5), not lifecycle events.
  - Single attacker model (Haiku 3.5 by default; others available for ablation).
  - The attacker generates chain-manipulation code, not payload forges.
  - Budget cap: $100 per chain design ($300 total).

Dependencies: anthropic>=0.99.0, tenacity
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

CHAIN_DESIGNS = ("linked_list", "merkle", "signed_dag")
BASELINE_DESIGNS = ("no_chain", "hash_only", "naive_sig")
ALL_DESIGNS = CHAIN_DESIGNS + BASELINE_DESIGNS
ATTACKS = ("B1", "B2", "B3", "B4", "B5")

ATTACK_DESCRIPTIONS = {
    "B1": "Orphan-child injection: forge a child whose parent_id references a never-created node",
    "B2": "Chain-break splice: remove a middle node and relink endpoints directly",
    "B3": "Cross-chain merge: graft a node from chain X into chain Y",
    "B4": "Ancestor revision: mutate an already-extended ancestor's payload",
    "B5": "Sibling fanout obfuscation: create many legitimate siblings to hide one malicious",
}

VARIANTS_PER_ROUND = 20
DEFAULT_ROUNDS = 100
PRNG_SEEDS: tuple[int, ...] = (42, 137, 271, 1729, 65537)

# Budget caps per design — pre-registered at $100/design
DESIGN_BUDGET_CAP = 100.0
TOTAL_BUDGET_CAP = 300.0

# Default attacker model — Haiku 3.5 (pre-registered §3.1)
# Model IDs use the Bedrock cross-region inference prefix (us.) required by
# the Salesforce AI gateway proxy.
DEFAULT_ATTACKER = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
DEFAULT_ATTACKER_NAME = "haiku_3_5"

# Pricing for cost estimation (per 1M tokens)
PRICING = {
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {"input_per_m": 1.0, "output_per_m": 5.0},
    "us.anthropic.claude-opus-4-20250514-v1:0": {"input_per_m": 15.0, "output_per_m": 75.0},
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {"input_per_m": 3.0, "output_per_m": 15.0},
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"input_per_m": 3.0, "output_per_m": 15.0},
}


# ============================================================================
# System Prompt — attack-oriented (different from lifecycle red-team)
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """\
You are a security researcher attempting to forge or manipulate ancestry chains \
in a multi-agent AI identity system. Your goal is to construct chain \
manipulations that bypass the verifier's chain-shape checks.

## Chain Design Under Attack

Design: {design}

## Attack Type

{attack_code}: {attack_description}

## Source Code (verifier + chain backend)

{source_code}

## Seed Attacks (hand-crafted B1–B5 from the fixed-adversary baseline)

{seed_attacks}

## Prior Successful Attacks (from this adaptive loop)

{corpus}

## Prior Round Outcomes

{outcomes}

## Instructions

Propose exactly {n_variants} DISTINCT attack variants as executable Python snippets.

Each snippet MUST:
1. Import from the project's modules (src.core.*, src.defenses.*, src.attacks.*)
2. Build a benign chain using MultiDeployerHarness + ChainBuilder
3. Apply a chain manipulation that targets the {attack_code} attack category
4. The manipulation should attempt to bypass verify_chain() for the {design} backend
5. Be syntactically valid Python wrapped in ```python ... ``` blocks

Strategies to explore:
- Exploit edge cases in the {design} verification logic
- Construct payloads that are structurally valid but semantically fraudulent
- Find gaps between what chain_proof enforces and what verify_chain checks
- Try manipulations that preserve some invariants while violating others
- Attempt timing-based or ordering-based attacks on the chain structure

Number each variant: # Variant 1, # Variant 2, ... # Variant {n_variants}
"""

# Additional context when key-compromise mode is active (E3)
KEY_COMPROMISE_ADDENDUM = """

## KEY COMPROMISE MODE (E3 Ablation)

You have access to a compromised deployer's PRIVATE signing key.
In your code, the following variables are pre-loaded:

- `compromised_harness` — a MultiDeployerHarness with deployers "D_A" and "D_B"
- `compromised_key` — the KeyPair for "D_A" (includes .private for signing)
- `compromised_deployer_name` — "D_A"
- `sign_fn(private_key, data_bytes)` — returns hex signature string

You can RE-SIGN mutated payloads using the compromised key:
```python
# Example: re-sign a mutated payload
import dataclasses
mutated_payload = dataclasses.replace(original_payload, output_hash="...")
new_sig = sign_fn(compromised_key.private, mutated_payload.to_canonical_json())
re_signed = dataclasses.replace(original_signed, payload=mutated_payload, signature_hex=new_sig)
```

Your goal: exploit the compromised key to perform {attack_code} attacks that
bypass verify_chain() even when chain-shape proofs are checked. This tests
whether chain designs degrade gracefully under single-deployer key compromise.
"""


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AdaptiveRoundResult:
    """One round's outcome for a single (design × attack) cell."""

    design: str
    attack: str
    round_idx: int
    seed: int
    n_variants_parsed: int
    n_successes: int
    asr: float
    successful_variants: list[str] = field(default_factory=list)

    # API usage
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    status: str = "success"
    error: Optional[str] = None


@dataclass
class DesignBudget:
    """Per-design cost tracker."""

    spent: float = 0.0

    def add(self, cost: float) -> None:
        self.spent += cost

    def within_budget(self) -> bool:
        return self.spent < DESIGN_BUDGET_CAP


# ============================================================================
# Source Code + Seed Attack Collector
# ============================================================================

def collect_chain_source(repo_root: Path, design: str) -> str:
    """Collect the source code the attacker sees (white-box)."""
    files = [
        "src/core/agent_id_v2.py",
        "src/core/crypto.py",
        "src/defenses/chain_designs/base.py",
        f"src/defenses/chain_designs/{design}.py",
        "src/core/verifier.py",
        "src/runtime/multi_deployer.py",
    ]
    parts: list[str] = []
    for rel in files:
        full = repo_root / rel
        if full.exists():
            parts.append(f"### {rel}\n```python\n{full.read_text()}\n```")
        else:
            parts.append(f"### {rel}\n(not yet implemented)")
    return "\n\n".join(parts)


def collect_seed_attacks(repo_root: Path) -> str:
    """Collect the hand-crafted B1–B5 attack implementations as seed context."""
    attack_files = [
        "src/attacks/ancestry/b1_orphan.py",
        "src/attacks/ancestry/b2_splice.py",
        "src/attacks/ancestry/b3_cross_chain.py",
        "src/attacks/ancestry/b4_ancestor_revision.py",
        "src/attacks/ancestry/b5_sibling_fanout.py",
    ]
    parts: list[str] = []
    for rel in attack_files:
        full = repo_root / rel
        if full.exists():
            parts.append(f"### {rel}\n```python\n{full.read_text()}\n```")
    return "\n\n".join(parts)


# ============================================================================
# Variant Parser + Executor (stub — Week 3 fills executor)
# ============================================================================

def parse_variants(response_text: str) -> list[str]:
    """Extract ```python ... ``` blocks from attacker response."""
    variants: list[str] = []
    in_block = False
    current: list[str] = []

    for line in response_text.split("\n"):
        if line.strip().startswith("```python"):
            in_block = True
            current = []
        elif line.strip() == "```" and in_block:
            in_block = False
            code = "\n".join(current).strip()
            if code:
                variants.append(code)
        elif in_block:
            current.append(line)

    return variants


def execute_attack_variant(
    variant_code: str,
    design: str,
    attack: str,
    *,
    compromised_deployer: Optional[str] = None,
) -> bool:
    """Execute one attack variant in a sandboxed namespace.

    The sandbox (src.adversary.sandbox) provides:
      - Project imports (src.core.*, src.defenses.*, src.attacks.*, src.runtime.*)
      - Pre-loaded Backend class and backend instance for the target design
      - Convenience aliases (BoundIDPayloadV2, ChainNode, ChainBuilder, etc.)
      - Safe stdlib (dataclasses, copy, hashlib, json, uuid)
      - 5-second timeout via signal.alarm
      - Blocked: os, subprocess, sys, socket, open, exec, eval, __import__

    In key-compromise mode (E3), the attacker also receives:
      - compromised_key: KeyPair with .private for one deployer
      - compromised_harness: MultiDeployerHarness with known keys
      - sign_fn: sign(private_key, data) convenience function

    The attacker code should set ATTACK_RESULT = True if the manipulated
    chain passes verify_chain(). If the variable is not set or the code
    raises an exception, the attack is considered to have failed.

    Returns True iff the attack bypasses all chain-shape checks.
    """
    from .sandbox import execute_in_sandbox
    return execute_in_sandbox(
        variant_code, design,
        compromised_deployer=compromised_deployer,
    )


# ============================================================================
# API Client
# ============================================================================

def _resolve_ca_bundle() -> str | bool:
    """Resolve the CA certificate bundle for the Bedrock proxy.

    Priority: SSL_CERT_FILE → NODE_EXTRA_CA_CERTS → ~/.devbar/certs/ → True (default).
    """
    for env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"):
        path = os.environ.get(env_var)
        if path and os.path.isfile(path):
            return path
    # Salesforce devbar default location
    devbar_cert = os.path.expanduser("~/.devbar/certs/corporate-ca-bundle.pem")
    if os.path.isfile(devbar_cert):
        return devbar_cert
    return True  # use system default


def create_client() -> anthropic.AnthropicBedrock | anthropic.Anthropic:
    """Create Anthropic client (Bedrock proxy or standard).

    Auth priority:
      1. ANTHROPIC_BEDROCK_BASE_URL → Bedrock proxy; uses ANTHROPIC_AUTH_TOKEN
         as api_key for the gateway (no AWS credential chain needed).
      2. Fallback → standard Anthropic client (uses ANTHROPIC_API_KEY).

    SSL: The Salesforce Bedrock proxy uses an internal CA. We auto-detect
    the corporate CA bundle from ~/.devbar/certs/ or env vars.
    """
    import httpx

    bedrock_url = os.environ.get("ANTHROPIC_BEDROCK_BASE_URL")
    if bedrock_url:
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        ca_bundle = _resolve_ca_bundle()
        http_client = httpx.Client(verify=ca_bundle)
        return anthropic.AnthropicBedrock(
            aws_region="us-east-2",
            base_url=bedrock_url,
            http_client=http_client,
            # The Salesforce Bedrock proxy authenticates via bearer token,
            # not AWS IAM. Pass the token as api_key to skip the default
            # AWS credential chain (which fails on non-EC2 hosts).
            **({"api_key": auth_token} if auth_token else {}),
        )
    return anthropic.Anthropic()


@retry(
    retry=retry_if_exception_type((
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_attacker(
    client: anthropic.AnthropicBedrock | anthropic.Anthropic,
    model_id: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4096,
) -> anthropic.types.Message:
    """Single API call with retry + backoff + prompt caching."""
    return client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )


# ============================================================================
# Single Round
# ============================================================================

def run_adaptive_round(
    *,
    client: anthropic.AnthropicBedrock | anthropic.Anthropic,
    model_id: str,
    design: str,
    attack: str,
    round_idx: int,
    seed: int,
    corpus: list[str],
    prior_outcomes: list[dict[str, Any]],
    source_code: str,
    seed_attacks: str,
    budget: DesignBudget,
    log_dir: Path,
    compromised_deployer: Optional[str] = None,
) -> AdaptiveRoundResult:
    """Run one round of the adaptive adversary loop."""

    if not budget.within_budget():
        logger.warning(
            "Budget exceeded for %s ($%.2f spent); skipping round %d",
            design, budget.spent, round_idx,
        )
        return AdaptiveRoundResult(
            design=design, attack=attack, round_idx=round_idx, seed=seed,
            n_variants_parsed=0, n_successes=0, asr=0.0,
            status="budget_exceeded",
        )

    # Build prompts
    corpus_text = (
        "\n\n".join(f"```python\n{c}\n```" for c in corpus[-10:])
        if corpus else "(none yet — this is the first round)"
    )
    outcomes_text = (
        json.dumps(prior_outcomes[-5:], indent=2)
        if prior_outcomes else "(first round — no prior outcomes)"
    )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        design=design,
        attack_code=attack,
        attack_description=ATTACK_DESCRIPTIONS.get(attack, attack),
        source_code=source_code,
        seed_attacks=seed_attacks,
        corpus=corpus_text,
        outcomes=outcomes_text,
        n_variants=VARIANTS_PER_ROUND,
    )

    # E3: Append key-compromise instructions if active
    if compromised_deployer:
        system_prompt += KEY_COMPROMISE_ADDENDUM.format(attack_code=attack)

    user_message = (
        f"Round {round_idx + 1}/{DEFAULT_ROUNDS}. "
        f"Design: {design}. Attack: {attack}. "
        f"Corpus size: {len(corpus)} successful exploit(s) so far. "
        f"Propose {VARIANTS_PER_ROUND} new, distinct attack variants."
    )

    # Make API call
    t0 = time.monotonic()
    try:
        response = _call_attacker(
            client=client,
            model_id=model_id,
            system_prompt=system_prompt,
            user_message=user_message,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        # Extract usage
        input_tok = response.usage.input_tokens
        output_tok = response.usage.output_tokens
        cached_tok = getattr(response.usage, "cache_read_input_tokens", 0) or 0

        # Estimate cost
        pricing = PRICING.get(model_id, PRICING[DEFAULT_ATTACKER])
        cost = (
            (input_tok / 1_000_000) * pricing["input_per_m"]
            + (output_tok / 1_000_000) * pricing["output_per_m"]
        )
        budget.add(cost)

        # Extract response text
        response_text = "".join(
            b.text for b in response.content if b.type == "text"
        )

    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.error("API call failed: %s", e)
        return AdaptiveRoundResult(
            design=design, attack=attack, round_idx=round_idx, seed=seed,
            n_variants_parsed=0, n_successes=0, asr=0.0,
            latency_ms=latency_ms, status="error", error=str(e),
        )

    # Parse and execute
    variants = parse_variants(response_text)
    n_successes = 0
    successful: list[str] = []

    for v in variants:
        try:
            if execute_attack_variant(
                v, design, attack,
                compromised_deployer=compromised_deployer,
            ):
                n_successes += 1
                successful.append(v)
        except Exception:
            pass  # malformed code → attack fails

    asr = n_successes / max(len(variants), 1)

    # Log
    _write_round_log(
        log_dir, design, attack, round_idx, seed,
        model_id, input_tok, output_tok, cached_tok,
        cost, latency_ms, len(variants), n_successes, response_text,
    )

    logger.info(
        "[%s/%s] round %3d: %d variants, %d successes, ASR=%.3f ($%.4f, %.0fms)",
        design, attack, round_idx, len(variants), n_successes, asr, cost, latency_ms,
    )

    return AdaptiveRoundResult(
        design=design, attack=attack, round_idx=round_idx, seed=seed,
        n_variants_parsed=len(variants), n_successes=n_successes, asr=asr,
        successful_variants=successful,
        input_tokens=input_tok, output_tokens=output_tok,
        cached_tokens=cached_tok, estimated_cost_usd=cost,
        latency_ms=latency_ms,
    )


# ============================================================================
# Full Adaptive Loop
# ============================================================================

def run_adaptive_benchmark(
    *,
    repo_root: Path,
    output_dir: Path,
    designs: Sequence[str] | None = None,
    attacks: Sequence[str] | None = None,
    n_rounds: int = DEFAULT_ROUNDS,
    seeds: Sequence[int] | None = None,
    model_id: str = DEFAULT_ATTACKER,
    dry_run: bool = False,
    compromised_deployer: Optional[str] = None,
) -> Path:
    """Run the full adaptive adversary benchmark.

    Args:
        repo_root: Path to agent-id-experiment repo root.
        output_dir: Output directory for results.
        designs: Chain designs to test (default: all 3 primary).
        attacks: Attacks to run (default: all 5).
        n_rounds: Rounds per cell (default: 100).
        seeds: PRNG seeds (default: pre-registered 5).
        model_id: Attacker model (default: Haiku 3.5).
        dry_run: If True, log the plan only.
        compromised_deployer: If set, run in key-compromise mode (E3).
            The attacker receives this deployer's private key.

    Returns:
        Path to the ASR curves CSV.
    """
    if designs is None:
        designs = CHAIN_DESIGNS
    if attacks is None:
        attacks = ATTACKS
    if seeds is None:
        seeds = PRNG_SEEDS

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "api_logs"
    log_dir.mkdir(exist_ok=True)

    total_cells = len(designs) * len(attacks) * len(seeds)
    total_calls = total_cells * n_rounds

    logger.info(
        "Adaptive adversary plan: %d designs × %d attacks × %d seeds × "
        "%d rounds = %d API calls (model: %s)",
        len(designs), len(attacks), len(seeds), n_rounds, total_calls, model_id,
    )

    if dry_run:
        _write_plan(output_dir, designs, attacks, seeds, n_rounds, model_id, total_calls)
        return output_dir / "asr_curves.csv"

    # Collect source code and seed attacks
    source_cache: dict[str, str] = {}
    for d in designs:
        source_cache[d] = collect_chain_source(repo_root, d)
    seed_attacks = collect_seed_attacks(repo_root)

    client = create_client()

    # CSV output
    csv_path = output_dir / "asr_curves.csv"
    csv_fields = [
        "design", "attack", "seed", "round", "n_variants",
        "n_successes", "asr", "cumulative_corpus_size",
        "input_tokens", "output_tokens", "cached_tokens",
        "estimated_cost_usd", "latency_ms", "status",
    ]

    budgets: dict[str, DesignBudget] = {d: DesignBudget() for d in designs}

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for design in designs:
            for attack in attacks:
                for seed in seeds:
                    # Per-cell state
                    corpus: list[str] = []
                    prior_outcomes: list[dict[str, Any]] = []

                    for round_idx in range(n_rounds):
                        result = run_adaptive_round(
                            client=client,
                            model_id=model_id,
                            design=design,
                            attack=attack,
                            round_idx=round_idx,
                            seed=seed,
                            corpus=corpus,
                            prior_outcomes=prior_outcomes,
                            source_code=source_cache[design],
                            seed_attacks=seed_attacks,
                            budget=budgets[design],
                            log_dir=log_dir,
                            compromised_deployer=compromised_deployer,
                        )

                        # Update corpus
                        corpus.extend(result.successful_variants)
                        prior_outcomes.append({
                            "round": round_idx,
                            "asr": result.asr,
                            "n_successes": result.n_successes,
                            "n_variants": result.n_variants_parsed,
                        })

                        # Write CSV row
                        writer.writerow({
                            "design": design,
                            "attack": attack,
                            "seed": seed,
                            "round": round_idx,
                            "n_variants": result.n_variants_parsed,
                            "n_successes": result.n_successes,
                            "asr": f"{result.asr:.4f}",
                            "cumulative_corpus_size": len(corpus),
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "cached_tokens": result.cached_tokens,
                            "estimated_cost_usd": f"{result.estimated_cost_usd:.6f}",
                            "latency_ms": f"{result.latency_ms:.1f}",
                            "status": result.status,
                        })
                        f.flush()

                        if not budgets[design].within_budget():
                            logger.warning(
                                "Budget exceeded for %s; skipping remaining rounds.",
                                design,
                            )
                            break

    # Write cost summary
    _write_cost_summary(output_dir, budgets, model_id)

    logger.info("Adaptive benchmark complete. Results: %s", csv_path)
    return csv_path


# ============================================================================
# Output Helpers
# ============================================================================

def _write_round_log(
    log_dir: Path, design: str, attack: str, round_idx: int, seed: int,
    model_id: str, input_tok: int, output_tok: int, cached_tok: int,
    cost: float, latency_ms: float, n_variants: int, n_successes: int,
    response_text: str,
) -> None:
    """Append one round's data to a JSONL log."""
    log_path = log_dir / f"{design}_{attack}_seed{seed}.jsonl"
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_id": model_id,
        "design": design,
        "attack": attack,
        "round": round_idx,
        "seed": seed,
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "cached_tokens": cached_tok,
        "estimated_cost_usd": cost,
        "latency_ms": latency_ms,
        "n_variants": n_variants,
        "n_successes": n_successes,
    }
    # Truncate response for storage
    if len(response_text) > 500:
        entry["response_hash"] = hashlib.sha256(response_text.encode()).hexdigest()[:16]
        entry["response_preview"] = response_text[:500]
    else:
        entry["response_text"] = response_text

    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _write_cost_summary(
    output_dir: Path,
    budgets: dict[str, DesignBudget],
    model_id: str,
) -> None:
    summary_path = output_dir / "cost_summary.txt"
    lines = [
        "Adaptive Adversary — Cost Summary",
        f"Model: {model_id}",
        f"Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        "",
        f"{'Design':<15} {'Spent':>10} {'Cap':>10} {'Status':>10}",
        "-" * 50,
    ]
    total = 0.0
    for design, budget in budgets.items():
        status = "OK" if budget.within_budget() else "EXCEEDED"
        lines.append(
            f"{design:<15} ${budget.spent:>8.2f} ${DESIGN_BUDGET_CAP:>8.2f} {status:>10}"
        )
        total += budget.spent
    lines.append("-" * 50)
    lines.append(
        f"{'TOTAL':<15} ${total:>8.2f} ${TOTAL_BUDGET_CAP:>8.2f} "
        f"{'OK' if total < TOTAL_BUDGET_CAP else 'EXCEEDED':>10}"
    )
    summary_path.write_text("\n".join(lines) + "\n")


def _write_plan(
    output_dir: Path,
    designs: Sequence[str],
    attacks: Sequence[str],
    seeds: Sequence[int],
    n_rounds: int,
    model_id: str,
    total_calls: int,
) -> None:
    plan = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_id": model_id,
        "designs": list(designs),
        "attacks": list(attacks),
        "seeds": list(seeds),
        "n_rounds": n_rounds,
        "variants_per_round": VARIANTS_PER_ROUND,
        "total_api_calls": total_calls,
        "total_variants": total_calls * VARIANTS_PER_ROUND,
        "design_budget_cap": DESIGN_BUDGET_CAP,
        "total_budget_cap": TOTAL_BUDGET_CAP,
    }
    plan_path = output_dir / "benchmark_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    logger.info("Plan written to %s", plan_path)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    """CLI for running the adaptive adversary benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Paper 2 — Adaptive adversary loop (Haiku 3.5 default)",
    )
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output dir (default: repo/results/paper2/adaptive/)",
    )
    parser.add_argument(
        "--designs", nargs="+",
        choices=list(ALL_DESIGNS),
        default=list(CHAIN_DESIGNS),
        help="Chain designs to test. Default: 3 primary. "
             "Baselines: no_chain, hash_only, naive_sig.",
    )
    parser.add_argument(
        "--attacks", nargs="+",
        choices=list(ATTACKS),
        default=list(ATTACKS),
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS,
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_ATTACKER,
        help=f"Attacker model ID (default: {DEFAULT_ATTACKER})",
    )
    parser.add_argument(
        "--key-compromise", type=str, default=None, metavar="DEPLOYER",
        help="E3 ablation: expose this deployer's private key to the attacker. "
             "Use 'D_A' to compromise the first deployer.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = args.output_dir or (args.repo_root / "results" / "paper2" / "adaptive")

    run_adaptive_benchmark(
        repo_root=args.repo_root,
        output_dir=output_dir,
        designs=args.designs,
        attacks=args.attacks,
        n_rounds=args.rounds,
        seeds=args.seeds,
        model_id=args.model,
        dry_run=args.dry_run,
        compromised_deployer=args.key_compromise,
    )


if __name__ == "__main__":
    main()
