"""AgentID v2 schema: bound identity with output binding and nonce anti-replay.

Design rationale (Research Foundation v4 §4.1):
  - Security-critical fields are signed as one unit (Ed25519 over canonical JSON).
  - Routing hints (quality_hint, latency_hint_ms_p50, cost_hint, reasoning_profile)
    are a *separate* envelope and are NEVER signed. This is the LDP-derived
    "don't let self-reported metadata be authoritative" rule.

Including routing hints in the signed token would create:
  1. Side channels (observable quality_hint correlates with output patterns)
  2. Confused-deputy attacks where unsigned metadata is attached to a signed token

This module defines only the data shapes. Signing/verification lives in crypto.py;
issuance lives in deployer.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


BindingScope = Literal["per_session", "per_turn", "per_tool_call", "per_n_tokens"]


@dataclass(frozen=True)
class BoundIDPayload:
    """The security-critical fields. These are what the deployer signs.

    Hashes and IDs are stored as lowercase hex. Timestamps are RFC 3339 UTC.
    """

    # Identity
    instance_id: str           # e.g. "claude-3-sonnet:sess_2026050401"
    system_id: str             # e.g. "claude-3-sonnet"
    deployer: str              # e.g. "anthropic"

    # Temporal bounds
    created_at: str            # RFC 3339 UTC, e.g. "2026-05-04T10:00:00Z"
    expires_at: str

    # Ancestry
    parent_id: Optional[str]   # None for root; otherwise the parent's instance_id
    ancestry_depth: int        # 0 for root. Verifier MUST cross-check this against
                               # the presented chain length (closes drop-hops attack).

    # Behavioral binding
    output_hash: str           # SHA-256 hex of the bound content for this scope
    nonce: str                 # random 128-bit hex; single-use; anti-replay
    binding_scope: BindingScope

    # Attributes the verifier may rely on
    certifications: tuple[str, ...] = ()     # e.g. ("no_prompt_injection",)
    incident_count: int = 0

    # Multi-turn chain linkage (see §4.3 "Multi-turn linking")
    # SHA-256 hex of the previous turn's canonical payload; None on turn 0.
    prev_turn_hash: Optional[str] = None

    def to_canonical_json(self) -> bytes:
        """Deterministic JSON encoding suitable for signing.

        Contract: two payloads that are semantically equal MUST produce byte-identical
        output. Sorting keys + no whitespace + ensure_ascii guarantees this across
        Python versions. Tuples are serialised as JSON arrays.
        """
        d = asdict(self)
        # dataclass asdict preserves tuples as tuples; json dumps them as arrays,
        # but we normalise to list for explicitness.
        d["certifications"] = list(self.certifications)
        return json.dumps(
            d,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")


@dataclass(frozen=True)
class RoutingHints:
    """Unsigned, advisory metadata. Never treated as authoritative.

    Lives in a separate envelope from the signed BoundIDPayload. Verifiers
    MUST NOT grant security decisions on the basis of these fields.
    """

    quality_hint: Optional[float] = None         # [0.0, 1.0]
    latency_hint_ms_p50: Optional[int] = None
    cost_hint: Optional[Literal["low", "medium", "high"]] = None
    reasoning_profile: Optional[str] = None      # e.g. "chain-of-thought"


@dataclass(frozen=True)
class SignedBoundID:
    """A BoundIDPayload plus its Ed25519 signature and the signer's key id.

    The wire format is `(payload_canonical_json, signature_hex, key_id)`.
    `key_id` lets the verifier pick the right public key (supports key rotation
    per §4.5 of the foundation).
    """

    payload: BoundIDPayload
    signature_hex: str
    key_id: str

    def to_wire(self) -> dict[str, Any]:
        """Wire-format dict for transport. Routing hints travel separately."""
        return {
            "payload": json.loads(self.payload.to_canonical_json().decode("utf-8")),
            "signature_hex": self.signature_hex,
            "key_id": self.key_id,
        }


@dataclass(frozen=True)
class AgentIDEnvelope:
    """Top-level envelope combining the signed bound ID and unsigned hints.

    The two are distinct top-level fields so a consumer cannot accidentally
    conflate them: `envelope.bound_id` is authoritative, `envelope.hints` is not.
    """

    bound_id: SignedBoundID
    hints: RoutingHints = field(default_factory=RoutingHints)

    def to_wire(self) -> dict[str, Any]:
        return {
            "bound_id": self.bound_id.to_wire(),
            "hints": asdict(self.hints),
        }
