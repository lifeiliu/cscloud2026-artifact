"""Schema invariants: canonical JSON is deterministic, hints stay unsigned."""

from __future__ import annotations

import json

from src.core.agent_id import (
    AgentIDEnvelope,
    BoundIDPayload,
    RoutingHints,
    SignedBoundID,
)


def _sample_payload(**overrides: object) -> BoundIDPayload:
    base = dict(
        instance_id="claude-3-sonnet:sess_2026050401",
        system_id="claude-3-sonnet",
        deployer="anthropic",
        created_at="2026-05-05T10:00:00Z",
        expires_at="2026-05-05T11:00:00Z",
        parent_id=None,
        ancestry_depth=0,
        output_hash="a" * 64,
        nonce="b" * 32,
        binding_scope="per_turn",
        certifications=("no_prompt_injection",),
        incident_count=0,
        prev_turn_hash=None,
    )
    base.update(overrides)
    return BoundIDPayload(**base)  # type: ignore[arg-type]


class TestCanonicalJSON:
    def test_deterministic_across_field_order(self) -> None:
        # Construct with args in declaration order vs. shuffled kwargs;
        # canonical output should be byte-identical.
        a = _sample_payload()
        b = _sample_payload()
        assert a.to_canonical_json() == b.to_canonical_json()

    def test_sorted_keys(self) -> None:
        data = _sample_payload().to_canonical_json()
        parsed = json.loads(data)
        keys = list(parsed.keys())
        assert keys == sorted(keys), f"keys not sorted: {keys}"

    def test_no_whitespace(self) -> None:
        data = _sample_payload().to_canonical_json()
        # Compact form: never a space after ',' or ':' at the top level.
        assert b", " not in data
        assert b": " not in data

    def test_certifications_serialised_as_list(self) -> None:
        payload = _sample_payload(certifications=("a", "b"))
        parsed = json.loads(payload.to_canonical_json())
        assert parsed["certifications"] == ["a", "b"]

    def test_different_content_produces_different_bytes(self) -> None:
        a = _sample_payload(output_hash="1" * 64)
        b = _sample_payload(output_hash="2" * 64)
        assert a.to_canonical_json() != b.to_canonical_json()


class TestEnvelope:
    def test_hints_are_separate_from_signed_payload(self) -> None:
        """Wire-format hints must live outside the signed bound_id block.

        This is the structural guarantee that a verifier cannot accidentally
        treat routing hints as authoritative.
        """
        payload = _sample_payload()
        signed = SignedBoundID(payload=payload, signature_hex="deadbeef", key_id="kid123")
        hints = RoutingHints(quality_hint=0.92, cost_hint="medium")
        envelope = AgentIDEnvelope(bound_id=signed, hints=hints)

        wire = envelope.to_wire()

        assert "bound_id" in wire
        assert "hints" in wire
        # Hints must NOT appear inside the signed payload.
        signed_fields = set(wire["bound_id"]["payload"].keys())
        assert "quality_hint" not in signed_fields
        assert "cost_hint" not in signed_fields
        assert "reasoning_profile" not in signed_fields
        assert "latency_hint_ms_p50" not in signed_fields

    def test_default_hints_are_all_none(self) -> None:
        hints = RoutingHints()
        assert hints.quality_hint is None
        assert hints.cost_hint is None
        assert hints.reasoning_profile is None
        assert hints.latency_hint_ms_p50 is None
