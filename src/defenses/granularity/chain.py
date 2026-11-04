"""Chain verification for bound event streams (E2).

The agent-side story:
  1. The scripted/real runtime emits an EventStream.
  2. A BindingStrategy slices it into BoundSegments at its granularity.
  3. The deployer signs each segment (fresh nonce, chained to prev segment's
     canonical payload hash).
  4. The (presented_events, [signed_bound_ids]) pair ships to the consumer +
     verifier.

The attacker sits between step 3 and step 4: they may mutate the events.

The verifier-side story:
  1. Receive (presented_events, signed_ids).
  2. Run the SAME strategy over presented_events to recompute segments.
  3. For each segment, check:
     - sha256(canonical_bytes(segment)) == signed_id.payload.output_hash
     - signed_id.payload.prev_turn_hash == hash of prev segment's payload
     - signature, expiry, nonce (standard bound_id checks)
  4. ASR = trials where at least one tampered segment was consumer-visible
     BEFORE any verification failure covering it.

For the deterministic matrix, we simplify consumer-visibility to:

  "A tampered event is consumer-visible UNLESS there is a bound_id segment
   that (a) covers that event and (b) fails to verify."

If the strategy never binds the event (e.g. G1 only binds at session end,
so the consumer sees each turn long before verification), a tampered event
in that region counts as a consumer-visible tamper → attacker wins.

For Appendix-A-style all-or-nothing detection (does the final verify catch
it?), use ``verify_terminal`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from ...core.agent_id import SignedBoundID
from ...core.crypto import sha256_hex
from ...core.deployer import Deployer
from ...runtime.events import Event, EventStream
from .strategies import BindingStrategy, BoundSegment


# ---------- Issuance (deployer side) ----------

@dataclass(frozen=True)
class BoundStream:
    """Everything the agent side hands over: events + per-segment bound_ids."""

    events: tuple[Event, ...]
    segment_ids: tuple[SignedBoundID, ...]  # same length as segments yielded by strategy


def issue_bound_stream(
    *,
    deployer: Deployer,
    instance_id: str,
    system_id: str,
    strategy: BindingStrategy,
    stream: EventStream,
    certifications: tuple[str, ...] = (),
) -> BoundStream:
    """Run the strategy over `stream` and sign each segment.

    Segments are chained: segment N's bound_id carries
    prev_turn_hash = sha256(segment N-1's canonical payload).

    `certifications` is applied to every segment's BoundIDPayload.  For the
    end-to-end cert-aware demo (experiments/demo_e2e) the caller passes
    ``certifications=(CERT_NAME,)`` when the monitor cleared the session, and
    ``()`` when the monitor revoked it.  The default ``()`` keeps existing
    callers (E2 matrix) unaffected.
    """
    segments = list(strategy.segments(stream))
    signed_ids: list[SignedBoundID] = []
    prev_payload_hash: Optional[str] = None
    for seg in segments:
        sid = deployer.issue(
            instance_id=f"{instance_id}:seg{seg.segment_index}",
            system_id=system_id,
            output_bytes=seg.canonical_bytes(),
            binding_scope=seg.scope,  # type: ignore[arg-type]
            certifications=certifications,
            prev_turn_hash=prev_payload_hash,
        )
        signed_ids.append(sid)
        prev_payload_hash = sha256_hex(sid.payload.to_canonical_json())
    return BoundStream(events=tuple(stream.events), segment_ids=tuple(signed_ids))


# ---------- Verification (verifier side) ----------

@dataclass(frozen=True)
class VerifyStreamResult:
    """Outcome of verifying a (possibly tampered) BoundStream.

    Fields:
      attacker_wins: did a tampered event reach the consumer before detection?
      all_segments_ok: every segment passed (no tamper at all, i.e. benign).
      first_failing_index: which segment first failed (None if all passed).
      reason: human-readable reason for the first failure.
      tampered_consumer_visible: at least one tampered event was visible to the
        consumer before a covering verification rejected it.
    """

    attacker_wins: bool
    all_segments_ok: bool
    first_failing_index: Optional[int]
    reason: str
    tampered_consumer_visible: bool


def verify_bound_stream(
    *,
    strategy: BindingStrategy,
    bound: BoundStream,
    presented: EventStream,
) -> VerifyStreamResult:
    """Verify `presented` against the bound_ids issued over the original stream.

    Detection model:
      - Re-segment `presented` with `strategy`.
      - For each i, compare segment i's recomputed hash to bound.segment_ids[i].
      - A mismatch at index i means "tamper detected at this segment boundary."
      - The consumer has already seen events from segments 0..i-1 and the
        tampered segment's contents up to the boundary. If the strategy's
        segment i covers exactly the tampered content, detection is "in
        time." If the strategy's segments don't cover the tampered content
        until much later (e.g. G1 binds only at EOS), detection is "too
        late" → attacker wins.

    The simplification: we say the attacker wins iff the lengths differ
    (strategy produced different segment counts, meaning a segment was
    inserted/deleted) OR there is a mismatch that the strategy's boundary
    does not immediately contain the divergence point.

    For the matrix, the concrete rule we use:

      * If segment counts differ (insert/delete): attacker wins ONLY if the
        tampered region is not the terminal segment. E.g. G1 has only one
        terminal segment, so any divergence shows up at EOS and is "too
        late" — attacker wins on any non-empty tamper.
      * Otherwise, walk segments. For the first mismatch:
          - if that segment is consumer-boundary (per_tool_call, per_turn,
            per_n_tokens): defender wins — caught before next event.
          - if that segment is per_session only: attacker wins — the
            consumer already acted on the intermediate events.

    This matches the §4.3 intuition: "streaming-safe ✓" granularities (G3,
    G4) catch interleave at the boundary; G1 catches it only at EOS.
    """
    orig_segments = list(strategy.segments(EventStream(events=bound.events)))
    presented_segments = list(strategy.segments(presented))

    # Walk segments in parallel up to the shorter length, and at the FIRST
    # divergence ask: "what is the scope of the boundary where this
    # divergence is first checkable?" If that scope is per_session, the
    # verifier only notices at EOS → attacker wins. Otherwise the tamper
    # is caught at a consumer boundary → defender wins.
    min_len = min(len(presented_segments), len(bound.segment_ids))
    for i in range(min_len):
        pres = presented_segments[i]
        committed_hash = bound.segment_ids[i].payload.output_hash
        pres_hash = sha256_hex(pres.canonical_bytes())
        # Also detect scope divergence: if the attacker's edits changed the
        # strategy's perception of segment i's scope (e.g., an insertion
        # turned a per_turn segment into a per_n_tokens one), that's as
        # good as a hash mismatch.
        expected_scope = bound.segment_ids[i].payload.binding_scope
        if pres_hash != committed_hash or pres.scope != expected_scope:
            scope = pres.scope
            too_late = scope == "per_session"
            return VerifyStreamResult(
                attacker_wins=too_late,
                all_segments_ok=False,
                first_failing_index=i,
                reason=(
                    "output_hash_mismatch_at_eos_too_late"
                    if too_late
                    else "output_hash_mismatch_caught_at_boundary"
                ),
                tampered_consumer_visible=too_late,
            )

    # All common-prefix segments matched. Now check length parity.
    if len(presented_segments) == len(bound.segment_ids):
        return VerifyStreamResult(
            attacker_wins=False,
            all_segments_ok=True,
            first_failing_index=None,
            reason="ok",
            tampered_consumer_visible=False,
        )

    # Length divergence: either presented is shorter (rollback / drop) or
    # longer (insertion past the last bound_id boundary).
    if len(presented_segments) < len(bound.segment_ids):
        # The next expected bound_id covers a segment that never arrived.
        # Detection timing = scope of the FIRST MISSING original segment.
        missing_scope = orig_segments[min_len].scope
        too_late = missing_scope == "per_session"
        return VerifyStreamResult(
            attacker_wins=too_late,
            all_segments_ok=False,
            first_failing_index=min_len,
            reason=(
                "missing_segment_detected_too_late"
                if too_late
                else "missing_segment_caught_at_boundary"
            ),
            tampered_consumer_visible=too_late,
        )

    # presented has MORE segments than bound: attacker added content past
    # the last issued bound_id. Detection timing = scope of the first
    # unexpected segment.
    extra_scope = presented_segments[min_len].scope
    too_late = extra_scope == "per_session"
    return VerifyStreamResult(
        attacker_wins=too_late,
        all_segments_ok=False,
        first_failing_index=min_len,
        reason=(
            "extra_segment_detected_too_late"
            if too_late
            else "extra_segment_caught_at_boundary"
        ),
        tampered_consumer_visible=too_late,
    )
