"""Binding strategies G1-G4: when does the deployer emit a SignedBoundID?

Each strategy walks the event stream and yields ``BoundSegment``s. A segment
is a contiguous slice of events whose canonical byte concatenation is what
the bound_id's ``output_hash`` covers. Successive segments are chained by
setting ``prev_turn_hash`` to the SHA-256 of the previous segment's canonical
payload (re-purposing that field as a generic prev-segment chain — the name
is the schema's; the semantics are "the thing this segment follows").

Granularities:
  G1 per-session: one segment covering every event.
  G2 per-turn:    one segment per (TurnStart..TurnEnd) pair; non-turn events
                  (SessionStart/End) are in their own pre/post segments.
  G3 per-tool-call: within a turn, a new segment starts at each ToolCall and
                   another starts after the ToolCall; turn boundaries still
                   delimit segments.
  G4 per-64-tokens: within a turn, segments also close every N text
                    characters of emitted tokens (approximating tokens).

Why this design:
  - A consumer sees events in order. They can only trust an event once its
    covering bound_id has been verified. So the boundary matters: a
    stream-interleave attack that appends tokens AFTER the last bound_id
    boundary is invisible to that strategy.
  - This directly operationalises the §4.3 hypothesis that G3 is the
    minimum viable granularity for streaming+tool-calling agents.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable, Iterator

from ...runtime.events import (
    Event,
    SessionEnd,
    SessionStart,
    Token,
    ToolCall,
    TurnEnd,
    TurnStart,
)


@dataclass(frozen=True)
class BoundSegment:
    """A contiguous run of events covered by one bound_id.

    Fields:
      segment_index: position in the stream's segment sequence (0-indexed).
      events: the events in this segment (in order).
      scope: which `BindingScope` this segment represents (for the bound_id).
      turn_index: the turn this segment belongs to (-1 for inter-turn).
    """

    segment_index: int
    events: tuple[Event, ...]
    scope: str           # matches BindingScope literal values
    turn_index: int = -1

    def canonical_bytes(self) -> bytes:
        """Concatenation of each event's canonical bytes with a NUL separator.

        NUL separator prevents length-extension ambiguity: ``[A, B]`` and
        ``[AB]`` (if such a fusion were possible) would produce different
        bytes. The events themselves already encode unambiguously via JSON.
        """
        return b"\x00".join(e.to_canonical_bytes() for e in self.events)


class BindingStrategy(abc.ABC):
    """Produces a sequence of BoundSegments from an event stream."""

    scope_name: str = ""

    @abc.abstractmethod
    def segments(self, events: Iterable[Event]) -> Iterator[BoundSegment]:
        ...


# ---------- G1: per-session ----------

class G1PerSession(BindingStrategy):
    scope_name = "per_session"

    def segments(self, events: Iterable[Event]) -> Iterator[BoundSegment]:
        evs = tuple(events)
        yield BoundSegment(
            segment_index=0,
            events=evs,
            scope=self.scope_name,
            turn_index=-1,
        )


# ---------- G2: per-turn ----------

class G2PerTurn(BindingStrategy):
    scope_name = "per_turn"

    def segments(self, events: Iterable[Event]) -> Iterator[BoundSegment]:
        evs = tuple(events)
        idx = 0
        i = 0
        pre: list[Event] = []
        # Session-level events that fall outside any turn go into their own
        # segments so we don't silently drop them from the binding set.
        while i < len(evs):
            ev = evs[i]
            if isinstance(ev, TurnStart):
                if pre:
                    yield BoundSegment(
                        segment_index=idx, events=tuple(pre),
                        scope="per_session", turn_index=-1,
                    )
                    idx += 1
                    pre = []
                # Collect the turn.
                turn_events: list[Event] = [ev]
                i += 1
                while i < len(evs) and not isinstance(evs[i], TurnEnd):
                    turn_events.append(evs[i])
                    i += 1
                if i < len(evs) and isinstance(evs[i], TurnEnd):
                    turn_events.append(evs[i])
                    i += 1
                yield BoundSegment(
                    segment_index=idx,
                    events=tuple(turn_events),
                    scope=self.scope_name,
                    turn_index=(
                        turn_events[0].turn_index if isinstance(turn_events[0], TurnStart) else -1
                    ),
                )
                idx += 1
            else:
                pre.append(ev)
                i += 1
        if pre:
            yield BoundSegment(
                segment_index=idx, events=tuple(pre),
                scope="per_session", turn_index=-1,
            )


# ---------- G3: per-tool-call (within turns) ----------

class G3PerToolCall(BindingStrategy):
    scope_name = "per_tool_call"

    def segments(self, events: Iterable[Event]) -> Iterator[BoundSegment]:
        evs = tuple(events)
        idx = 0
        buf: list[Event] = []
        current_turn = -1
        for ev in evs:
            if isinstance(ev, TurnStart):
                # flush pre-turn buffer
                if buf:
                    yield BoundSegment(
                        segment_index=idx, events=tuple(buf),
                        scope="per_session", turn_index=-1,
                    )
                    idx += 1
                    buf = []
                current_turn = ev.turn_index
                buf.append(ev)
                continue
            if isinstance(ev, TurnEnd):
                buf.append(ev)
                yield BoundSegment(
                    segment_index=idx, events=tuple(buf),
                    scope="per_turn", turn_index=current_turn,
                )
                idx += 1
                buf = []
                current_turn = -1
                continue
            if isinstance(ev, ToolCall):
                # Close current text-run as its own segment, then bind the
                # tool-call as its own segment. This is the key G3 move.
                if buf and not (len(buf) == 1 and isinstance(buf[0], TurnStart)):
                    yield BoundSegment(
                        segment_index=idx, events=tuple(buf),
                        scope=self.scope_name, turn_index=current_turn,
                    )
                    idx += 1
                    buf = []
                yield BoundSegment(
                    segment_index=idx, events=(ev,),
                    scope=self.scope_name, turn_index=current_turn,
                )
                idx += 1
                continue
            buf.append(ev)
        if buf:
            yield BoundSegment(
                segment_index=idx, events=tuple(buf),
                scope="per_session", turn_index=-1,
            )


# ---------- G4: per-N-tokens ----------

class G4PerNTokens(BindingStrategy):
    """Every N characters of accumulated token text, close a segment.

    Uses character count as a proxy for token count — the goal is to show
    finer-than-G3 granularity within a turn, and the exact N doesn't affect
    the attack-detection matrix shape. Default N=64 per the foundation.
    """

    scope_name = "per_n_tokens"

    def __init__(self, n_tokens: int = 64) -> None:
        self.n_tokens = n_tokens

    def segments(self, events: Iterable[Event]) -> Iterator[BoundSegment]:
        evs = tuple(events)
        idx = 0
        buf: list[Event] = []
        chars_in_buf = 0
        current_turn = -1
        for ev in evs:
            if isinstance(ev, TurnStart):
                if buf:
                    yield BoundSegment(
                        segment_index=idx, events=tuple(buf),
                        scope="per_session", turn_index=-1,
                    )
                    idx += 1
                    buf = []
                    chars_in_buf = 0
                current_turn = ev.turn_index
                buf.append(ev)
                continue
            if isinstance(ev, TurnEnd):
                buf.append(ev)
                yield BoundSegment(
                    segment_index=idx, events=tuple(buf),
                    scope="per_turn", turn_index=current_turn,
                )
                idx += 1
                buf = []
                chars_in_buf = 0
                current_turn = -1
                continue
            if isinstance(ev, ToolCall):
                if buf and not (len(buf) == 1 and isinstance(buf[0], TurnStart)):
                    yield BoundSegment(
                        segment_index=idx, events=tuple(buf),
                        scope=self.scope_name, turn_index=current_turn,
                    )
                    idx += 1
                    buf = []
                    chars_in_buf = 0
                yield BoundSegment(
                    segment_index=idx, events=(ev,),
                    scope=self.scope_name, turn_index=current_turn,
                )
                idx += 1
                continue
            if isinstance(ev, Token):
                buf.append(ev)
                chars_in_buf += len(ev.text)
                if chars_in_buf >= self.n_tokens:
                    yield BoundSegment(
                        segment_index=idx, events=tuple(buf),
                        scope=self.scope_name, turn_index=current_turn,
                    )
                    idx += 1
                    buf = []
                    chars_in_buf = 0
                continue
            buf.append(ev)
        if buf:
            yield BoundSegment(
                segment_index=idx, events=tuple(buf),
                scope="per_session", turn_index=-1,
            )
