"""Agent event stream model for E2 (output binding under streaming + tool calls).

Real MCP-based Claude runs emit token deltas, tool-call requests, tool-call
responses, and turn boundaries. For the E2 matrix, what matters is the
*shape* of that stream — not the specific content or LLM. So we model it as
a sequence of typed events that a scripted agent can produce deterministically
(for ASR measurement) and a real MCP wrapper can later produce for latency
numbers.

Event model:

  SessionStart
    TurnStart(turn_index=0)
      Token("Hello") Token(" world") ...
      ToolCall(call_id, name, input, output)
      Token(" the answer is 42")
    TurnEnd(turn_index=0)
    TurnStart(turn_index=1)
      ...
    TurnEnd(turn_index=1)
  SessionEnd

Key properties:
  - Deterministic byte serialisation: every event has a stable canonical
    encoding used for hashing. Two semantically equal streams must hash
    identically.
  - Boundary-consumable: the *consumer* sees events in order and may act
    on a ToolCall result, a Token batch, or an assistant turn the moment
    it is emitted. The binding verifier's job is to reject BEFORE the
    consumer acts, not just at SessionEnd.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from typing import Any


# ---------- Event types ----------

class Event(abc.ABC):
    """Base event. Subclasses must be frozen dataclasses with a `kind` literal."""

    kind: str = ""

    @abc.abstractmethod
    def to_canonical_bytes(self) -> bytes:
        """Deterministic byte encoding used for hashing."""
        ...


def _canon(obj: Any) -> bytes:
    """Shared canonical JSON encoding: sorted keys, no whitespace, ascii."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True)
class SessionStart(Event):
    session_id: str
    kind: str = "session_start"

    def to_canonical_bytes(self) -> bytes:
        return _canon({"kind": self.kind, "session_id": self.session_id})


@dataclass(frozen=True)
class SessionEnd(Event):
    kind: str = "session_end"

    def to_canonical_bytes(self) -> bytes:
        return _canon({"kind": self.kind})


@dataclass(frozen=True)
class TurnStart(Event):
    turn_index: int
    kind: str = "turn_start"

    def to_canonical_bytes(self) -> bytes:
        return _canon({"kind": self.kind, "turn_index": self.turn_index})


@dataclass(frozen=True)
class TurnEnd(Event):
    turn_index: int
    kind: str = "turn_end"

    def to_canonical_bytes(self) -> bytes:
        return _canon({"kind": self.kind, "turn_index": self.turn_index})


@dataclass(frozen=True)
class Token(Event):
    """Streaming token or token batch. `text` is the delta since the last Token."""
    text: str
    kind: str = "token"

    def to_canonical_bytes(self) -> bytes:
        return _canon({"kind": self.kind, "text": self.text})


@dataclass(frozen=True)
class ToolCall(Event):
    """A complete tool invocation — request + response as one atomic event.

    Real MCP emits these as two events (request then response); conflating
    them here keeps the binding story simple and mirrors the paper's
    ``(tool_name, tool_input_hash, tool_output_hash)`` tuple in §4.3.
    """
    call_id: str
    name: str
    input_json: str     # canonical JSON of the tool input
    output_json: str    # canonical JSON of the tool output
    kind: str = "tool_call"

    def to_canonical_bytes(self) -> bytes:
        return _canon({
            "kind": self.kind,
            "call_id": self.call_id,
            "name": self.name,
            "input_json": self.input_json,
            "output_json": self.output_json,
        })


# ---------- Stream container ----------

@dataclass(frozen=True)
class EventStream:
    """Immutable sequence of events. The thing an agent emits and an attacker
    (maybe) mutates before it reaches the verifier."""

    events: tuple[Event, ...]

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def with_mutation(self, new_events: tuple[Event, ...]) -> "EventStream":
        return EventStream(events=new_events)
