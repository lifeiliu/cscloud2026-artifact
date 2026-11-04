"""Deterministic scripted agent for E2 ASR measurement.

The goal of E2's 12-cell matrix is to measure whether each binding
granularity detects each attack. That's a property of the mechanism, not
of the LLM. A scripted runtime makes the matrix deterministic and keeps
each trial in the microsecond regime.

Real Claude+MCP integration will slot in later for latency numbers (E2's
secondary metric) by producing the same EventStream type.

The scripted agent emits the canonical 3-tool task from the foundation:
  Turn 0: user asks to "research X, summarize, draft an email"
    - ToolCall: web_search
    - ToolCall: summarize
    - Token stream: "Here is your draft..."
  Turn 1: user asks a follow-up
    - ToolCall: draft_email
    - Token stream: "Done."

This keeps all three attack vectors (stream, tool-output, turn-rollback)
present in every trial.
"""

from __future__ import annotations

from .events import (
    EventStream,
    SessionEnd,
    SessionStart,
    Token,
    ToolCall,
    TurnEnd,
    TurnStart,
)


def build_scripted_stream(session_id: str = "sess_e2_001") -> EventStream:
    """Canonical 2-turn, 3-tool scripted stream used by the E2 matrix."""
    events = [
        SessionStart(session_id=session_id),

        # ---- Turn 0: research + summarize ----
        TurnStart(turn_index=0),
        Token(text="I'll look that up."),
        ToolCall(
            call_id="c0",
            name="web_search",
            input_json='{"q":"agent identity binding"}',
            output_json='{"results":["paper1","paper2","paper3"]}',
        ),
        Token(text=" Here's what I found."),
        ToolCall(
            call_id="c1",
            name="summarize",
            input_json='{"docs":["paper1","paper2","paper3"]}',
            output_json='{"summary":"three papers on agent identity"}',
        ),
        Token(text=" Summary complete."),
        TurnEnd(turn_index=0),

        # ---- Turn 1: draft email ----
        TurnStart(turn_index=1),
        Token(text="Drafting now."),
        ToolCall(
            call_id="c2",
            name="draft_email",
            input_json='{"to":"alice@example.com","summary":"..."}',
            output_json='{"status":"ok","email_id":"e42"}',
        ),
        Token(text=" Draft saved."),
        TurnEnd(turn_index=1),

        SessionEnd(),
    ]
    return EventStream(events=tuple(events))
