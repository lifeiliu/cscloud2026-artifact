"""Operational attacks produce the tampered streams we expect."""

from __future__ import annotations

from src.attacks.operational import (
    StreamInterleave,
    ToolCallSubstitution,
    TurnRollback,
)
from src.defenses.granularity import G3PerToolCall
from src.runtime.events import Token, ToolCall, TurnEnd, TurnStart
from src.runtime.scripted_agent import build_scripted_stream


def test_stream_interleave_inserts_malicious_token():
    original = build_scripted_stream()
    strategy = G3PerToolCall()
    tampered = StreamInterleave().execute(original, strategy)
    # Exactly one more event than the original.
    assert len(tampered.events) == len(original.events) + 1
    # The inserted event is a Token whose text matches the malicious payload.
    malicious = [e for e in tampered.events if isinstance(e, Token)
                 and e.text == StreamInterleave.MALICIOUS_TEXT]
    assert len(malicious) == 1


def test_stream_interleave_insertion_is_before_turn0_end():
    original = build_scripted_stream()
    tampered = StreamInterleave().execute(original, G3PerToolCall())
    # Find the position of the malicious token and of turn 0 end.
    malicious_pos = next(
        i for i, e in enumerate(tampered.events)
        if isinstance(e, Token) and e.text == StreamInterleave.MALICIOUS_TEXT
    )
    turn0_end_pos = next(
        i for i, e in enumerate(tampered.events)
        if isinstance(e, TurnEnd) and e.turn_index == 0
    )
    assert malicious_pos < turn0_end_pos


def test_tool_call_substitution_forges_first_tool_output():
    original = build_scripted_stream()
    tampered = ToolCallSubstitution().execute(original, G3PerToolCall())
    assert len(tampered.events) == len(original.events)
    orig_tools = [e for e in original.events if isinstance(e, ToolCall)]
    tampered_tools = [e for e in tampered.events if isinstance(e, ToolCall)]
    assert tampered_tools[0].output_json == ToolCallSubstitution.FORGED_OUTPUT
    assert tampered_tools[0].output_json != orig_tools[0].output_json
    # Later tool calls untouched.
    for a, b in zip(orig_tools[1:], tampered_tools[1:]):
        assert a.output_json == b.output_json


def test_turn_rollback_replays_turn0_body_inside_turn1_envelope():
    """TurnRollback keeps the two-turn structural shape but swaps turn 1's
    body for a copy of turn 0's body. The turn indices of TurnStart/TurnEnd
    are preserved so the stream still parses as two turns; only the
    content between them has been rolled back."""
    from src.runtime.events import Token

    original = build_scripted_stream()
    tampered = TurnRollback().execute(original, G3PerToolCall())

    # Both TurnStart/TurnEnd markers are present for both turns.
    assert any(isinstance(e, TurnStart) and e.turn_index == 0 for e in tampered.events)
    assert any(isinstance(e, TurnEnd) and e.turn_index == 0 for e in tampered.events)
    assert any(isinstance(e, TurnStart) and e.turn_index == 1 for e in tampered.events)
    assert any(isinstance(e, TurnEnd) and e.turn_index == 1 for e in tampered.events)

    # Collect turn-1 body (events between TurnStart(1) and TurnEnd(1)).
    t1_start = next(i for i, e in enumerate(tampered.events)
                    if isinstance(e, TurnStart) and e.turn_index == 1)
    t1_end = next(i for i, e in enumerate(tampered.events)
                  if isinstance(e, TurnEnd) and e.turn_index == 1)
    t1_body = tampered.events[t1_start + 1:t1_end]

    # Collect turn 0 body from the ORIGINAL stream.
    o_t0_start = next(i for i, e in enumerate(original.events)
                      if isinstance(e, TurnStart) and e.turn_index == 0)
    o_t0_end = next(i for i, e in enumerate(original.events)
                    if isinstance(e, TurnEnd) and e.turn_index == 0)
    o_t0_body = original.events[o_t0_start + 1:o_t0_end]

    # Rolled-back turn 1 body == original turn 0 body.
    assert t1_body == o_t0_body

    # The original turn-1 tool call (draft_email) no longer appears.
    assert not any(
        isinstance(e, ToolCall) and e.name == "draft_email"
        for e in tampered.events
    )
