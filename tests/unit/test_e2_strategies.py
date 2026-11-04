"""Binding strategies (G1-G4): segment boundaries are where we expect them."""

from __future__ import annotations

from src.defenses.granularity.strategies import (
    G1PerSession,
    G2PerTurn,
    G3PerToolCall,
    G4PerNTokens,
)
from src.runtime.events import Token, ToolCall, TurnEnd, TurnStart
from src.runtime.scripted_agent import build_scripted_stream


def test_g1_emits_single_segment():
    stream = build_scripted_stream()
    segs = list(G1PerSession().segments(stream))
    assert len(segs) == 1
    assert segs[0].scope == "per_session"
    # The whole stream is one segment.
    assert segs[0].events == stream.events


def test_g2_emits_one_segment_per_turn():
    stream = build_scripted_stream()
    segs = list(G2PerTurn().segments(stream))
    # Pre-session event + 2 turns + post-session event = up to 4 segments.
    # The scripted stream has SessionStart before any TurnStart and
    # SessionEnd after the last TurnEnd, so G2 yields: [session_pre,
    # turn0, turn1, session_post] = 4 segments.
    scopes = [s.scope for s in segs]
    assert scopes.count("per_turn") == 2
    # Each per_turn segment spans from TurnStart to TurnEnd inclusive.
    for seg in segs:
        if seg.scope != "per_turn":
            continue
        assert isinstance(seg.events[0], TurnStart)
        assert isinstance(seg.events[-1], TurnEnd)


def test_g3_each_tool_call_is_own_segment():
    stream = build_scripted_stream()
    segs = list(G3PerToolCall().segments(stream))
    tool_segs = [s for s in segs if len(s.events) == 1 and isinstance(s.events[0], ToolCall)]
    # The scripted stream has 3 tool calls total.
    assert len(tool_segs) == 3
    names = [s.events[0].name for s in tool_segs]
    assert names == ["web_search", "summarize", "draft_email"]


def test_g3_text_runs_between_tool_calls_are_bound():
    """G3's key property: every text run between tool calls is bound as its
    own per_tool_call segment. The trailing text before TurnEnd is bundled
    with TurnEnd into a per_turn segment (closing the turn cleanly), which
    still binds the trailing tokens — an attacker cannot append to them
    without breaking the per_turn segment's hash.

    Scripted stream layout per turn:
      [TurnStart, Token_pre, ToolCall, Token_mid, ToolCall, Token_post, TurnEnd]

    G3 segmentation per turn:
      - per_tool_call: (TurnStart, Token_pre)
      - per_tool_call: (ToolCall #1)
      - per_tool_call: (Token_mid)
      - per_tool_call: (ToolCall #2)     [turn 0 only; turn 1 has 1 tool call]
      - per_turn:      (Token_post, TurnEnd)

    Turn 0 contributes 2 text-only per_tool_call segments ((TurnStart+Token_pre)
    and (Token_mid)); turn 1 contributes 1 ((TurnStart+Token_pre)).
    Total text-only per_tool_call segments = 3.
    """
    stream = build_scripted_stream()
    segs = list(G3PerToolCall().segments(stream))
    text_segs = [
        s for s in segs
        if s.scope == "per_tool_call"
        and all(isinstance(e, (Token, TurnStart)) for e in s.events)
    ]
    assert len(text_segs) == 3

    # And every trailing Token *is* covered — by a per_turn segment that
    # includes the token run leading up to TurnEnd.
    per_turn_segs = [s for s in segs if s.scope == "per_turn"]
    assert len(per_turn_segs) == 2  # one per turn
    for seg in per_turn_segs:
        # Each per_turn closing segment starts with a Token (the trailing run)
        # and ends with a TurnEnd.
        assert isinstance(seg.events[0], Token)
        assert isinstance(seg.events[-1], TurnEnd)


def test_g4_closes_segments_at_tool_calls_and_char_budget():
    """G4 closes a segment either at a ToolCall, a TurnEnd, or once N
    characters have accumulated. With the scripted stream's short texts
    and N=64 default, mostly ToolCall/TurnEnd will fire first — but the
    count must be >= G3's count by construction."""
    stream = build_scripted_stream()
    g3_count = sum(1 for _ in G3PerToolCall().segments(stream))
    g4_count = sum(1 for _ in G4PerNTokens(n_tokens=64).segments(stream))
    assert g4_count >= g3_count


def test_g4_small_n_splits_more_aggressively():
    """With a very small N (e.g. 5 chars), G4 must split more finely than
    the default. This confirms the character-budget branch fires."""
    stream = build_scripted_stream()
    g4_default = sum(1 for _ in G4PerNTokens(n_tokens=64).segments(stream))
    g4_tiny = sum(1 for _ in G4PerNTokens(n_tokens=5).segments(stream))
    assert g4_tiny > g4_default
