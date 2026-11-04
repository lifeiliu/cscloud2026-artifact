"""Scripted injection-aware agent for E3 deterministic matrix.

Given an InjectionCase and a boolean saying whether the injection
succeeds in this trial, emits the case's compromised or benign output.
Real Claude + injection-via-tool-result will slot in later; the
deterministic flow lets the matrix answer:

  "Given that an injection succeeded, how often does the monitor
   catch it before the cert ships, and how often does it cost us a
   false revocation on benign turns?"

This is the right framing because injection *success rate* is a model-
alignment property, not a defense property. The cert's decoupling
behavior is the defense property we are measuring.
"""

from __future__ import annotations

from dataclasses import dataclass

from .injection import InjectionCase


@dataclass(frozen=True)
class AgentTurnOutput:
    """What the runtime observes after one agent turn."""

    output: str
    was_injected: bool  # ground truth — the runtime itself records this


class ScriptedInjectionAgent:
    """Emits benign or compromised output per-case, deterministically.

    Policy:
      injection_succeeded=True  -> emit case.expected_compromised_output
      injection_succeeded=False -> emit case.benign_output
    """

    def run_turn(self, *, case: InjectionCase, injection_succeeded: bool) -> AgentTurnOutput:
        if injection_succeeded:
            return AgentTurnOutput(output=case.expected_compromised_output, was_injected=True)
        return AgentTurnOutput(output=case.benign_output, was_injected=False)
