"""Monitor interface + verdict type."""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorVerdict:
    """Result of a monitor evaluating one agent output.

    Fields:
      compromised: True iff the monitor believes the output was produced
        by a prompt-injected / behaviorally-deviant agent.
      reason: short human-readable reason string.
      latency_ms: wall-clock time the monitor took to reach the verdict.
    """

    compromised: bool
    reason: str
    latency_ms: float


class Monitor(abc.ABC):
    name: str = ""

    @abc.abstractmethod
    def evaluate(self, *, system_intent: str, output: str) -> MonitorVerdict:
        """Return a verdict for `output` given the declared `system_intent`.

        `system_intent` is a short description of what the agent was
        authorised to do this turn (e.g. "summarise the Q2 report"). The
        monitor's job is to decide whether `output` looks like a faithful
        execution of that intent or a hijacked deviation.
        """
        ...
