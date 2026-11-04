"""M1: no monitor. Baseline — cert stays valid for its TTL regardless."""

from __future__ import annotations

from .base import Monitor, MonitorVerdict


class NoneMonitor(Monitor):
    """Always returns `compromised=False`.

    Purpose: shows the decoupling problem without any mitigation. An IPI
    payload that hijacks the agent will produce a session where the
    output is bad but the `no_prompt_injection` cert remains valid until
    TTL expires — the Decoupling Problem §3.2.
    """

    name = "M1_none"

    def evaluate(self, *, system_intent: str, output: str) -> MonitorVerdict:
        return MonitorVerdict(compromised=False, reason="no_monitor", latency_ms=0.0)
