"""Behavioral monitors M1/M2/M3 for E3.

A monitor observes an agent's output for a turn and decides whether the
"no_prompt_injection" certification should remain valid for the bound_id
issued over that output.

Three designs compared in the E3 matrix:

  M1 NoneMonitor         - always valid; baseline
  M2 RuleBasedMonitor    - pattern-match known injection markers
  M3 LLMJudgeMonitor     - Claude reads the output and the system-prompt
                           intent and returns a compromised/ok verdict

All monitors return MonitorVerdict(compromised, reason, latency_ms).
"""

from .base import Monitor, MonitorVerdict
from .m1_none import NoneMonitor
from .m2_rules import RuleBasedMonitor
from .m3_judge import ClaudeLLMJudge, LLMJudge, LLMJudgeMonitor, MockLLMJudge

__all__ = [
    "Monitor",
    "MonitorVerdict",
    "NoneMonitor",
    "RuleBasedMonitor",
    "LLMJudgeMonitor",
    "MockLLMJudge",
    "ClaudeLLMJudge",
    "LLMJudge",
]
