"""Classical (Chan §3.3) attacks: Tampering, ID Spoofing, Instance Spoofing.

These are the three attacks in the Appendix A sanity-check matrix. They are
NOT the contribution of the paper — they replicate known PKI results. Their
purpose is to verify the test harness itself, and to anchor reviewers.
"""

from .attacks import (
    Attack,
    AttackOutcome,
    AttackResult,
    IDSpoofing,
    InstanceSpoofing,
    Replay,
    Tampering,
)

__all__ = [
    "Attack",
    "AttackOutcome",
    "AttackResult",
    "IDSpoofing",
    "InstanceSpoofing",
    "Replay",
    "Tampering",
]
