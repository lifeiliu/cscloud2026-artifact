"""Binding granularity strategies G1-G4 for E2."""

from .strategies import (
    BindingStrategy,
    BoundSegment,
    G1PerSession,
    G2PerTurn,
    G3PerToolCall,
    G4PerNTokens,
)
from .chain import verify_bound_stream, VerifyStreamResult

__all__ = [
    "BindingStrategy",
    "BoundSegment",
    "G1PerSession",
    "G2PerTurn",
    "G3PerToolCall",
    "G4PerNTokens",
    "verify_bound_stream",
    "VerifyStreamResult",
]
