"""Statistical utilities for Paper 2 — Wilson CI, two-proportion z-test.

Reuses the pattern from Paper 1's E2/E3 experiment harnesses but
centralises it for cross-experiment use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from scipy.stats import binomtest


class WilsonCI(NamedTuple):
    """Wilson 95% confidence interval on a proportion."""
    low: float
    high: float
    point: float   # sample proportion
    n: int
    k: int         # number of successes


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> WilsonCI:
    """Compute the Wilson score interval for a binomial proportion.

    Args:
        k: number of successes (e.g., attack successes).
        n: number of trials.
        confidence: confidence level (default 0.95).

    Returns:
        WilsonCI with low, high, point estimate, n, k.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not (0 <= k <= n):
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")

    result = binomtest(k, n)
    ci = result.proportion_ci(confidence_level=confidence, method="wilson")
    return WilsonCI(
        low=ci.low,
        high=ci.high,
        point=k / n,
        n=n,
        k=k,
    )


@dataclass(frozen=True)
class CellResult:
    """Result for one (design × attack) cell in the ASR matrix."""

    design: str
    attack: str
    n_trials: int
    n_successes: int
    asr: float
    ci_low: float
    ci_high: float

    @property
    def ci_str(self) -> str:
        return f"{self.asr:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}]"
