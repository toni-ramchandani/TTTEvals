"""Day 4, Module 11: how much a number can tell you.

Five small tools, each a few lines of standard-library Python, each the one a
cited source recommends for the case in hand.

wilson          an interval for one pass rate. Bowyer, Aitchison and Ivanova
                (ICML 2025) show that the textbook interval (mean plus or
                minus 1.96 standard errors, from the central limit theorem)
                is too narrow below a few hundred questions, and recommend
                the Wilson score interval instead.
mcnemar_exact   is the candidate better than the shipped build on the same
                questions? Only the questions whose verdict changed carry
                information (McNemar 1947). The exact version is a sign test
                on those questions, right for small counts.
prob_better     the same evidence read as a probability: with a uniform
                prior, the chance that a changed question is more likely to
                be a fix than a break. The paired Bayesian reading Bowyer et
                al. recommend.
min_discordant  how many changed questions a test needs before any result
                can reach significance. Below it, the run cannot decide.
percentile      p50 and p95 of latency, nearest-rank. Averages hide the slow
                tail users feel (Google SRE, service level objectives).

Miller (2024), "Adding Error Bars to Evals", makes the general case: report
an interval with every score, and compare two systems on the same questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95 percent Wilson score interval for k passes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided p for k successes in n fair coin flips (the sign test)."""
    if n == 0:
        return 1.0
    low = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(low + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_exact(fixed: int, broke: int) -> float:
    """Exact McNemar test: are fixes and breaks a fair split of the changed questions?"""
    return binom_two_sided(fixed, fixed + broke)


def prob_better(fixed: int, broke: int) -> float:
    """P(a changed question is more often a fix than a break), uniform prior.

    The posterior for that share is Beta(1 + fixed, 1 + broke). For whole
    numbers, P(Beta(a, b) <= 1/2) equals P(Binomial(a + b - 1, 1/2) >= a),
    so the probability needs only a binomial sum.
    """
    a, b = 1 + fixed, 1 + broke
    n = a + b - 1
    at_most_half = sum(math.comb(n, i) for i in range(a, n + 1)) / 2 ** n
    return 1 - at_most_half


def min_discordant(alpha: float = 0.05) -> int:
    """The fewest changed questions for which even a clean sweep is significant."""
    n = 1
    while mcnemar_exact(n, 0) >= alpha:
        n += 1
    return n


def smallest_win(changed: int, alpha: float = 0.05) -> int | None:
    """Of `changed` questions, how many must be fixes for significance. None if none suffices."""
    for k in range(changed // 2 + 1, changed + 1):
        if mcnemar_exact(k, changed - k) < alpha:
            return k
    return None


def reviews_for(half_width: float, rate: float = 0.5, z: float = Z95) -> int:
    """About how many reviewed answers put a failure rate within +/- half_width.

    The normal approximation, which is fine for planning a review budget. The
    count depends on how many answers are read, not on what share of traffic
    that is: 5 percent of a quiet week and 5 percent of a busy one are
    different evidence.
    """
    return math.ceil(z * z * rate * (1 - rate) / (half_width * half_width))


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least q percent at or below it."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    rank = max(1, math.ceil(q / 100 * len(vals)))
    return vals[rank - 1]


@dataclass
class Paired:
    """The 2 x 2 table of a paired comparison, with the ids in each cell."""
    both_pass: list[str] = field(default_factory=list)
    both_fail: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)        # failed before, passes now
    broke: list[str] = field(default_factory=list)        # passed before, fails now

    @property
    def n(self) -> int:
        return len(self.both_pass) + len(self.both_fail) + len(self.fixed) + len(self.broke)

    @property
    def p(self) -> float:
        return mcnemar_exact(len(self.fixed), len(self.broke))

    @property
    def prob_better(self) -> float:
        return prob_better(len(self.fixed), len(self.broke))


def paired(before: dict[str, bool], after: dict[str, bool]) -> Paired:
    """Pair two runs by question id. Both runs must hold the same questions."""
    if set(before) != set(after):
        missing = sorted(set(before) ^ set(after))
        raise ValueError(f"the two runs hold different questions: {', '.join(missing[:5])}")
    t = Paired()
    for k in before:
        cell = (t.both_pass if before[k] and after[k] else t.both_fail if not before[k] and not after[k]
                else t.fixed if after[k] else t.broke)
        cell.append(k)
    return t


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def prob_text(x: float) -> str:
    """A probability for print: never rounds a near-certainty up to 100 percent."""
    return ">99.9%" if x > 0.999 else "<0.1%" if x < 0.001 else pct(x)


def interval(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {pct(k / n if n else None)} [{pct(lo)}, {pct(hi)}]"


def p_text(p: float) -> str:
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.1e}"
