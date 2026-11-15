"""Day 4, Module 10: evaluation after release, on a simulated month of traffic.

There is no real traffic in a workshop, so this builds some, and says so
everywhere it prints. The requests are Day 2's tuples (topic x asker x kind
of question) in the reference wording, drawn with fixed weights and a fixed
seed. Every number below follows from three things written in this file:
the traffic mix, the feedback model, and the review budget. Change them and
see which conclusions survive; the ones that survive are the lessons.

What the simulation knows about each request, and what production knows:

    known to the simulation    whether the request should be declined (from
                               its tuple), and every code check
    known to production        the code checks, which need no reference
                               answer (access, current policy, groundedness),
                               and a thumbs-up or thumbs-down from a few users
    known only after review    whether a decline or an answer was right

Content errors (a wrong figure) are invisible here, because a tuple carries
no expected figure. The regression suite covers those.

The four weeks. The shipped build serves weeks 1 to 3; the mix is the same,
so any movement between them is chance. In week 4 two things change at once:
repaired + current-only is released, and somebody edits the current-policy
instrument (`stale_by_date`, below). The week 4 dashboard mixes both.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field, replace

from evals.day2_synthetic import ASKERS, Tuple_, render_offline
from evals.day3_generation import citation_support
from evals.day4_stats import wilson
from evals.day4_suite import GROUNDED_FLOOR, Build, guard_rule, guarded_trace, make_index
from helix import answer
from helix.ingest import document_status
from helix.retrieve import ENTITLEMENTS
from tracing.schema import Trace

#: The traffic mix: share of requests by topic and by kind of question.
#: Askers are equally likely. An assumption, stated, not a measurement.
TOPIC_MIX = {"parental leave": 0.25, "annual leave": 0.25, "notice period": 0.10,
             "home office expenses": 0.15, "remote work": 0.15, "salary bands": 0.10}
KIND_MIX = {"clear question": 0.55, "ambiguous request": 0.25, "multi-part": 0.10, "out of scope": 0.10}
REQUESTS_PER_WEEK = 1000
SEED = 2026

#: The feedback model. Most users never click; those who do react to what
#: they can see. Nobody can see that a page was superseded, or that a page
#: they read was restricted, so those failures rarely get a thumbs-down.
CLICK_RATE = 0.10
THUMBS_DOWN = {          # P(thumbs-down | the user clicked), by what happened
    "false refusal": 0.60,   # asked a fair question, got nothing
    "ungrounded": 0.30,
    "missed refusal": 0.15,  # asked Helix to act or advise, got policy text
    "stale": 0.10,
    "leak": 0.03,            # shown salary bands: rarely a complaint
    "ok": 0.08,
}
REVIEW_BUDGET = 50       # answers one person can read properly in a week


def _check_names() -> tuple[str, ...]:
    return ("leak", "stale", "ungrounded")


@dataclass
class Request:
    week: int
    n: int
    tuple_: Tuple_
    text: str
    trace: Trace
    flags: set[str] = field(default_factory=set)      # code checks, on every request
    truth: str = "ok"                                 # what review would find; worst first
    clicked: bool = False
    thumbs_down: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.text, self.tuple_.asker.office, self.tuple_.asker.role)


def draw_week(week: int, n: int = REQUESTS_PER_WEEK, seed: int = SEED) -> list[Tuple_]:
    """n requests drawn from the stated mix, seeded per week."""
    rng = random.Random(f"{seed}:{week}")
    topics, kinds = list(TOPIC_MIX), list(KIND_MIX)
    out = []
    for _ in range(n):
        t = rng.choices(topics, weights=[TOPIC_MIX[x] for x in topics])[0]
        k = rng.choices(kinds, weights=[KIND_MIX[x] for x in kinds])[0]
        a = rng.choice(ASKERS)
        out.append(Tuple_(t, a, k))
    return out


# ---------------------------------------------------------------------------
# Code checks that need no reference answer: they run on every request.
# ---------------------------------------------------------------------------
def leak(trace: Trace) -> bool:
    allowed = ENTITLEMENTS.get(trace.user_role, {"all"})
    return any(c.access_label not in allowed for c in trace.retrieved if not c.dropped_at)


def _cited_docs(trace: Trace) -> list:
    blob = " ".join(trace.citations) + " " + (trace.answer or "")
    return [c for c in trace.retrieved if not c.dropped_at and c.chunk_id in blob]


def stale(trace: Trace, status: dict | None = None) -> bool:
    """Cites a page a newer page supersedes: the charter's definition."""
    status = status or document_status()
    return any(status.get(c.doc_id, {}).get("status") == "superseded" for c in _cited_docs(trace))


def stale_by_date(trace: Trace, before: str = "2025-01-01") -> bool:
    """The planted instrument edit: 'a page not reviewed since 2025 counts as stale'.

    It sounds stricter. It is a different definition: the current 2024
    parental leave page now counts as stale, and every correct parental
    leave answer with it. Same name on the dashboard, new yardstick.
    """
    return any(c.last_modified < before for c in _cited_docs(trace))


def ungrounded(trace: Trace) -> bool:
    if trace.refused:
        return False
    rows = citation_support(trace)
    return not rows or sum(r["supported"] for r in rows) / len(rows) < GROUNDED_FLOOR


def flags_of(trace: Trace, stale_check=stale) -> set[str]:
    out = set()
    if leak(trace):
        out.add("leak")
    if stale_check(trace):
        out.add("stale")
    if ungrounded(trace):
        out.add("ungrounded")
    return out


def truth_of(r: Request) -> str:
    """The worst thing that happened, as a reviewer would name it."""
    should = r.tuple_.should_refuse
    if r.flags & {"leak"}:
        return "leak"
    if should and not r.trace.refused:
        return "missed refusal"
    if not should and r.trace.refused:
        return "false refusal"
    if "stale" in r.flags:
        return "stale"
    if "ungrounded" in r.flags:
        return "ungrounded"
    return "ok"


# ---------------------------------------------------------------------------
# Serving the traffic
# ---------------------------------------------------------------------------
class Server:
    """One build, answering requests. Each distinct request is answered once.

    A repeated question from the same kind of asker gets the same answer
    here. A live model at temperature 0 mostly repeats itself too; how often
    it does not is Module 11's `--repeat`.
    """

    def __init__(self, build: Build, on_new=None) -> None:
        self.build = build
        self.index = make_index(build)
        self.cache: dict[tuple, Trace] = {}
        self.on_new = on_new

    def __call__(self, t: Tuple_, text: str) -> Trace:
        key = (text, t.asker.office, t.asker.role)
        if key not in self.cache:
            cfg = self.build.cfg
            rule = guard_rule(self.build.guard, text) if self.build.guard else ""
            if rule:
                trace = guarded_trace(text, rule, self.build.guard, cfg, t.asker.office, t.asker.role)
            else:
                trace = answer(text, self.index, cfg, user_office=t.asker.office, user_role=t.asker.role)
            self.cache[key] = trace
            if self.on_new:
                self.on_new(len(self.cache), text)
        return self.cache[key]


def serve_week(week: int, server: Server, stale_check=stale, seed: int = SEED) -> list[Request]:
    rng = random.Random(f"{seed}:feedback:{week}")
    out = []
    for i, t in enumerate(draw_week(week, seed=seed)):
        text = render_offline(t)
        trace = server(t, text)
        r = Request(week, i, t, text, trace, flags_of(trace, stale_check))
        r.truth = truth_of(replace(r, flags=flags_of(trace)))       # truth uses the charter's definitions
        r.clicked = rng.random() < CLICK_RATE
        r.thumbs_down = r.clicked and rng.random() < THUMBS_DOWN[r.truth]
        out.append(r)
    return out


def rate_line(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k:>4}/{n}  {100 * k / n:5.1f}%  [{100 * lo:4.1f}, {100 * hi:4.1f}]"


def online_rates(requests: list[Request]) -> dict[str, int]:
    c = Counter()
    for r in requests:
        c.update(r.flags)
    return {k: c[k] for k in _check_names()}


# ---------------------------------------------------------------------------
# Review strategies at a fixed budget
#
# The failures only a reviewer can see here are refusal errors: a request
# Helix should have declined and answered, or the reverse. Three ways to spend
# the same 50 reviews, each drawn many times so the spread is visible:
#
#   random        50 requests at random. Finds what is common; its rate is
#                 unbiased.
#   thumbs-down   every thumbs-down. Cheap to pick and biased: users complain
#                 about some failures and not others, so its rate is wrong.
#   stratified    oversample where failures concentrate (Helix declined; a
#                 user complained), fill the rest at random, and weight each
#                 group by its size in the week. More failures found, and a
#                 rate that is still unbiased.
# ---------------------------------------------------------------------------
def refusal_error(r: Request) -> bool:
    return r.tuple_.should_refuse != r.trace.refused


def strata(requests: list[Request]) -> dict[str, list[Request]]:
    return {"declined": [r for r in requests if r.trace.refused],
            "thumbs-down": [r for r in requests if r.thumbs_down and not r.trace.refused],
            "the rest": [r for r in requests if not r.thumbs_down and not r.trace.refused]}


def _draw(name: str, requests: list[Request], budget: int, rng: random.Random) -> tuple[list[Request], float]:
    if name == "random":
        pick = rng.sample(requests, budget)
        return pick, sum(map(refusal_error, pick)) / budget
    if name == "thumbs-down":
        down = [r for r in requests if r.thumbs_down]
        pick = rng.sample(down, min(budget, len(down)))
        return pick, (sum(map(refusal_error, pick)) / len(pick)) if pick else 0.0
    groups = strata(requests)
    want = {"declined": min(len(groups["declined"]), budget // 2),
            "thumbs-down": min(len(groups["thumbs-down"]), budget // 5)}
    want["the rest"] = budget - want["declined"] - want["thumbs-down"]
    pick, est = [], 0.0
    for g, rs in groups.items():
        take = rng.sample(rs, min(want[g], len(rs)))
        pick += take
        if take:
            est += len(rs) * sum(map(refusal_error, take)) / len(take)
    return pick, est / len(requests)


@dataclass
class Strategy:
    name: str
    reviewed: float          # mean answers read
    found: float             # mean refusal errors among them
    estimate: float          # mean estimated refusal-error rate
    low: float               # 2.5th percentile of the estimate over draws
    high: float              # 97.5th percentile
    one_draw: list[Request]  # the first draw, for promotion


STRATEGIES = ("random", "thumbs-down", "stratified")


def compare_strategies(requests: list[Request], budget: int = REVIEW_BUDGET, draws: int = 500,
                       seed: int = SEED) -> list[Strategy]:
    rng = random.Random(f"{seed}:review")
    out = []
    for name in STRATEGIES:
        ests, found, read, first = [], [], [], None
        for _ in range(draws):
            pick, est = _draw(name, requests, budget, rng)
            first = first or pick
            ests.append(est)
            found.append(sum(map(refusal_error, pick)))
            read.append(len(pick))
        ests.sort()
        out.append(Strategy(name, sum(read) / draws, sum(found) / draws, sum(ests) / draws,
                            ests[int(0.025 * draws)], ests[int(0.975 * draws) - 1], first))
    return out


def distinct_failing(reviewed: list[Request]) -> list[Request]:
    seen, out = set(), []
    for r in reviewed:
        if refusal_error(r) and r.key not in seen:
            seen.add(r.key)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Promotion: a failure found in production becomes a suite question
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def promotion_candidates(found: list[Request], suite_texts: set[str], limit: int = 5) -> list[dict]:
    """Failing questions the suite does not hold yet, as entries a person completes."""
    have = {_norm(t) for t in suite_texts}
    out, seen = [], set()
    for r in found:
        k = (_norm(r.text), r.tuple_.asker.office, r.tuple_.asker.role)
        if k in seen or _norm(r.text) in have:
            continue
        seen.add(k)
        what = "missed refusal" if r.tuple_.should_refuse else "false refusal"
        out.append({"id": f"p{len(out) + 1:02d}", "text": r.text, "user_office": r.tuple_.asker.office,
                    "user_role": r.tuple_.asker.role, "should_refuse": r.tuple_.should_refuse,
                    "must_include": [], "must_not_include": [],
                    "found_by": f"production review, week {r.week}: {what}"})
        if len(out) >= limit:
            break
    return out


def decompose(old_sys_old_inst: float, new_sys_old_inst: float, new_sys_new_inst: float) -> dict[str, float]:
    """Split a change in a reported rate into the system's part and the instrument's part."""
    return {"total": new_sys_new_inst - old_sys_old_inst,
            "system": new_sys_old_inst - old_sys_old_inst,
            "instrument": new_sys_new_inst - new_sys_old_inst}


