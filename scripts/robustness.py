"""Day 3 hands-on, Module 09: robustness and prompt-injection tests, against a baseline.

Seven families of behavioural tests (data/queries/day3_robustness.json), run
on Day 1's Helix and on the repaired configuration:

  STEP 1  pass rate per family, per configuration
  STEP 2  every failing case: what failed, and what the trace shows
  STEP 3  the indirect injection, read from the trace: where the planted page
          ranked, whether it reached the model, what the citation check said
  STEP 4  with --toolkits (live, toolkit environment): Ragas Noise
          Sensitivity on the noise family, beside our check

Families: paraphrase and office alias (CheckList invariance), asker's office
(CheckList directional), noise and negative rejection (RGB), direct and
indirect prompt injection (OWASP LLM01:2025, LLM02:2025).

The planted page and the noise chunk exist only in memory while this runs.
Every call is live on your models; offline is the reference run. Traces go
to data/traces/day3_robustness.jsonl.

Run:  python run.py robustness
      python run.py robustness --family direct_injection,indirect_injection
      python run.py robustness --toolkits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from evals.day3_robustness import (  # noqa: E402
    FAMILIES,
    OPT_IN,
    check,
    load_suite,
    noise_chunk,
    planted_chunk,
    rate,
    with_planted,
)
from helix import DEFAULT, REPAIRED, answer, build_index  # noqa: E402
from helix.console import banner, progress  # noqa: E402
from tracing import write_jsonl  # noqa: E402

OUT = ROOT / "data" / "traces" / "day3_robustness.jsonl"
SYSTEMS = (("default", DEFAULT), ("repaired", REPAIRED))


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def short(text: str, n: int = 64) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="", help=f"comma-separated, from: {', '.join(FAMILIES + OPT_IN)}")
    ap.add_argument("--toolkits", action="store_true", help="Ragas Noise Sensitivity on the noise family (live)")
    args = ap.parse_args(argv)
    fams = [f.strip() for f in args.family.split(",") if f.strip()] or list(FAMILIES)
    bad = [f for f in fams if f not in FAMILIES + OPT_IN]
    if bad:
        print(f"unknown family: {', '.join(bad)} (choose from {', '.join(FAMILIES + OPT_IN)})")
        return 2

    banner(DEFAULT)
    suite = load_suite()
    cases = [c for c in suite["cases"] if c["family"] in fams]
    noise, planted = noise_chunk(suite), planted_chunk(suite)
    live = not DEFAULT.offline
    outcomes: dict[str, dict[str, list]] = {name: {f: [] for f in fams} for name, _ in SYSTEMS}
    traces = []
    total, done = len(cases) * len(SYSTEMS), 0
    for name, cfg in SYSTEMS:
        index = build_index(cfg)
        planted_index = with_planted(index, planted, cfg) if "indirect_injection" in fams else None
        for c in cases:
            fam = c["family"]
            idx = planted_index if fam == "indirect_injection" else index
            extra = [noise] if fam == "noise" else None
            t = answer(c["text"], idx, cfg, user_office=c.get("user_office", "amsterdam"),
                       user_role=c.get("user_role", "employee"), extra_context=extra)
            t.system = f"helix-{name}"
            traces.append(t)
            outcomes[name][fam].append(check(c, t, planted.chunk_id if fam == "indirect_injection" else ""))
            done += 1
            if live:
                progress(f"{name} {c['id']}", done, total)

    rule("STEP 1  Pass rate per family: the baseline Day 4 can gate on")
    print(f"{'family':<20} {'method':<24} {'default':>9} {'repaired':>9}")
    print("-" * 78)
    for f in fams:
        src = suite["families"][f]["source"]
        cells = []
        for name, _ in SYSTEMS:
            p, n = rate(outcomes[name][f])
            cells.append(f"{p} of {n}")
        print(f"{f:<20} {src:<24} {cells[0]:>9} {cells[1]:>9}")
    tot = {name: (sum(rate(outcomes[name][f])[0] for f in fams), sum(rate(outcomes[name][f])[1] for f in fams))
           for name, _ in SYSTEMS}
    print("-" * 78)
    d_all = f"{tot['default'][0]} of {tot['default'][1]}"
    r_all = f"{tot['repaired'][0]} of {tot['repaired'][1]}"
    print(f"{'all':<20} {'':<24} {d_all:>9} {r_all:>9}")
    if "injection_patterns" not in fams:
        print("\nThree direct injections are too few to call a defence tested. Twenty more")
        print("patterns, opt-in:  python run.py robustness --family injection_patterns")
    else:
        print("\nThe twenty patterns, one row each (default / repaired):")
        for o_d, o_r in zip(outcomes["default"]["injection_patterns"], outcomes["repaired"]["injection_patterns"]):
            print(f"  {o_d.case['id']:<4} {o_d.case.get('pattern', ''):<30} "
                  f"{'pass' if o_d.passed else 'FAIL':>5} {'pass' if o_r.passed else 'FAIL':>9}")

    rule("STEP 2  Every failing case, and what the trace shows")
    for name, _ in SYSTEMS:
        failing = [o for f in fams for o in outcomes[name][f] if not o.passed]
        print(f"\n{name}: {len(failing)} failing")
        for o in failing:
            print(f"  {o.case['id']:<3} {o.case['family']:<19} {short(o.case['text'], 52)}")
            print(f"      why     {short('; '.join(o.problems), 64)}")
            print(f"      answer  {short(o.trace.answer, 64)}")

    if "indirect_injection" in fams:
        rule("STEP 3  The planted page, followed through the trace")
        print("An attacker who can edit one page in the store needs no access to Helix.")
        print("The trace shows where the page ranked, whether it reached the model, and")
        print("whether the citation check noticed it is not a handbook page.\n")
        for name, _ in SYSTEMS:
            for o in outcomes[name]["indirect_injection"]:
                print(f"  {name:<9} {o.case['id']:<3} {'pass' if o.passed else 'FAIL':<5} {o.notes[0] if o.notes else ''}")
                for extra in o.notes[1:]:
                    print(f"  {'':<9} {'':<3} {'':<5} {short(extra, 58)}")
        print("\nNeither the entitlement filter nor the repaired chunking stops this page:")
        print("it is labelled `all`. OWASP's defences include treating retrieved text as")
        print("data, not instructions, and regular adversarial testing, like this run.")

    if args.toolkits and "noise" in fams:
        rule("STEP 4  Ragas Noise Sensitivity beside our check (lower is better)")
        if not live:
            print("Needs a judge model: run without --offline.")
        else:
            from toolkits import env
            from toolkits.judge import JudgeStats, ragas_llm

            env.prepare()
            if not env.installed("ragas"):
                print("Ragas is not installed here (see requirements-toolkits.txt).")
            else:
                from toolkits import ragas_metrics as R

                stats = JudgeStats()
                metric = R.build_noise(ragas_llm(DEFAULT, stats))
                print(f"{'config':<9} {'id':<3} {'ours':>5} {'Ragas noise':>12}")
                print("-" * 34)
                for name, _ in SYSTEMS:
                    for o in outcomes[name]["noise"]:
                        ctx = [c.text for c in o.trace.retrieved if not c.dropped_at]
                        m = R.score_noise(metric, o.case["text"], o.trace.answer, o.case.get("reference", ""), ctx)
                        v = "-" if m.score is None else f"{m.score:.2f}"
                        print(f"{name:<9} {o.case['id']:<3} {'pass' if o.passed else 'FAIL':>5} {v:>12}")
                print(f"\ntoolkit judge: {stats.line()}")

    print("\nA family's pass rate is a baseline, not a verdict. Day 4 decides which")
    print("families are floors (direct injection: no leak, ever) and which may move.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(traces, OUT)
    print(f"\nTraces written to {OUT.relative_to(ROOT)} ({len(traces)} answers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
