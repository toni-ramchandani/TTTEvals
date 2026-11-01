"""Day 2, Module 04: open coding, axial coding, labelling.

The three steps, in order, and the order matters:

1. **Open coding**: read traces and write free-form notes about what went
   wrong. No categories yet. An LLM should *not*
   do this step: it requires your judgment and taste about what counts as a
   failure.
2. **Axial coding**: cluster those notes into a taxonomy of failure modes.
   An LLM *can* help here, because its job is to organise observations you
   have already made rather than to supply taste. It must never be asked to
   invent the taxonomy from scratch: models have strong priors toward generic
   categories ("hallucination", "tone") and weak priors toward the
   application-specific issues that only reading real traces reveals.
3. **Labelling**: go back through every trace and mark which failure modes
   are present, as 1/0 columns, then compute prevalence.

Stop when you reach saturation: new traces stop producing new categories.
Start with about 100 traces and label at least the first 30 yourself; the
signal to stop is saturation, not the count (`saturation` measures it).
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from helix.config import Config
from helix.llm import complete
from tracing.schema import Trace

AXIAL_PROMPT = """\
Below is a list of open-ended annotations describing failures in an LLM-driven
internal HR policy assistant. Please group them into a small set of coherent
failure categories, where each category captures similar types of mistakes.
Each group should have a short descriptive title and a brief one-line
definition. Do not invent new failure types; only cluster based on what is
present in the notes.

{notes}
"""


@dataclass
class Annotation:
    """One trace, one reader, one judgment.

    `binary` is the holistic acceptable/unacceptable call recommended
    alongside the free-form note. Forcing a yes-or-no sharpens
    thinking in a way a 1-to-5 Likert scale does not, and produces far better
    inter-annotator agreement when rubrics are still underspecified.
    """

    trace_id: str
    query: str
    open_code: str
    acceptable: bool
    labels: set[str] = field(default_factory=set)


def open_coding_sheet(traces: list[Trace], path: str | Path) -> Path:
    """Export traces to a CSV with empty columns for the reader to fill.

    Many teams find a spreadsheet the easiest annotation
    surface. The format matters less than having a complete record you can
    read and annotate.

    Record the FIRST failure you see, not every symptom: one upstream error
    usually cascades, and fixing the first often resolves the chain.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trace_id", "query", "answer", "cited", "open_code", "acceptable"])
        for t in traces:
            w.writerow([
                t.trace_id, t.query, t.answer,
                "; ".join(t.citations), "", "",
            ])
    return path


def load_annotations(path: str | Path) -> list[Annotation]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("open_code"):
                continue
            out.append(Annotation(
                trace_id=row["trace_id"],
                query=row["query"],
                open_code=row["open_code"],
                acceptable=row.get("acceptable", "").strip().lower() in {"1", "y", "yes", "true"},
            ))
    return out


def propose_taxonomy(annotations: list[Annotation], cfg: Config, on_token=None) -> str:
    """Axial coding with model assistance.

    The output is a proposal, never a decision: the model does not
    know how your system works or how you would fix each failure, so it will
    merge things that should stay separate and split things that belong
    together. Expect to rename, merge and split by hand.
    """
    notes = "\n".join(f"- {a.open_code}" for a in annotations)
    return complete([{"role": "user", "content": AXIAL_PROMPT.format(notes=notes)}], cfg,
                    on_token=on_token)[0]


def prevalence(annotations: list[Annotation]) -> dict[str, float]:
    """Share of traces exhibiting each failure mode."""
    n = len(annotations) or 1
    counts: Counter[str] = Counter()
    for a in annotations:
        counts.update(a.labels)
    return {label: round(c / n, 3) for label, c in counts.most_common()}


def saturation(annotations: list[Annotation], block: int = 10) -> list[tuple[int, int, list[str]]]:
    """New failure modes per block of traces, in the order they were read.

    Returns (first trace, last trace, modes first seen in this block) for each
    block. When a block of fresh traces adds no new mode, you have reached
    saturation for this kind of traffic: reading more of the same finds more
    examples, not more kinds. A new kind of traffic (a new office, a new
    topic) can start the count again.
    """
    seen: set[str] = set()
    out = []
    for start in range(0, len(annotations), block):
        chunk = annotations[start:start + block]
        new = []
        for a in chunk:
            for m in sorted(a.labels):
                if m not in seen:
                    seen.add(m)
                    new.append(m)
        out.append((start + 1, start + len(chunk), new))
    return out


def failures_by(annotations: list[Annotation], attribute: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Failing traces per value of one dimension: {value: (failing, total)}.

    `attribute` maps a question to its value (its topic, say, or its kind of
    question). Failures that pile up in one value point at a cause.
    """
    out: dict[str, tuple[int, int]] = {}
    for a in annotations:
        v = attribute.get(a.query)
        if v is None:
            continue
        bad, total = out.get(v, (0, 0))
        out[v] = (bad + (not a.acceptable), total + 1)
    return out


def label_matrix(annotations: list[Annotation], modes: list[str], path: str | Path) -> Path:
    """The label matrix: one row per trace, one 1/0 column per failure mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trace_id", "open_code", *modes])
        for a in annotations:
            w.writerow([a.trace_id, a.open_code, *[int(m in a.labels) for m in modes]])
    return path
