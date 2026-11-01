"""A short tracer, written by hand on purpose.

Day 1 deliberately uses no observability vendor and no evaluation library.
Participants instrument before they measure, so that when a toolkit arrives on
Day 2 they already know what it is reading. The span names below follow the
OpenTelemetry GenAI semantic conventions closely enough that swapping in a
real exporter later is a mechanical change.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .schema import Span, Trace, trace_from_dict


@contextmanager
def span(trace: Trace, name: str, **attributes) -> Iterator[Span]:
    """Open a span on `trace`, attach it, and close it on exit.

    Exceptions are recorded rather than swallowed: a trace of a crashed
    request is still evidence, and is often the most interesting kind.
    """
    s = Span(name=name, attributes=dict(attributes))
    trace.spans.append(s)
    try:
        yield s
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        s.attributes["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        from .schema import _now

        s.ended_at = _now()


def write_jsonl(traces: list[Trace], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[Trace]:
    path = Path(path)
    out: list[Trace] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(trace_from_dict(json.loads(line)))
    return out
