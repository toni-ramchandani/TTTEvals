"""Build time for GraphRAG: handbook pages -> facts -> graph.

Vanilla RAG decides its unit of evidence at query time: whatever chunks the
retriever returns. GraphRAG decides it at build time, when an extractor reads
every page once and writes down atomic facts with their qualifiers attached.
That moves a whole class of failure from request time to build time, where it
is cheaper to fix and far more expensive to miss: a fact extracted wrongly is
wrong for every question that touches it, permanently, until the next rebuild.

By default the extractor is a set of rules over tables and sentences, so the
graph is the same on every laptop. Set ``HELIX_GRAPH_EXTRACTOR=llm`` and your
live model does the extraction instead, prompted to emit JSON. Either way, provenance
(document, version, date, access label, source chunk) is attached by code,
never by the model. A model asked to remember where a fact came from will
sometimes forget, and lineage that is sometimes missing is not lineage.

Three build switches, each of which plants one graph failure when turned off:

=====================  =====================================================
``unit="section"``     extract from whole sections. ``"chunk"`` extracts from
                       the Day 1 fixed-size chunks instead, so a table row can
                       lose its office heading at a chunk boundary and be
                       written down as company-wide. GR2.
``carry_access_label`` copy the page's access label onto every fact. Off, the
                       graph store has no ACL column and every fact defaults to
                       visible. GR4.
``infer_supersession`` add a SUPERSEDES edge when two pages share a title and
                       one is newer. Off, the 2022 policy's facts sit in the
                       graph as if current. GR3 (only if the query side does
                       not filter either).
=====================  =====================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from helix.config import DEFAULT, REPAIRED, Config
from helix.ingest import HANDBOOK, Document, _fixed_chunks, _structural_chunks, parse
from helix.structure import OFFICES, office_mentions, offices_in, table_rows

from .graph import Fact, Graph


@dataclass(frozen=True)
class GraphBuildConfig:
    unit: str = "section"            # "section" | "chunk"
    carry_access_label: bool = True
    infer_supersession: bool = True
    #: "rules" (default) reads tables and sentences with fixed rules, so the
    #: graph is identical on every laptop. "llm" asks your live model to
    #: extract the facts instead: set HELIX_GRAPH_EXTRACTOR=llm in .env and
    #: compare the two graphs. Expect the model to miss or misfile some facts.
    extractor: str = field(default_factory=lambda: os.getenv("HELIX_GRAPH_EXTRACTOR", "rules"))

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


_SECTION_CFG = Config(chunk_strategy="structural", prepend_heading=False)


def _units(doc: Document, bcfg: GraphBuildConfig) -> list[tuple[str, str, str]]:
    """(chunk_id, heading_path, text) for every unit the extractor reads.

    Chunk ids match the Helix index built with the same strategy, so a fact's
    ``source_chunk_id`` points at a real chunk a reviewer can open.
    """
    pairs = _fixed_chunks(doc, DEFAULT) if bcfg.unit == "chunk" else _structural_chunks(doc, _SECTION_CFG)
    return [(f"{doc.doc_id}-{i:02d}", path, text) for i, (path, text) in enumerate(pairs)]


def _office_label(office: str | None) -> str:
    return f"{office.title()} office" if office else "All offices"


def _office_before(mentions: list[tuple[int, str]], pos: int) -> str | None:
    """The office most recently mentioned before ``pos`` in this unit.

    This is how a reader attributes a table row to the heading above it. It
    only sees the current unit, so a heading that fell into the previous chunk
    is invisible, and the row is attributed to nobody.
    """
    office = None
    for p, o in mentions:
        if p < pos:
            office = o
    return office


# ---------------------------------------------------------------------------
# Rule-based extraction
# ---------------------------------------------------------------------------
_PROSE_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "parental_leave",
        re.compile(
            r"entitled\s+to\s+(\d+)\s+weeks\s+of\s+paid\s+parental\s+leave,\s+equal\s+to\s+"
            r"(\d+)\s+working\s+days,\s+at\s+(\d+)%\s+of\s+base\s+salary",
            re.I,
        ),
        "All offices, any caregiver role: {0} weeks of paid parental leave "
        "({1} working days) at {2}% of base salary.",
    ),
    (
        "annual_leave",
        re.compile(r"receive\s+(\d+)\s+days\s+of\s+paid\s+annual\s+leave\s+per\s+calendar\s+year", re.I),
        "All offices: {0} days of paid annual leave per calendar year, in addition to public holidays.",
    ),
    (
        "annual_leave",
        re.compile(
            r"Up\s+to\s+(\d+)\s+days\s+may\s+be\s+carried\s+into\s+the\s+following\s+year\s+"
            r"and\s+must\s+be\s+taken\s+by\s+(\d+\s+\w+)",
            re.I,
        ),
        "All offices: up to {0} days of annual leave may be carried over, to be taken by {1}.",
    ),
]


def _rule_facts(doc: Document, chunk_id: str, section: str, text: str) -> list[dict]:
    """Facts in one unit, before provenance is attached."""
    out: list[dict] = []
    mentions = office_mentions(text)
    rows = table_rows(text)
    notice_header: list[str] | None = None

    for pos, cells in rows:
        head = cells[0].lower()

        if head == "office":
            notice_header = cells
            continue

        # Parental leave row: Primary caregiver | 26 | 130 | 100% for the first 9 weeks
        if head in ("primary caregiver", "secondary caregiver") and len(cells) >= 4:
            office = _office_before(mentions, pos)
            role = head.split()[0]
            weeks, days, topup = cells[1], cells[2], cells[3]
            out.append({
                "topic": "parental_leave",
                "office": office,
                "slug": f"{office or 'all'}-{role}",
                "attrs": {"role": role, "weeks": int(weeks), "working_days": int(days), "salary_top_up": topup},
                "statement": f"{_office_label(office)}, {role} caregiver: {weeks} weeks of parental leave "
                             f"({days} working days), salary top-up {topup}.",
            })
            continue

        # Notice row: Dublin | 4 weeks | 6 weeks | 12 weeks
        row_offices = offices_in(cells[0], aliases=False)
        if row_offices and len(cells) >= 4 and cells[0].lower() in OFFICES:
            office = row_offices[0]
            labels = notice_header[1:] if notice_header else [f"column {i}" for i in range(2, len(cells) + 1)]
            bands = dict(zip(labels, cells[1:]))
            parts = "; ".join(f"{k.lower()} {v}" for k, v in bands.items())
            out.append({
                "topic": "notice_period",
                "office": office,
                "slug": f"{office}-notice",
                "attrs": {"bands": bands, "header_found": notice_header is not None},
                "statement": f"{_office_label(office)} notice period on resignation: {parts}.",
            })
            continue

        # Salary band row: E4 | Senior engineer | 81000 | 95000 | 109000
        if re.fullmatch(r"E\d", cells[0]) and len(cells) >= 5:
            office = _office_before(mentions, pos)
            level, title, lo, mid, hi = cells[:5]
            out.append({
                "topic": "salary_band",
                "office": office,
                "slug": f"{office or 'all'}-{level.lower()}",
                "attrs": {"level": level, "title": title, "minimum": int(lo), "midpoint": int(mid), "maximum": int(hi)},
                "statement": f"{_office_label(office)}, {level} {title}: salary band minimum {lo}, "
                             f"midpoint {mid}, maximum {hi}.",
            })

    for topic, pattern, template in _PROSE_RULES:
        for m in pattern.finditer(text):
            groups = m.groups()
            slug = f"all-{topic}-{groups[0]}"
            attrs: dict = {"values": list(groups)}
            if topic == "parental_leave":
                attrs = {"role": "any", "weeks": int(groups[0]), "working_days": int(groups[1]),
                         "salary_top_up": f"{groups[2]}% of base salary"}
            out.append({
                "topic": topic,
                "office": None,
                "slug": slug,
                "attrs": attrs,
                "statement": template.format(*groups),
            })
    return out


# ---------------------------------------------------------------------------
# Model-based extraction (live path)
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """\
### Role
You extract facts from an employee handbook into a knowledge graph.

### Instructions
Read the extract below and list every fact about these topics only:
parental_leave, notice_period, salary_band, annual_leave.
For each fact give: topic; office (one of dublin, amsterdam, berlin, london,
singapore, or null if it applies to every office); a one-sentence statement
that includes the office and every qualifier; and attrs, a JSON object of the
numbers in the fact. Do not include where the fact came from. That is added
separately.

### Output format
A JSON list and nothing else.

### Extract
{text}
"""


def _llm_facts(doc: Document, chunk_id: str, section: str, text: str, cfg: Config) -> list[dict]:
    from helix.llm import complete

    raw, _ = complete([{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}], cfg)
    match = re.search(r"\[.*\]", raw, re.S)
    try:
        items = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        items = []
    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict) or "statement" not in it:
            continue
        office = (it.get("office") or None)
        office = office.lower() if isinstance(office, str) and office.lower() in OFFICES else None
        out.append({
            "topic": str(it.get("topic", "unknown")),
            "office": office,
            "slug": f"{office or 'all'}-{i}",
            "attrs": it.get("attrs") or {},
            "statement": str(it["statement"]),
        })
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_graph(
    bcfg: GraphBuildConfig = GraphBuildConfig(),
    handbook: Path = HANDBOOK,
    model_cfg: Config = REPAIRED,
) -> Graph:
    g = Graph()
    g.add_node("company", "company", "All offices")
    for office in OFFICES:
        g.add_node(f"office:{office}", "office", office.title())
        g.add_edge(f"office:{office}", "PART_OF", "company")

    docs = [parse(p) for p in sorted(handbook.glob("*.md"))]
    units_read = 0
    for doc in docs:
        g.add_node(
            f"doc:{doc.doc_id}", "document", doc.title,
            doc_version=doc.doc_version, last_modified=doc.last_modified,
            access_label=doc.access_label if bcfg.carry_access_label else "all",
        )
        for chunk_id, section, text in _units(doc, bcfg):
            units_read += 1
            raw = (
                _llm_facts(doc, chunk_id, section, text, model_cfg)
                if bcfg.extractor == "llm" and not model_cfg.offline  # offline: always rules
                else _rule_facts(doc, chunk_id, section, text)
            )
            for f in raw:
                g.add_node(f"topic:{f['topic']}", "topic", f["topic"])
                g.add_fact(Fact(
                    fact_id=f"{chunk_id}#{f['slug']}",
                    topic=f["topic"],
                    statement=f["statement"],
                    office=f["office"],
                    attrs=f["attrs"],
                    doc_id=doc.doc_id,
                    doc_version=doc.doc_version,
                    last_modified=doc.last_modified,
                    # A store with no ACL column defaults every fact to visible.
                    access_label=doc.access_label if bcfg.carry_access_label else "all",
                    source_chunk_id=chunk_id,
                    section=section,
                ))

    if bcfg.infer_supersession:
        by_title: dict[str, list[Document]] = {}
        for d in docs:
            by_title.setdefault(d.title.strip().lower(), []).append(d)
        for group in by_title.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda d: d.last_modified)
            newest = group[-1]
            for older in group[:-1]:
                g.add_edge(
                    f"doc:{newest.doc_id}", "SUPERSEDES", f"doc:{older.doc_id}",
                    rule="same title, newer last_modified", inferred=True,
                )

    g.build_info = {
        "build_config": asdict(bcfg),
        "build_hash": bcfg.hash(),
        "documents": len(docs),
        "units_read": units_read,
        "facts": len(g.facts),
    }
    return g
