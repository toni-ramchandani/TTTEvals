"""Source → parse → chunk → enrich → embed → index.

Six build-time stages. They run once, before anyone asks a question, and the
only thing that crosses into request time is the index. That is why
re-chunking is not a deploy, it is a rebuild.

Two of the stages here plant Day 1 failures:

* ``chunk_strategy="fixed"`` splits on a token count and ignores document
  structure, so a country heading and the table it governs land in different
  chunks. FP10, severed context.
* ``enrich`` attaches ``access_label`` to every chunk, but the retriever only
  honours it when ``Config.enforce_entitlements`` is on. FP11, entitlement
  leakage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .llm import embed
from tracing.schema import Chunk

HANDBOOK = Path(__file__).parent / "data" / "handbook"
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_WORD = re.compile(r"\S+")


@dataclass
class Document:
    doc_id: str
    doc_version: str
    last_modified: str
    access_label: str
    title: str
    body: str
    path: Path


def parse(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    m = _FRONTMATTER.match(raw)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        raw = raw[m.end() :]
    # Workshop annotations are invisible to the pipeline, as they would be in
    # any real corpus where the comment is an HTML comment.
    raw = _COMMENT.sub("", raw)
    return Document(
        doc_id=meta.get("doc_id", path.stem),
        doc_version=meta.get("doc_version", "1"),
        last_modified=meta.get("last_modified", "unknown"),
        access_label=meta.get("access_label", "all"),
        title=meta.get("title", path.stem),
        body=raw.strip(),
        path=path,
    )


def document_status(handbook: Path = HANDBOOK) -> dict[str, dict]:
    """Which handbook pages are current, and which a newer page supersedes.

    Pages with the same title are versions of one policy; the newest by
    last_modified is current. The agent's check_document_status tool and the
    Day 2 stale-evidence check both read this, so they cannot disagree.
    """
    by_title: dict[str, list[Document]] = {}
    for d in (parse(p) for p in sorted(handbook.glob("*.md"))):
        by_title.setdefault(d.title.strip().lower(), []).append(d)
    status: dict[str, dict] = {}
    for group in by_title.values():
        group.sort(key=lambda d: d.last_modified)
        newest = group[-1]
        for d in group:
            current = d is newest
            status[d.doc_id] = {
                "doc_id": d.doc_id,
                "doc_version": d.doc_version,
                "last_modified": d.last_modified,
                "status": "current" if current else "superseded",
                "superseded_by": None if current else newest.doc_id,
            }
    return status


def _fixed_chunks(doc: Document, cfg: Config) -> list[tuple[str, str]]:
    """Fixed-size windows over the raw text. Structure is not consulted.

    Returns (heading_path, text) pairs. heading_path is recorded for the
    trace but is NOT part of the chunk text, which is the whole point: the
    retriever and the model never see it.
    """
    words = _WORD.findall(doc.body)
    size = cfg.chunk_tokens
    step = max(1, size - cfg.chunk_overlap)
    out: list[tuple[str, str]] = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            continue
        text = " ".join(window)
        headings = re.findall(r"#{1,3}\s+([^\n#]+)", text)
        out.append((headings[-1].strip() if headings else doc.title, text))
    return out


def _structural_chunks(doc: Document, cfg: Config) -> list[tuple[str, str]]:
    """Split on H2 sections and keep each section whole.

    With ``prepend_heading`` on, the heading path is written into the chunk
    text, so the qualifier travels with the fact.
    """
    parts = re.split(r"\n(?=##\s)", doc.body)
    out: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"#{1,3}\s+([^\n]+)", part)
        heading = m.group(1).strip() if m else doc.title
        path = f"{doc.title} > {heading}" if heading != doc.title else doc.title
        text = f"{path}\n\n{part}" if cfg.prepend_heading else part
        out.append((path, text))
    return out


def build_index(cfg: Config, handbook: Path = HANDBOOK) -> "Index":
    docs = [parse(p) for p in sorted(handbook.glob("*.md"))]
    chunks: list[Chunk] = []
    for doc in docs:
        pairs = (
            _structural_chunks(doc, cfg)
            if cfg.chunk_strategy == "structural"
            else _fixed_chunks(doc, cfg)
        )
        for i, (heading_path, text) in enumerate(pairs):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{i:02d}",
                    doc_id=doc.doc_id,
                    text=text,
                    doc_version=doc.doc_version,
                    last_modified=doc.last_modified,
                    heading_path=heading_path,
                    access_label=doc.access_label,
                )
            )
    vectors = embed([c.text for c in chunks], cfg)
    return Index(chunks=chunks, vectors=vectors, config_hash=cfg.hash())


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray
    config_hash: str

    def __len__(self) -> int:
        return len(self.chunks)
