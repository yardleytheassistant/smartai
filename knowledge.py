"""Knowledge base — the retrieval layer of memory (Layer 3).

STATE.md is project memory and Skills are procedural memory; the knowledge base
is reference memory: versioned documents the agent can search and cite instead
of re-deriving. It lives in `knowledge/` as plain Markdown/text and is chunked
on load.

Retrieval is deterministic token-overlap (BM25-ish) by default, so it works with
zero dependencies and is fully testable offline. If EMBED_MODEL is set, it uses
the OpenAI-compatible /embeddings endpoint (e.g. Ollama's nomic-embed-text) and
ranks by cosine similarity — no extra Python packages required.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from config import config

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Chunk:
    doc: str          # source file name
    text: str
    index: int        # chunk position within the doc


@dataclass
class Hit:
    chunk: Chunk
    score: float


def knowledge_root() -> Path:
    p = Path(config.knowledge_dir).expanduser()
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


def _chunk_text(text: str, *, max_chars: int = 800) -> list[str]:
    """Split on blank lines, then pack paragraphs into ~max_chars chunks."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if buf and len(buf) + len(para) + 2 > max_chars:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def load_chunks(directory: str | Path | None = None) -> list[Chunk]:
    root = Path(directory) if directory is not None else knowledge_root()
    if not root.is_dir():
        return []
    out: list[Chunk] = []
    for f in sorted([*root.glob("*.md"), *root.glob("*.txt")]):
        for i, piece in enumerate(_chunk_text(f.read_text(encoding="utf-8"))):
            out.append(Chunk(doc=f.name, text=piece, index=i))
    return out


# --- Keyword (token-overlap / idf-weighted) scoring -------------------------

def _keyword_rank(query: str, chunks: list[Chunk], k: int) -> list[Hit]:
    q_tokens = set(_tokens(query))
    if not q_tokens or not chunks:
        return []
    # Document frequency for idf weighting.
    df: dict[str, int] = {}
    chunk_tokens: list[set[str]] = []
    for c in chunks:
        toks = set(_tokens(c.text))
        chunk_tokens.append(toks)
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n = len(chunks)
    hits: list[Hit] = []
    for c, toks in zip(chunks, chunk_tokens):
        overlap = q_tokens & toks
        if not overlap:
            continue
        score = sum(math.log(1 + n / df[t]) for t in overlap)
        hits.append(Hit(chunk=c, score=score))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


# --- Embedding scoring (optional, OpenAI-compatible) ------------------------

def _embed(texts: list[str], client) -> list[list[float]]:
    resp = client.embeddings.create(model=config.embed_model, input=texts)
    return [d.embedding for d in resp.data]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _embedding_rank(query: str, chunks: list[Chunk], k: int, client) -> list[Hit]:
    vectors = _embed([query] + [c.text for c in chunks], client)
    qv, cvs = vectors[0], vectors[1:]
    hits = [Hit(chunk=c, score=_cosine(qv, cv)) for c, cv in zip(chunks, cvs)]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


def search(query: str, *, k: int = 4, directory: str | Path | None = None, client=None) -> list[Hit]:
    """Return the top-k knowledge chunks for a query.

    Uses embeddings when EMBED_MODEL is set and a client is available; otherwise
    deterministic keyword ranking.
    """
    chunks = load_chunks(directory)
    if not chunks:
        return []
    if config.embed_model and client is not None:
        try:
            return _embedding_rank(query, chunks, k, client)
        except Exception:  # noqa: BLE001 - fall back to keyword on any embed failure
            pass
    return _keyword_rank(query, chunks, k)


def search_text(query: str, *, k: int = 4, directory: str | Path | None = None, client=None) -> str:
    """Render the top-k hits as a citable text block (for tools / prompts)."""
    hits = search(query, k=k, directory=directory, client=client)
    if not hits:
        return "No matching knowledge found."
    return "\n\n".join(f"[{h.chunk.doc}#{h.chunk.index}] {h.chunk.text}" for h in hits)


def add_document(name: str, content: str, *, directory: str | Path | None = None) -> Path:
    """Write a document into the knowledge base (creating the dir if needed)."""
    root = Path(directory) if directory is not None else knowledge_root()
    root.mkdir(parents=True, exist_ok=True)
    if not name.endswith((".md", ".txt")):
        name += ".md"
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path
