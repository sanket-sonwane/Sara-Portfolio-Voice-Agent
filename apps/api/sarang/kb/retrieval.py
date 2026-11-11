"""Tier-2 retrieval over long-form evidence.

Two backends behind one interface:

* ``LocalRetriever``  BM25 over ``kb/docs/*.md``. No service, no dependency,
                      used by default and in tests.
* ``PgRetriever``     Postgres full-text + optional pgvector cosine, used when
                      ``DATABASE_URL`` is set and ``asyncpg`` is installed.

The interface returns evidence chunks with a score so the orchestration can decide
whether the result is good enough to speak.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import DOCS_DIR

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*")
# Function words, question words and discourse markers. BM25 cannot rank what
# every chunk contains, so dropping these costs no discrimination - measured
# corpus check: the discourse markers appear in 0-2 of 58 chunks ("actually"
# is the only hit) while they dominate filler-laden queries. Kept in sync
# with ``kb.lookup.STOPWORDS`` where the word sets overlap.
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that", "as", "by",
    "from", "at", "he", "she", "his", "her", "they", "them", "we", "our", "you",
    # Determiners, pronouns and auxiliaries the tier-1 set already stops.
    "me", "my", "am", "us", "him", "your", "i",
    "do", "does", "did", "has", "have", "had", "will", "would", "can", "could",
    "should", "may", "might", "must", "shall",
    # Question and clause words.
    "what", "when", "where", "which", "who", "why", "how", "if", "so", "than",
    "then", "there", "here", "not", "no", "yes", "please", "tell", "say",
    "give", "just", "very", "really", "actually", "also",
    # Discourse markers and filled pauses.
    "however", "mean", "well", "anyway", "basically", "maybe", "probably",
    "honestly", "literally", "yeah", "okay", "ok", "alright", "hmm", "um",
    "uh", "eh", "huh", "kinda", "sorta",
}


def tokens(text: str) -> list[str]:
    out = []
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) > 1 and token not in STOP:
            out.append(token[:-1] if len(token) > 4 and token.endswith("s") else token)
    return out


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str
    score: float = 0.0

    def to_payload(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": " ".join(self.text.split())[:320],
            "score": round(self.score, 3),
        }


def chunk_markdown(text: str, doc_id: str, target_chars: int = 700) -> list[Chunk]:
    """Heading-aware chunking, because the documents are structured that way."""
    chunks: list[Chunk] = []
    state = {"title": doc_id, "heading": ""}
    section: list[str] = []

    def flush() -> None:
        nonlocal section
        if not section:
            return
        body = " ".join(" ".join(section).split())
        section = []
        if not body:
            return
        label = f"{state['title']} — {state['heading']}" if state["heading"] else state["title"]
        if len(body) <= target_chars:
            chunks.append(Chunk(doc_id=doc_id, title=label, text=body))
            return
        words = body.split()
        buffer: list[str] = []
        size = 0
        for word in words:
            if size + len(word) + 1 > target_chars and buffer:
                chunks.append(Chunk(doc_id=doc_id, title=label, text=" ".join(buffer)))
                buffer, size = [], 0
            buffer.append(word)
            size += len(word) + 1
        if buffer:
            chunks.append(Chunk(doc_id=doc_id, title=label, text=" ".join(buffer)))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            flush()
            state["heading"] = stripped[3:].strip()
            continue
        if stripped.startswith("# "):
            flush()
            state["title"] = stripped[2:].strip() or doc_id
            state["heading"] = ""
            continue
        if not stripped:
            continue
        section.append(stripped)
    flush()
    return chunks


class LocalRetriever:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or DOCS_DIR
        self.chunks: list[Chunk] = []
        self._df: dict[str, int] = {}
        self._postings: list[dict[str, float]] = []
        self._load()

    def _load(self) -> None:
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.chunks.extend(chunk_markdown(text, path.stem))
        for chunk in self.chunks:
            counts: dict[str, float] = {}
            for token in tokens(chunk.text):
                counts[token] = counts.get(token, 0.0) + 1.0
            self._postings.append(counts)
            for token in counts:
                self._df[token] = self._df.get(token, 0) + 1

    @property
    def ready(self) -> bool:
        return bool(self.chunks)

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        if not self.chunks:
            return []
        query_tokens = tokens(query)
        if not query_tokens:
            return []
        total = len(self.chunks)
        avg_len = sum(len(p) for p in self._postings) / max(1, total)
        k1, b = 1.4, 0.75
        scored: list[Chunk] = []
        for chunk, postings in zip(self.chunks, self._postings, strict=False):
            length = max(1, len(postings))
            score = 0.0
            for token in set(query_tokens):
                freq = postings.get(token, 0.0)
                if not freq:
                    continue
                idf = math.log(1 + (total - self._df.get(token, 0) + 0.5) / (self._df.get(token, 0) + 0.5))
                denom = freq + k1 * (1 - b + b * length / avg_len)
                score += idf * (freq * (k1 + 1)) / denom
            if score <= 0:
                continue
            scored.append(Chunk(chunk.doc_id, chunk.title, chunk.text, round(score, 3)))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]


class PgRetriever:
    """Postgres/tsvector retrieval. Vector search activates when embeddings exist.

    Deliberately thin: the contract is the same as ``LocalRetriever``. If the
    connection or the tables are missing at startup the factory falls back, so a
    misconfigured database can never take the voice path down.
    """

    def __init__(self, dsn: str, embed_fn=None) -> None:
        self.dsn = dsn
        self.embed_fn = embed_fn
        self._pool = None
        self._loop = None
        self.ready = False

    async def connect(self) -> bool:
        try:
            import asyncpg
        except ModuleNotFoundError:
            return False
        try:
            self._loop = asyncio.get_running_loop()
            self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4, timeout=5)
            async with self._pool.acquire() as connection:
                await connection.execute("create extension if not exists vector")
                await connection.execute(
                    """
                    create table if not exists evidence_chunks (
                        id text primary key,
                        doc_id text not null,
                        title text not null,
                        body text not null,
                        body_tsv tsvector generated always as (to_tsvector('english', body)) stored,
                        embedding vector(384)
                    )
                    """
                )
                await connection.execute(
                    "create index if not exists evidence_chunks_tsv on evidence_chunks using gin(body_tsv)"
                )
            self.ready = True
            return True
        except Exception:
            self._pool = None
            self.ready = False
            return False

    async def ingest(self, chunks: list[Chunk]) -> int:
        if not self._pool:
            return 0
        rows = []
        for chunk in chunks:
            embedding = None
            if self.embed_fn is not None:
                try:
                    embedding = list(await self.embed_fn(chunk.text))
                except Exception:
                    embedding = None
            rows.append((f"{chunk.doc_id}:{hash(chunk.text) & 0xFFFFFFFF}", chunk.doc_id,
                         chunk.title, chunk.text, embedding))
        async with self._pool.acquire() as connection:
            await connection.executemany(
                """
                insert into evidence_chunks (id, doc_id, title, body, embedding)
                values ($1, $2, $3, $4, $5)
                on conflict (id) do update set body = excluded.body, embedding = excluded.embedding
                """,
                rows,
            )
        return len(rows)

    async def search(self, query: str, k: int = 4) -> list[Chunk]:
        if not self._pool:
            return []
        embedding = None
        if self.embed_fn is not None:
            try:
                embedding = list(await self.embed_fn(query))
            except Exception:
                embedding = None
        async with self._pool.acquire() as connection:
            if embedding is not None:
                rows = await connection.fetch(
                    """
                    with lexical as (
                        select id, doc_id, title, body,
                               ts_rank_cd(body_tsv, plainto_tsquery('english', $1)) as lex
                        from evidence_chunks
                        where body_tsv @@ plainto_tsquery('english', $1)
                        order by lex desc limit 20
                    ), semantic as (
                        select id, doc_id, title, body,
                               1 - (embedding <=> $2::vector) as sem
                        from evidence_chunks
                        where embedding is not null
                        order by embedding <=> $2::vector limit 20
                    )
                    select coalesce(l.doc_id, s.doc_id) as doc_id,
                           coalesce(l.title, s.title) as title,
                           coalesce(l.body, s.body) as body,
                           coalesce(l.lex, 0) * 0.5 + coalesce(s.sem, 0) * 0.5 as score
                    from lexical l full outer join semantic s using (id)
                    order by score desc limit $3
                    """,
                    query, embedding, k,
                )
            else:
                rows = await connection.fetch(
                    """
                    select doc_id, title, body,
                           ts_rank_cd(body_tsv, plainto_tsquery('english', $1)) as score
                    from evidence_chunks
                    where body_tsv @@ plainto_tsquery('english', $1)
                    order by score desc limit $2
                    """,
                    query, k,
                )
        return [Chunk(r["doc_id"], r["title"], r["body"], float(r["score"])) for r in rows]

    def search_sync(self, query: str, k: int = 4) -> list[Chunk]:
        """Bridge for the thread-pool caller in ``Orchestrator._retrieve``.

        The turn engine deliberately runs retrieval off the event loop; this
        hands the coroutine back to the loop that owns the pool and waits with a
        hard ceiling so a slow database cannot stall the answer.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return []
        future = asyncio.run_coroutine_threadsafe(self.search(query, k=k), loop)
        try:
            return list(future.result(timeout=0.6))
        except Exception:
            future.cancel()
            return []


_RETRIEVER = None


def get_retriever(database_url: str = ""):
    """Return the active retriever. SQLite-less, dependency-free by default."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = LocalRetriever()
    return _RETRIEVER


def set_retriever(retriever) -> None:
    """Install the retriever chosen at startup (used by the pgvector path)."""
    global _RETRIEVER
    _RETRIEVER = retriever


def search_docs(query: str, k: int = 3) -> list[Chunk]:
    retriever = get_retriever()
    if isinstance(retriever, PgRetriever):
        return retriever.search_sync(query, k=k)
    return retriever.search(query, k=k)
