from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx

from biblio_agent_bot.models import Article, EmbeddingRank, TopicConfig

ProgressCallback = Callable[[str], None]
MAX_EMBEDDING_TEXT_CHARS = 6000


class EmbeddingProvider(Protocol):
    name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class NullEmbeddingProvider:
    name = "none"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("No embedding provider configured")


class OpenRouterEmbeddingProvider:
    name = "openrouter"

    def __init__(
        self,
        *,
        model: str,
        batch_size: int,
        cache_path: Path,
        progress: ProgressCallback | None = None,
    ) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for embedding_provider=openrouter")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.cache = EmbeddingCache(cache_path)
        self.progress = progress or (lambda _message: None)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str, str]] = []
        for index, text in enumerate(texts):
            key = cache_key(self.model, text)
            cached = self.cache.get(key)
            if cached is None:
                missing.append((index, key, text))
            else:
                embeddings[index] = cached

        if missing:
            self.progress(
                f"openrouter embeddings: {len(missing)} uncached texts, "
                f"{len(texts) - len(missing)} cache hits."
            )
        else:
            self.progress(f"openrouter embeddings: all {len(texts)} texts loaded from cache.")

        for batch_number, start in enumerate(range(0, len(missing), self.batch_size), start=1):
            batch = missing[start : start + self.batch_size]
            batch_texts = [item[2] for item in batch]
            self.progress(
                f"openrouter embeddings: requesting batch {batch_number} "
                f"({len(batch_texts)} texts) with {self.model}."
            )
            batch_embeddings = self._request_embeddings(batch_texts)
            for (index, key, _text), embedding in zip(batch, batch_embeddings, strict=True):
                embeddings[index] = embedding
                self.cache.set(key, embedding)

        if any(embedding is None for embedding in embeddings):
            raise RuntimeError("Embedding provider returned fewer vectors than requested")
        return [embedding for embedding in embeddings if embedding is not None]

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-OpenRouter-Title"] = title
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        with httpx.Client(timeout=90) as client:
            response = client.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        data = response.json()["data"]
        ordered = sorted(data, key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
        )

    def get(self, key: str) -> list[float] | None:
        row = self.connection.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, vector: list[float]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            (key, json.dumps(vector)),
        )
        self.connection.commit()


def embedding_provider_from_config(
    config: TopicConfig,
    *,
    cache_dir: Path,
    progress: ProgressCallback | None = None,
) -> EmbeddingProvider:
    if config.embedding_provider == "none":
        return NullEmbeddingProvider()
    if config.embedding_provider == "openrouter":
        safe_model = config.embedding_model.replace("/", "__").replace(":", "_")
        return OpenRouterEmbeddingProvider(
            model=config.embedding_model,
            batch_size=config.embedding_batch_size,
            cache_path=cache_dir / f"{safe_model}.sqlite3",
            progress=progress,
        )
    raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")


def rank_articles_by_embedding(
    articles: list[Article],
    config: TopicConfig,
    provider: EmbeddingProvider,
) -> tuple[list[EmbeddingRank], list[Article]]:
    candidates: list[tuple[Article, str, str]] = []
    for article in articles:
        text, source = article_embedding_text(article)
        if text:
            candidates.append((article, text, source))
    if not candidates:
        return [], []

    query_text = query_embedding_text(config)
    embeddings = provider.embed_texts([query_text] + [candidate[1] for candidate in candidates])
    query_embedding = embeddings[0]
    article_embeddings = embeddings[1:]

    scored: list[tuple[Article, str, float]] = []
    for (article, _text, text_source), embedding in zip(candidates, article_embeddings, strict=True):
        scored.append((article, text_source, cosine_similarity(query_embedding, embedding)))

    scored.sort(key=lambda item: item[2], reverse=True)
    ranks: list[EmbeddingRank] = []
    selected_articles: list[Article] = []
    for rank, (article, text_source, score) in enumerate(scored, start=1):
        ranks.append(
            EmbeddingRank(
                stable_id=article.stable_id,
                rank=rank,
                score=score,
                text_source=text_source,
            )
        )
        if rank <= config.embedding_top_k and (
            config.embedding_min_score is None or score >= config.embedding_min_score
        ):
            selected_articles.append(article)
    return ranks, selected_articles


def query_embedding_text(config: TopicConfig) -> str:
    parts = [
        config.title,
        config.question,
        f"Population: {config.population}",
        f"Exposure: {config.exposure}",
        f"Comparator: {config.comparator or ''}",
        f"Outcome: {config.outcome}",
        "Include: " + "; ".join(config.inclusion_criteria),
        "Keywords: " + "; ".join(config.include_terms),
    ]
    return truncate_embedding_text("\n".join(part for part in parts if part))


def article_embedding_text(article: Article) -> tuple[str, str]:
    if article.abstract:
        return truncate_embedding_text(f"{article.title}\n\nAbstract: {article.abstract}"), "title+abstract"
    if article.title:
        return truncate_embedding_text(article.title), "title_only"
    return "", "missing"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def cache_key(model: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}:{digest}"


def truncate_embedding_text(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= MAX_EMBEDDING_TEXT_CHARS:
        return compact
    return compact[:MAX_EMBEDDING_TEXT_CHARS]
