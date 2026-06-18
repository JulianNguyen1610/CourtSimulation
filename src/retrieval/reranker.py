"""Semantic reranking utilities for legal evidence retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from src.models import EvidenceDocument
from src.retrieval.legal_retriever import tokenize


class Reranker(Protocol):
    """Rank retrieved evidence documents for a query."""

    def rerank(
        self,
        query: str,
        documents: list[EvidenceDocument],
        top_k: int,
    ) -> list[EvidenceDocument]:
        """Return top-k reranked documents."""


@dataclass(frozen=True)
class SemanticRerankerConfig:
    """Config for sentence-transformer semantic reranking."""

    model_name: str = "BAAI/bge-m3"
    batch_size: int = 16


class LexicalReranker:
    """Dependency-free overlap reranker used for tests/offline fallback."""

    def rerank(
        self,
        query: str,
        documents: list[EvidenceDocument],
        top_k: int,
    ) -> list[EvidenceDocument]:
        query_terms = set(tokenize(query))
        scored = []
        for document in documents:
            doc_terms = set(tokenize(document.text))
            overlap = len(query_terms.intersection(doc_terms))
            score = float(overlap) + float(document.score or 0.0) * 0.01
            scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            document.model_copy(
                update={
                    "score": score,
                    "metadata": {
                        **document.metadata,
                        "rerank_score": score,
                        "reranker": "lexical_overlap",
                    },
                }
            )
            for score, document in scored[:top_k]
        ]


class SemanticReranker:
    """Sentence-transformer reranker using cosine similarity.

    Imports are lazy so BM25-only and CI runs do not require model downloads.
    """

    def __init__(self, config: SemanticRerankerConfig | None = None) -> None:
        self.config = config or SemanticRerankerConfig()
        self._model = None

    def rerank(
        self,
        query: str,
        documents: list[EvidenceDocument],
        top_k: int,
    ) -> list[EvidenceDocument]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not documents:
            return []

        model = self._get_model()
        texts = [document.text for document in documents]
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )[0]
        document_embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        scored = []
        for document, embedding in zip(documents, document_embeddings, strict=True):
            semantic_score = _dot(query_embedding, embedding)
            combined_score = semantic_score + math.log1p(float(document.score or 0.0)) * 0.05
            scored.append((combined_score, semantic_score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            document.model_copy(
                update={
                    "score": combined_score,
                    "metadata": {
                        **document.metadata,
                        "rerank_score": semantic_score,
                        "reranker": self.config.model_name,
                    },
                }
            )
            for combined_score, semantic_score, document in scored[:top_k]
        ]

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Semantic reranking requires `sentence-transformers`. "
                    "Use retrieval method bm25_only/off or install requirements."
                ) from exc
            self._model = SentenceTransformer(self.config.model_name)
        return self._model


def _dot(left, right) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right, strict=True)))
