"""Legal evidence retrieval with BM25 rough retrieval and optional reranking."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

from src.models import CaseProfile, EvidenceDocument

if TYPE_CHECKING:
    from src.retrieval.reranker import Reranker


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
RetrievalMethod = Literal["off", "bm25_only", "bm25_rerank"]

_RETRIEVAL_METHOD_ALIASES = {
    "lightweight_bm25": "bm25_only",
}


def normalize_retrieval_method(method: str) -> RetrievalMethod:
    normalized = _RETRIEVAL_METHOD_ALIASES.get(method, method)
    if normalized not in ("off", "bm25_only", "bm25_rerank"):
        raise ValueError(f"Unsupported retrieval method: {method}")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval settings for ablation-friendly experiments."""

    method: RetrievalMethod = "bm25_only"
    rough_top_n: int = 100
    legal_evidence_top_k: int = 5
    reranker_model: str = "BAAI/bge-m3"
    uts_vlc_dataset: str = "VietnamAIHub/UTS_VLC"
    uts_vlc_split: str = "train"


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese/English legal text with a small Unicode regex."""

    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class LegalRetriever:
    """BM25 retriever over legal contexts.

    This intentionally avoids fitting on labels or gold answers. For ViLQA, the
    indexed text is only the legal `context` and optional question metadata.
    """

    def __init__(
        self,
        documents: list[EvidenceDocument],
        k1: float = 1.5,
        b: float = 0.75,
        reranker: "Reranker | None" = None,
        method: RetrievalMethod = "bm25_only",
        rough_top_n: int = 100,
    ) -> None:
        if not documents:
            raise ValueError("LegalRetriever requires at least one document.")
        self.method = normalize_retrieval_method(method)

        self.documents = documents
        self.k1 = k1
        self.b = b
        self.reranker = reranker
        self.rough_top_n = rough_top_n
        self._tokenized_documents = [tokenize(document.text) for document in documents]
        self._doc_term_counts = [
            Counter(tokens) for tokens in self._tokenized_documents
        ]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized_documents]
        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        self._idf = self._compute_idf()

    @classmethod
    def from_cases(
        cls,
        cases: list[CaseProfile],
        *,
        extra_documents: list[EvidenceDocument] | None = None,
        reranker: "Reranker | None" = None,
        method: RetrievalMethod = "bm25_only",
        rough_top_n: int = 100,
    ) -> "LegalRetriever":
        """Build a retriever from case contexts without using answers."""

        documents = [
            EvidenceDocument(
                doc_id=f"{case.case_id}-context",
                text=case.context,
                source="vilqa_context",
                metadata={
                    "case_id": case.case_id,
                    "question": case.question,
                    "row_index": case.metadata.get("row_index"),
                    "article_id": case.metadata.get("article_id"),
                    "law_name": case.metadata.get("law_name"),
                    "source_type": "alqac_context",
                },
            )
            for case in cases
        ]
        if extra_documents:
            documents.extend(extra_documents)
        return cls(
            documents,
            reranker=reranker,
            method=method,
            rough_top_n=rough_top_n,
        )

    @classmethod
    def from_documents(
        cls,
        documents: list[EvidenceDocument],
        *,
        reranker: "Reranker | None" = None,
        method: RetrievalMethod = "bm25_only",
        rough_top_n: int = 100,
    ) -> "LegalRetriever":
        """Build a retriever from pre-normalized legal documents."""

        return cls(
            documents,
            reranker=reranker,
            method=method,
            rough_top_n=rough_top_n,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[EvidenceDocument]:
        """Return top-k evidence documents for the configured retrieval method."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.method == "off":
            return []

        query_tokens = tokenize(query)
        scored_documents = []
        for index, document in enumerate(self.documents):
            score = self._score(query_tokens, index)
            if score <= 0.0:
                continue
            scored_documents.append((score, document))

        scored_documents.sort(key=lambda item: item[0], reverse=True)
        rough_top_n = max(top_k, self.rough_top_n)
        rough_documents = [
            document.model_copy(update={"score": score})
            for score, document in scored_documents[:rough_top_n]
        ]
        if self.method == "bm25_rerank":
            if self.reranker is None:
                raise ValueError("bm25_rerank requires a reranker instance.")
            return self.reranker.rerank(query, rough_documents, top_k=top_k)
        return rough_documents[:top_k]

    def _compute_idf(self) -> dict[str, float]:
        document_count = len(self.documents)
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokenized_documents:
            document_frequency.update(set(tokens))

        return {
            term: math.log(1.0 + (document_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def _score(self, query_tokens: list[str], document_index: int) -> float:
        score = 0.0
        term_counts = self._doc_term_counts[document_index]
        doc_length = self._doc_lengths[document_index]
        for term in query_tokens:
            term_frequency = term_counts.get(term, 0)
            if term_frequency == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * doc_length / self._avg_doc_length
            )
            score += idf * (term_frequency * (self.k1 + 1.0)) / denominator
        return score


def load_uts_vlc_documents(
    dataset_name: str = "VietnamAIHub/UTS_VLC",
    split: str = "train",
    limit: int | None = None,
) -> list[EvidenceDocument]:
    """Load Vietnamese legal corpus documents from Hugging Face.

    The loader is field-tolerant because public legal datasets often differ in
    column names across versions. It never includes QA gold answers.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading UTS_VLC requires the `datasets` package. "
            "Install requirements or disable external legal corpus retrieval."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    documents: list[EvidenceDocument] = []
    for index, row in enumerate(dataset):
        if limit is not None and len(documents) >= limit:
            break
        if not isinstance(row, dict):
            continue
        text = _first_text(row, ("text", "content", "article_text", "document", "law_text"))
        if not text:
            continue
        article_id = _first_text(row, ("article_id", "article", "article_number", "id"))
        law_name = _first_text(row, ("law_name", "law", "title", "document_title"))
        source = _first_text(row, ("source", "source_url", "url")) or dataset_name
        documents.append(
            EvidenceDocument(
                doc_id=f"uts_vlc-{article_id or index}",
                text=text,
                source=source,
                metadata={
                    "article_id": article_id,
                    "law_name": law_name,
                    "source_type": "uts_vlc_regulation",
                    "row_index": index,
                    "dataset": dataset_name,
                    "split": split,
                },
            )
        )
    return documents


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
