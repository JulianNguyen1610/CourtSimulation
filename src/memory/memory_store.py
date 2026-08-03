"""JSON-backed three-tier memory store for debate experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.llm import LLMClient
from src.models import CaseProfile, DebateResult, EvalResult, MemoryContext
from src.retrieval.legal_retriever import tokenize

MemoryMode = Literal["off", "read_only", "read_update"]
MemoryRetrieval = Literal["lexical", "embedding"]


@dataclass(frozen=True)
class MemoryConfig:
    """Config for memory retrieval/update ablations."""

    mode: MemoryMode = "read_update"
    retrieval: MemoryRetrieval = "lexical"
    top_k: int = 5
    max_entries_per_bucket: int = 1000
    embedding_model: str = "intfloat/multilingual-e5-large"
    prompt_dir: str = "configs/prompts"


class MemoryStore:
    """Store regulations, debate experiences, and case memories."""

    def __init__(
        self,
        path: str | Path = "memory-bank/baseline_memory.json",
        *,
        mode: MemoryMode = "read_update",
        retrieval: MemoryRetrieval = "lexical",
        max_entries_per_bucket: int = 1000,
        embedding_model: str = "intfloat/multilingual-e5-large",
        reflection_llm: LLMClient | None = None,
        prompt_dir: str | Path = "configs/prompts",
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self.retrieval = retrieval
        self.max_entries_per_bucket = max_entries_per_bucket
        self.embedding_model = embedding_model
        self.reflection_llm = reflection_llm
        self.prompt_dir = Path(prompt_dir)
        self.regulations: list[dict[str, Any]] = []
        self.experiences: list[dict[str, Any]] = []
        self.cases: list[dict[str, Any]] = []
        self._embedding_model = None

    @classmethod
    def load(
        cls,
        path: str | Path = "memory-bank/baseline_memory.json",
        *,
        mode: MemoryMode = "read_update",
        retrieval: MemoryRetrieval = "lexical",
        max_entries_per_bucket: int = 1000,
        embedding_model: str = "intfloat/multilingual-e5-large",
        reflection_llm: LLMClient | None = None,
        prompt_dir: str | Path = "configs/prompts",
    ) -> "MemoryStore":
        """Load a memory store, returning an empty store if the file is absent."""

        store = cls(
            path,
            mode=mode,
            retrieval=retrieval,
            max_entries_per_bucket=max_entries_per_bucket,
            embedding_model=embedding_model,
            reflection_llm=reflection_llm,
            prompt_dir=prompt_dir,
        )
        if not store.path.exists():
            return store

        with store.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        store.regulations = list(data.get("regulations", []))
        store.experiences = list(data.get("experiences", []))
        store.cases = list(data.get("cases", []))
        return store

    def save(self) -> None:
        """Persist memory to disk."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "regulations": self.regulations,
            "experiences": self.experiences,
            "cases": self.cases,
        }
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def snapshot_hash(self) -> str:
        """Hash the current inference memory without writing it."""
        import hashlib
        payload = json.dumps({"regulations": self.regulations, "experiences": self.experiences, "cases": self.cases}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_case_isolation(self, forbidden_case_ids: set[str]) -> None:
        contaminated = sorted({str(entry.get("case_id")) for bucket in (self.regulations, self.experiences, self.cases) for entry in bucket if str(entry.get("case_id")) in forbidden_case_ids})
        if contaminated:
            raise ValueError(f"Memory contamination: entries reference validation/test case IDs: {contaminated[:10]}")

    def query(self, case: CaseProfile, top_k: int = 5) -> MemoryContext:
        """Retrieve relevant memory entries without returning same-case memories."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.mode == "off":
            return MemoryContext()

        query_text = f"{case.question} {case.context[:1000]}"
        return MemoryContext(
            regulations=self._top_entries(
                self.regulations,
                query_text,
                top_k,
                exclude_case_id=case.case_id,
            ),
            experiences=self._top_entries(
                self.experiences,
                query_text,
                top_k,
                exclude_case_id=case.case_id,
            ),
            cases=self._top_entries(
                self.cases,
                query_text,
                top_k,
                exclude_case_id=case.case_id,
            ),
        )

    def update_from_debate(
        self,
        case: CaseProfile,
        result: DebateResult,
        evaluation: EvalResult | None = None,
    ) -> None:
        """Append simple post-debate memories from a completed debate."""

        if self.mode != "read_update":
            return
        if result.verdict is None:
            raise ValueError("Cannot update memory without a verdict.")

        reflected = self._reflect_memory(case, result, evaluation)
        if reflected:
            self._extend_from_reflection(case, reflected)
        else:
            self._append_default_memories(case, result)

        self._deduplicate_and_trim()

    def add_regulations(self, regulations: list[dict[str, Any]]) -> None:
        """Add external legal regulations to the regulation memory bucket."""

        for regulation in regulations:
            entry = {
                **regulation,
                "id": str(regulation.get("id") or regulation.get("article_id") or len(self.regulations)),
                "source_type": regulation.get("source_type", "regulation"),
                "text": str(regulation.get("text") or regulation),
            }
            self.regulations.append(entry)
        self._deduplicate_and_trim()

    def _append_default_memories(self, case: CaseProfile, result: DebateResult) -> None:
        evidence_ids = result.transcript[0].evidence_ids if result.transcript else []
        confidence = result.verdict.confidence if result.verdict else 0.0
        regulation_text = " ".join(
            evidence.text[:500]
            for evidence in result.legal_evidence
            if evidence.metadata.get("source_type") in {"uts_vlc_regulation", "regulation"}
        )
        if regulation_text:
            reg_entry = self._sanitize_entry(
                {
                    "id": f"reg-{case.case_id}-{len(self.regulations)}",
                    "case_id": case.case_id,
                    "source_type": "derived_regulation",
                    "source_evidence_ids": evidence_ids,
                    "confidence": confidence,
                    "text": regulation_text,
                },
                case,
            )
            self.regulations.append(reg_entry)
        exp_entry = self._sanitize_entry(
            {
                "id": f"exp-{case.case_id}-{len(self.experiences)}",
                "case_id": case.case_id,
                "source_type": "debate_experience",
                "rounds": len(result.belief_history),
                "strategy": "proponent/opponent debate with judge belief tracking",
                "confidence": confidence,
                "text": " ".join(turn.public_argument for turn in result.transcript)
                or f"{case.question} {result.verdict.reasoning if result.verdict else ''}",
            },
            case,
        )
        self.experiences.append(exp_entry)
        case_entry = self._sanitize_entry(
            {
                "id": f"case-{case.case_id}-{len(self.cases)}",
                "case_id": case.case_id,
                "source_type": "similar_qa_case",
                "case_type": case.case_type,
                "question": case.question,
                "context_excerpt": case.context[:500],
                "keywords": tokenize(case.question)[:10],
                "text": f"{case.question} {case.context[:500]}",
            },
            case,
        )
        self.cases.append(case_entry)

    def _reflect_memory(
        self,
        case: CaseProfile,
        result: DebateResult,
        evaluation: EvalResult | None,
    ) -> dict[str, Any]:
        if self.reflection_llm is None:
            return {}
        template_path = self.prompt_dir / "memory_reflection.txt"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        prompt = template_path.read_text(encoding="utf-8").format(
            case_profile=case.agent_view(),
            transcript=[turn.model_dump() for turn in result.transcript],
            verdict=result.verdict.model_dump() if result.verdict else {},
            evaluation=evaluation.model_dump() if evaluation else {},
        )
        raw_output = self.reflection_llm.generate(prompt)
        try:
            parsed = json.loads(_extract_json_text(raw_output) or raw_output)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _extend_from_reflection(self, case: CaseProfile, reflected: dict[str, Any]) -> None:
        for bucket_name in ("regulations", "experiences", "cases"):
            bucket = getattr(self, bucket_name)
            entries = reflected.get(bucket_name, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                safe_entry = self._sanitize_entry(entry, case)
                safe_entry.setdefault("id", f"{bucket_name[:3]}-{case.case_id}-{len(bucket)}")
                safe_entry.setdefault("case_id", case.case_id)
                safe_entry.setdefault("source_type", bucket_name[:-1])
                safe_entry.setdefault("text", str(entry))
                bucket.append(safe_entry)

    def _sanitize_entry(self, entry: dict[str, Any], case: CaseProfile) -> dict[str, Any]:
        safe_entry = dict(entry)
        # Never persist hidden gold answer fields in memory used for inference.
        for key in ("gold_answer", "answer", "ground_truth", "label", "prediction"):
            safe_entry.pop(key, None)
        if case.answer:
            for key, value in list(safe_entry.items()):
                if isinstance(value, str) and case.answer in value:
                    safe_entry[key] = value.replace(case.answer, "[REDACTED_ANSWER]")
        return safe_entry

    def _top_entries(
        self,
        entries: list[dict[str, Any]],
        query_text: str,
        top_k: int,
        exclude_case_id: str | None = None,
    ) -> list[dict[str, Any]]:
        entries = [
            entry for entry in entries if str(entry.get("case_id")) != str(exclude_case_id)
        ]
        if self.retrieval == "embedding":
            return self._top_entries_by_embedding(entries, query_text, top_k)
        return self._top_entries_by_lexical(entries, query_text, top_k)

    @staticmethod
    def _top_entries_by_lexical(
        entries: list[dict[str, Any]],
        query_text: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_terms = set(tokenize(query_text))
        if not query_terms:
            return []

        scored = []
        for entry in entries:
            entry_text = str(entry.get("text") or entry)
            entry_terms = set(tokenize(entry_text))
            score = len(query_terms.intersection(entry_terms))
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def _top_entries_by_embedding(
        self,
        entries: list[dict[str, Any]],
        query_text: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not entries:
            return []
        model = self._get_embedding_model()
        query_embedding = model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        texts = [str(entry.get("text") or entry) for entry in entries]
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scored = []
        for entry, embedding in zip(entries, embeddings, strict=True):
            score = sum(
                float(a) * float(b)
                for a, b in zip(query_embedding, embedding, strict=True)
            )
            if not math.isnan(score):
                scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Embedding memory retrieval requires `sentence-transformers`."
                ) from exc
            self._embedding_model = SentenceTransformer(self.embedding_model)
        return self._embedding_model

    def _deduplicate_and_trim(self) -> None:
        self.regulations = self._deduplicate_bucket(self.regulations)
        self.experiences = self._deduplicate_bucket(self.experiences)
        self.cases = self._deduplicate_bucket(self.cases)

    def _deduplicate_bucket(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        deduped: list[dict[str, Any]] = []
        for entry in reversed(entries):
            identity = (
                str(entry.get("case_id")),
                str(entry.get("id")),
                str(entry.get("article_id")),
                str(entry.get("text", ""))[:200],
            )
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(entry)
        deduped.reverse()
        return deduped[-self.max_entries_per_bucket :]


def _extract_json_text(raw_output: str) -> str | None:
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return raw_output[start : end + 1]
