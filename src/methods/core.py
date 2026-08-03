"""Comparable Phase-1 method interface and shared LLM instrumentation."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Protocol
from pydantic import BaseModel, Field
from src.llm import LLMClient
from src.models import CaseProfile
from src.config import MethodConfig
from src.utils.answer_postprocess import shorten_legal_answer

class ContextBundle(BaseModel):
    primary_context: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    visible_context_chars: int
    visible_context_tokens: int
    source_doc_id: str = "primary_context"

    @classmethod
    def from_case(cls, case: CaseProfile) -> "ContextBundle":
        return cls(primary_context=case.context, visible_context_chars=len(case.context), visible_context_tokens=max(1, len(case.context.split())), source_doc_id=case.case_id)

class MethodPrediction(BaseModel):
    case_id: str; method: str; raw_output: str; raw_answer: str; normalized_answer: str
    source_doc_id: str | None = None; start_offset: int | None = None; end_offset: int | None = None
    confidence: float | None = None; reasoning: str | None = None; cited_evidence_ids: list[str] = Field(default_factory=list)
    llm_calls: int = 0; input_tokens: int | None = None; output_tokens: int | None = None; latency_ms: float = 0
    parse_retries: int = 0; fallback_count: int = 0; metadata: dict[str, Any] = Field(default_factory=dict)

class MethodRunner(Protocol):
    def predict(self, case: CaseProfile, context_bundle: ContextBundle, config: MethodConfig) -> MethodPrediction: ...

def _contract() -> str: return Path("configs/prompts/shared/answer_contract.txt").read_text(encoding="utf-8")
def _answer(raw: str, context: str) -> tuple[str, str | None]:
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}")+1]); return str(data.get("answer", "")).strip(), str(data.get("reasoning", "")).strip() or None
    except Exception: return raw.strip(), None

def run_llm_method(method: str, case: CaseProfile, bundle: ContextBundle, llm: LLMClient, rounds: int = 1) -> MethodPrediction:
    """Uniform context/contract; self-debate is deliberately one call."""
    stages = 1 if method == "self_debate_single_call" else (rounds * 4 + 1 if method == "unstructured_multi_agent" else 1)
    outputs: list[str] = []; start = time.perf_counter()
    for stage in range(stages):
        prompt = "{contract}\nMETHOD: {method}\nSTAGE: {stage}/{stages}\nQUESTION: {question}\nCONTEXT: {context}\nReturn JSON.".format(contract=_contract(), method=method, stage=stage + 1, stages=stages, question=case.question, context=bundle.primary_context)
        outputs.append(llm.generate(prompt))
    raw_output = outputs[-1]; raw_answer, reasoning = _answer(raw_output, bundle.primary_context)
    normalized = shorten_legal_answer(raw_answer, bundle.primary_context, case.question)
    offset = bundle.primary_context.find(normalized) if normalized else -1
    return MethodPrediction(case_id=case.case_id, method=method, raw_output=raw_output, raw_answer=raw_answer, normalized_answer=normalized, source_doc_id=bundle.source_doc_id if offset >= 0 else None, start_offset=offset if offset >= 0 else None, end_offset=offset + len(normalized) if offset >= 0 else None, reasoning=reasoning, llm_calls=stages, input_tokens=sum(max(1, len((bundle.primary_context + case.question).split())) for _ in outputs), output_tokens=sum(max(1, len(item.split())) for item in outputs), latency_ms=(time.perf_counter()-start)*1000, metadata={"visible_context_chars": bundle.visible_context_chars, "visible_context_tokens": bundle.visible_context_tokens, "answer_contract": True, "coverage_gold_span_visible": (case.answer or "") in bundle.primary_context})
