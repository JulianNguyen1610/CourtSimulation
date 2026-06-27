"""Baseline methods for ViLQA legal QA experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.llm import LLMClient, extract_candidate_from_context
from src.models import CaseProfile

_EXTRACTIVE_QA_PIPELINES: dict[str, Any] = {}
_FINETUNED_READERS: dict[str, Any] = {}


def direct_context_candidate(case: CaseProfile) -> str:
    """Return a simple context-derived answer candidate.

    This baseline is intentionally weak and deterministic. It provides a
    sanity-check floor before adding real extractive QA models.
    """

    return extract_candidate_from_context(case)


def direct_llm_prediction(
    case: CaseProfile,
    llm: LLMClient,
    prompt_dir: str | Path = "configs/prompts",
) -> str:
    """Single LLM direct answer baseline."""

    prompt = _render_prompt(
        prompt_dir,
        "direct_prediction.txt",
        case_profile=case.agent_view(),
    )
    return _parse_answer_text(llm.generate(prompt), fallback=direct_context_candidate(case))


def cot_llm_prediction(
    case: CaseProfile,
    llm: LLMClient,
    prompt_dir: str | Path = "configs/prompts",
) -> str:
    """Single LLM chain-of-thought-style baseline with final JSON answer."""

    prompt = _render_prompt(
        prompt_dir,
        "cot_prediction.txt",
        case_profile=case.agent_view(),
    )
    return _parse_answer_text(llm.generate(prompt), fallback=direct_context_candidate(case))


def vanilla_debate_prediction(
    case: CaseProfile,
    llm: LLMClient,
    rounds: int = 3,
    prompt_dir: str | Path = "configs/prompts",
) -> str:
    """Unstructured two-agent debate baseline without a judge agent."""

    prompt = _render_prompt(
        prompt_dir,
        "vanilla_debate.txt",
        case_profile=case.agent_view(),
        rounds=rounds,
    )
    return _parse_answer_text(llm.generate(prompt), fallback=direct_context_candidate(case))


def extractive_qa_prediction(
    case: CaseProfile,
    model_name: str = "deepset/xlm-roberta-base-squad2",
) -> str:
    """Extractive QA reader baseline using a Hugging Face QA pipeline.

    The import is lazy so offline/unit-test runs do not require transformer
    dependencies unless this baseline is selected. The underlying pipeline is
    cached per ``model_name`` for the process lifetime so batch runs do not
    reload weights on every case.
    """

    reader = _get_extractive_qa_pipeline(model_name)
    output = reader(question=case.question, context=case.context)
    answer = output.get("answer") if isinstance(output, dict) else None
    if not answer:
        raise RuntimeError(f"Extractive QA model returned no answer for {case.case_id}.")
    return str(answer).strip()


def _get_extractive_qa_pipeline(model_name: str) -> Any:
    """Return a cached Hugging Face question-answering pipeline."""

    cached = _EXTRACTIVE_QA_PIPELINES.get(model_name)
    if cached is not None:
        return cached

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Extractive QA baseline requires `transformers` and a supported "
            "PyTorch/TensorFlow backend."
        ) from exc

    reader = pipeline("question-answering", model=model_name, tokenizer=model_name)
    _EXTRACTIVE_QA_PIPELINES[model_name] = reader
    return reader


def clear_extractive_qa_pipeline_cache() -> None:
    """Clear cached extractive QA pipelines (mainly for tests)."""

    _EXTRACTIVE_QA_PIPELINES.clear()


def finetuned_reader_prediction(
    case: CaseProfile,
    model_path: str | Path = "checkpoints/legal_qa_reader/best_model",
    *,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    max_answer_length: int = 50,
) -> str:
    """Fine-tuned extractive QA reader baseline.

    Uses a reader model fine-tuned on the ALQAC train split, which
    should outperform the generic SQuAD2-pretrained model on Vietnamese
    legal text patterns.
    """
    reader = _get_finetuned_reader(
        model_path,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_answer_length=max_answer_length,
    )
    result = reader.predict(question=case.question, context=case.context)
    answer = result.answer
    if not answer:
        raise RuntimeError(
            f"Fine-tuned reader returned no answer for {case.case_id}."
        )
    return answer.strip()


def tuned_bm25_reader_prediction(
    case: CaseProfile,
    retrieved_contexts: list[str],
    model_path: str | Path = "checkpoints/legal_qa_reader/best_model",
    *,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    max_answer_length: int = 50,
    top_k_answers: int = 5,
) -> str:
    """Tuned BM25 + fine-tuned reader baseline.

    Retrieves top-k evidence via tuned BM25, then runs the fine-tuned
    reader across original context and all retrieved passages. Returns
    the highest-confidence answer span.
    """
    reader = _get_finetuned_reader(
        model_path,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_answer_length=max_answer_length,
    )
    result = reader.predict_with_retrieved_context(
        question=case.question,
        context=case.context,
        retrieved_contexts=retrieved_contexts,
        top_k_answers=top_k_answers,
    )
    answer = result.answer
    if not answer:
        # Fallback to direct context only
        result = reader.predict(question=case.question, context=case.context)
        answer = result.answer
    if not answer:
        raise RuntimeError(
            f"Tuned BM25+reader returned no answer for {case.case_id}."
        )
    return answer.strip()


def _get_finetuned_reader(
    model_path: str | Path,
    *,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    max_answer_length: int = 50,
) -> Any:
    """Return a cached LegalQAReader instance for the given model path."""
    from src.reader.finetune_reader import LegalQAReader

    key = str(model_path)
    cached = _FINETUNED_READERS.get(key)
    if cached is not None:
        return cached

    reader = LegalQAReader(
        model_path=model_path,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_answer_length=max_answer_length,
    )
    _FINETUNED_READERS[key] = reader
    return reader


def clear_finetuned_reader_cache() -> None:
    """Clear cached fine-tuned reader instances (mainly for tests)."""

    _FINETUNED_READERS.clear()


def bm25_reader_prediction(
    case: CaseProfile,
    retrieved_contexts: list[str],
    model_name: str = "deepset/xlm-roberta-base-squad2",
) -> str:
    """BM25 + reader baseline over retrieved top-k context chunks."""

    joined_context = "\n\n".join(retrieved_contexts).strip() or case.context
    retrieved_case = case.model_copy(update={"context": joined_context})
    return extractive_qa_prediction(retrieved_case, model_name=model_name)


def _render_prompt(
    prompt_dir: str | Path,
    template_name: str,
    **values: object,
) -> str:
    template_path = Path(prompt_dir) / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8").format(**values)


def _parse_answer_text(raw_output: str, fallback: str) -> str:
    data = _loads_json_or_empty(raw_output)
    for key in ("answer", "prediction", "final_answer", "consensus_answer"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    recovered = _recover_json_field(raw_output)
    if recovered:
        return recovered

    labeled_match = re.search(
        r"(?:answer|final answer|final_answer|consensus|đáp án)\s*[:：]\s*(.+)",
        raw_output,
        flags=re.IGNORECASE,
    )
    if labeled_match:
        return labeled_match.group(1).strip().strip('"`')
    cleaned = raw_output.strip().strip('"`')
    return cleaned or fallback


def _recover_json_field(raw_output: str) -> str | None:
    """Recover an answer value from JSON that was truncated by token limits.

    Reasoning models can emit a long ``reasoning`` field that exceeds the output
    token budget, leaving JSON unterminated so ``json.loads`` fails. We still try
    to recover the ``answer``/``prediction`` value via a tolerant regex.
    """

    for key in ("answer", "prediction", "final_answer", "consensus_answer"):
        match = re.search(
            rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"',
            raw_output,
            flags=re.IGNORECASE,
        )
        if match and match.group(1).strip():
            return match.group(1).strip()
    return None


def _loads_json_or_empty(raw_output: str) -> dict[str, Any]:
    json_text = _extract_json_text(raw_output)
    if not json_text:
        return {}
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_json_text(raw_output: str) -> str | None:
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        raw_output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1)
    block_match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if block_match:
        return block_match.group(0)
    stripped = raw_output.strip()
    return stripped if stripped.startswith("{") and stripped.endswith("}") else None
