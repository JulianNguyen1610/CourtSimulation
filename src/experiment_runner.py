"""Batch experiment runner for the initial ViLQA baseline."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone

# Python 3.10 compatibility: UTC alias
UTC = timezone.utc
from pathlib import Path
from statistics import mean
from typing import Literal

from src.agents.debate_agent import DebateAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.judge_agent import JudgeAgent
from src.artifacts import save_debate_result
from src.baselines import (
    bm25_reader_prediction,
    cot_llm_prediction,
    direct_llm_prediction,
    extractive_qa_prediction,
    finetuned_reader_prediction,
    tuned_bm25_reader_prediction,
    vanilla_debate_prediction,
)
from src.evaluation.evaluator import ViLQAEvaluator
from src.llm import LLMBackend, LLMClient, LLMConfig, create_role_llm_clients
from src.memory.memory_store import MemoryMode, MemoryRetrieval, MemoryStore
from src.models import CaseProfile, DebateResult, EvalResult, PredictionRecord
from src.orchestrator import OrchestratorMode, create_debate_orchestrator
from src.retrieval.legal_retriever import (
    LegalRetriever,
    RetrievalMethod,
    load_uts_vlc_documents,
)
from src.retrieval.reranker import SemanticReranker, SemanticRerankerConfig
from src.utils.answer_postprocess import shorten_legal_answer
from src.methods import ContextBundle, run_llm_method


SplitName = Literal["train", "validation", "test"]
MethodName = Literal[
    "direct",
    "cot",
    "vanilla",
    "debate",
    "both",
    "all",
    "extractive_qa",
    "bm25_reader",
    "finetuned_reader",
    "tuned_bm25_reader",
    "self_debate_single_call", "unstructured_multi_agent",
]


@dataclass(frozen=True)
class BatchRunConfig:
    """Configuration for a batch baseline run."""

    split_name: SplitName
    method: MethodName
    limit: int
    rounds: int
    evidence_top_k: int
    memory_top_k: int
    output_dir: Path
    memory_path: Path
    update_memory: bool = False
    save_debate_artifacts: bool = False
    llm_backend: LLMBackend = "mock"
    model: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    api_key: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 1024
    top_p: float = 0.95
    role_llm_configs: dict[str, LLMConfig] | None = None
    extractive_qa_model: str = "deepset/xlm-roberta-base-squad2"
    retrieval_method: RetrievalMethod = "bm25_only"
    retrieval_rough_top_n: int = 100
    retrieval_reranker_model: str = "BAAI/bge-m3"
    include_uts_vlc: bool = False
    uts_vlc_dataset: str = "VietnamAIHub/UTS_VLC"
    uts_vlc_split: str = "train"
    uts_vlc_limit: int | None = None
    memory_mode: MemoryMode = "read_only"
    memory_retrieval: MemoryRetrieval = "lexical"
    memory_max_entries: int = 1000
    memory_embedding_model: str = "intfloat/multilingual-e5-large"
    include_closing_statements: bool = True
    enable_judge_question: bool = False
    early_stop_confidence: float | None = None
    enable_llm_evaluator: bool = False
    argument_max_tokens: int = 500
    max_context_chars: int | None = None
    max_evidence_docs: int | None = None
    max_evidence_chars: int | None = None
    max_history_turns: int | None = None
    max_history_chars: int | None = None
    finetuned_reader_path: str = "checkpoints/legal_qa_reader/best_model"
    reader_max_seq_length: int = 384
    reader_doc_stride: int = 128
    reader_max_answer_length: int = 50
    orchestrator: OrchestratorMode = "judge_mediated"
    resolved_config: dict[str, object] | None = None
    dataset_sha256: str | None = None
    split_manifest_hash: str | None = None
    memory_snapshot_hash: str | None = None
    contaminated: bool = False
    evaluation_case_ids: list[str] | None = None


def select_split(
    split_name: SplitName,
    train: list[CaseProfile],
    validation: list[CaseProfile],
    test: list[CaseProfile],
) -> list[CaseProfile]:
    """Return the requested split."""

    if split_name == "train":
        return train
    if split_name == "validation":
        return validation
    if split_name == "test":
        return test
    raise ValueError(f"Unsupported split: {split_name}")


class BaselineBatchRunner:
    """Run direct and mock-debate baselines over a dataset split."""

    def __init__(
        self,
        train_cases: list[CaseProfile],
        memory_store: MemoryStore | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.train_cases = train_cases
        self.retriever = LegalRetriever.from_cases(train_cases)
        self.memory_store = memory_store or MemoryStore()
        self.evaluator = ViLQAEvaluator()
        self.llm_client = llm_client
        self._role_clients: dict[str, LLMClient] = {}
        self._fallback_counts: dict[str, int] = {}
        self._parse_attempt_counts: dict[str, int] = {}

    def run(
        self,
        cases: list[CaseProfile],
        config: BatchRunConfig,
    ) -> Path:
        """Run configured baselines and save metrics/predictions."""

        run_dir = self._make_run_dir(config)
        selected_cases = cases[: config.limit] if config.limit > 0 else cases
        records: list[PredictionRecord] = []
        self._role_clients = self._make_role_clients(config)
        self.retriever = self._build_retriever(config)
        self._configure_memory_store(config)
        if config.split_name in {"validation", "test"} and config.update_memory and not config.contaminated:
            raise ValueError("Memory updates are forbidden on validation/test; explicitly mark a research run contaminated.")
        self._fallback_counts = {}
        self._parse_attempt_counts = {}

        methods = self._expand_methods(config.method)
        for case in selected_cases:
            bundle = ContextBundle.from_case(case)
            for method in ("direct", "cot", "self_debate_single_call", "unstructured_multi_agent"):
                if method in methods:
                    role = "vanilla" if method == "self_debate_single_call" else ("judge" if method == "unstructured_multi_agent" else method)
                    records.append(self._run_comparable_llm(method, case, bundle, config, self._client_for(role)))
            if "debate" in methods:
                records.append(self._run_debate(case, config, run_dir))
            if "extractive_qa" in methods and config.llm_backend != "mock":
                records.append(self._run_extractive_qa(case, config))
            if "bm25_reader" in methods and config.llm_backend != "mock":
                records.append(self._run_bm25_reader(case, config))
            if "finetuned_reader" in methods and config.llm_backend != "mock":
                records.append(self._run_finetuned_reader(case, config))
            if "tuned_bm25_reader" in methods and config.llm_backend != "mock":
                records.append(self._run_tuned_bm25_reader(case, config))

        self._save_predictions(records, run_dir / "predictions.csv")
        self._save_metrics(records, run_dir / "metrics.json", config)
        self._save_config(config, run_dir / "config.json", len(selected_cases))
        (run_dir / "run_manifest.json").write_text(json.dumps({"dataset_sha256": config.dataset_sha256, "split_manifest_hash": config.split_manifest_hash, "memory_snapshot_hash": config.memory_snapshot_hash or self.memory_store.snapshot_hash(), "contaminated": config.contaminated, "resolved_config": config.resolved_config or {}}, ensure_ascii=False, indent=2), encoding="utf-8")
        if config.update_memory:
            self.memory_store.save()
        return run_dir

    def _run_direct(
        self,
        case: CaseProfile,
        config: BatchRunConfig,
    ) -> PredictionRecord:
        predicted_answer = shorten_legal_answer(
            direct_llm_prediction(case, self._client_for("direct")),
            case.context,
            case.question,
        )
        evaluation = self.evaluator.evaluate_answer(case, predicted_answer)
        return PredictionRecord(
            case_id=case.case_id,
            split=config.split_name,
            method="direct",
            question=case.question,
            gold_answer=case.answer or "",
            predicted_answer=predicted_answer,
            exact_match=evaluation.exact_match or 0.0,
            f1=evaluation.f1 or 0.0,
        )

    def _run_comparable_llm(self, method: str, case: CaseProfile, bundle: ContextBundle, config: BatchRunConfig, client: LLMClient) -> PredictionRecord:
        prediction = run_llm_method(method, case, bundle, client, config.rounds)
        raw_eval = self.evaluator.evaluate_answer(case, prediction.raw_answer)
        normalized_eval = self.evaluator.evaluate_answer(case, prediction.normalized_answer)
        return PredictionRecord(case_id=case.case_id, split=config.split_name, method=method, question=case.question, gold_answer=case.answer or "", predicted_answer=prediction.normalized_answer, exact_match=normalized_eval.exact_match or 0.0, f1=normalized_eval.f1 or 0.0, raw_prediction=prediction.raw_answer, normalized_prediction=prediction.normalized_answer, llm_calls=prediction.llm_calls, input_tokens=prediction.input_tokens, output_tokens=prediction.output_tokens, latency_ms=prediction.latency_ms, parse_retries=prediction.parse_retries, fallback_count=prediction.fallback_count, visible_context_chars=bundle.visible_context_chars, output_path=json.dumps({"raw_exact_match": raw_eval.exact_match, "raw_f1": raw_eval.f1, "metadata": prediction.metadata}, ensure_ascii=False))

    def _run_cot(self, case: CaseProfile, config: BatchRunConfig) -> PredictionRecord:
        predicted_answer = shorten_legal_answer(
            cot_llm_prediction(case, self._client_for("cot")),
            case.context,
            case.question,
        )
        evaluation = self.evaluator.evaluate_answer(case, predicted_answer)
        return PredictionRecord(
            case_id=case.case_id,
            split=config.split_name,
            method="cot",
            question=case.question,
            gold_answer=case.answer or "",
            predicted_answer=predicted_answer,
            exact_match=evaluation.exact_match or 0.0,
            f1=evaluation.f1 or 0.0,
        )

    def _run_vanilla(
        self,
        case: CaseProfile,
        config: BatchRunConfig,
    ) -> PredictionRecord:
        predicted_answer = shorten_legal_answer(
            vanilla_debate_prediction(
                case,
                self._client_for("vanilla"),
                rounds=config.rounds,
            ),
            case.context,
            case.question,
        )
        evaluation = self.evaluator.evaluate_answer(case, predicted_answer)
        return PredictionRecord(
            case_id=case.case_id,
            split=config.split_name,
            method="vanilla",
            question=case.question,
            gold_answer=case.answer or "",
            predicted_answer=predicted_answer,
            exact_match=evaluation.exact_match or 0.0,
            f1=evaluation.f1 or 0.0,
        )

    def _run_debate(
        self,
        case: CaseProfile,
        config: BatchRunConfig,
        run_dir: Path,
    ) -> PredictionRecord:
        judge = JudgeAgent(
            self._client_for("judge"),
            max_context_chars=config.max_context_chars,
            max_history_turns=config.max_history_turns,
            max_history_chars=config.max_history_chars,
        )
        agent_kwargs = self._debate_agent_kwargs(config)
        orchestrator = create_debate_orchestrator(
            config.orchestrator,
            proponent=DebateAgent("proponent", self._client_for("proponent"), **agent_kwargs),
            opponent=DebateAgent("opponent", self._client_for("opponent"), **agent_kwargs),
            judge=judge,
            rounds=config.rounds,
            legal_retriever=self.retriever,
            memory_store=self.memory_store,
            evidence_top_k=config.evidence_top_k,
            memory_top_k=config.memory_top_k,
            include_closing_statements=config.include_closing_statements,
            enable_judge_question=config.enable_judge_question,
            early_stop_confidence=config.early_stop_confidence,
        )
        result = orchestrator.run(case)
        if result.verdict is None:
            raise ValueError(f"Debate result for {case.case_id} has no verdict.")

        raw_answer = result.verdict.answer or result.verdict.prediction
        predicted_answer = shorten_legal_answer(raw_answer, case.context, case.question)
        result.verdict.answer = predicted_answer
        automated_evaluation = self.evaluator.evaluate_answer(case, predicted_answer)
        llm_evaluation = self._run_llm_evaluator(case, result, config)
        evaluation = self._merge_evaluations(automated_evaluation, llm_evaluation)
        result.evaluation = evaluation

        debate_method = (
            "debate_judge_mediated"
            if config.orchestrator == "judge_mediated"
            else "debate"
        )
        output_path = None
        if config.save_debate_artifacts:
            output_path = save_debate_result(result, case, run_dir / "debates")
        evaluation_path = self._save_case_evaluation(
            case=case,
            method=debate_method,
            automated_evaluation=automated_evaluation,
            llm_evaluation=llm_evaluation,
            path=run_dir / "evaluations" / f"{case.case_id}_{debate_method}.json",
        )

        if config.update_memory:
            self.memory_store.update_from_debate(case, result, evaluation=evaluation)

        self._fallback_counts[debate_method] = (
            self._fallback_counts.get(debate_method, 0) + judge.fallback_count
        )
        self._parse_attempt_counts[debate_method] = (
            self._parse_attempt_counts.get(debate_method, 0) + judge.parse_attempt_count
        )

        return PredictionRecord(
            case_id=case.case_id,
            split=config.split_name,
            method=debate_method,
            question=case.question,
            gold_answer=case.answer or "",
            predicted_answer=predicted_answer,
            exact_match=evaluation.exact_match or 0.0,
            f1=evaluation.f1 or 0.0,
            output_path=str(output_path or evaluation_path),
            raw_prediction=raw_answer,
            normalized_prediction=predicted_answer,
            llm_calls=len(result.transcript) + len(result.belief_history) + 1,
            parse_retries=max(0, judge.parse_attempt_count - 1),
            fallback_count=judge.fallback_count,
            visible_context_chars=len(case.context),
        )

    def _run_extractive_qa(
        self,
        case: CaseProfile,
        config: BatchRunConfig,
    ) -> PredictionRecord:
        predicted_answer = extractive_qa_prediction(
            case,
            model_name=config.extractive_qa_model,
        )
        evaluation = self.evaluator.evaluate_answer(case, predicted_answer)
        return PredictionRecord(
            case_id=case.case_id,
            split=config.split_name,
            method="extractive_qa",
            question=case.question,
            gold_answer=case.answer or "",
            predicted_answer=predicted_answer,
            exact_match=evaluation.exact_match or 0.0,
            f1=evaluation.f1 or 0.0,
        )

    def _run_bm25_reader(
        self,
        case: CaseProfile,
        config: BatchRunConfig,
    ) -> PredictionRecord:
        retrieved = self.retriever.retrieve(
            case.retrieval_query,
            top_k=config.evidence_top_k,
        )
        predicted_answer = bm25_reader_prediction(
            case,
            [document.text for document in retrieved],
            model_name=config.extractive_qa_model,
        )
        evaluation = self.evaluator.evaluate_answer(case, predicted_answer)
        return PredictionRecord(
            case_id=case.case_id,
            split=config.split_name,
            method="bm25_reader",
            question=case.question,
            gold_answer=case.answer or "",
            predicted_answer=predicted_answer,
            exact_match=evaluation.exact_match or 0.0,
            f1=evaluation.f1 or 0.0,
        )

    def _run_finetuned_reader(
        self,
        case: CaseProfile,
        config: BatchRunConfig,
    ) -> PredictionRecord:
        self._validate_reader_checkpoint(config)
        predicted_answer = finetuned_reader_prediction(
            case,
            model_path=config.finetuned_reader_path,
            max_seq_length=config.reader_max_seq_length,
            doc_stride=config.reader_doc_stride,
            max_answer_length=config.reader_max_answer_length,
        )
        evaluation = self.evaluator.evaluate_answer(case, predicted_answer)
        return PredictionRecord(
            case_id=case.case_id,
            split=config.split_name,
            method="finetuned_reader",
            question=case.question,
            gold_answer=case.answer or "",
            predicted_answer=predicted_answer,
            exact_match=evaluation.exact_match or 0.0,
            f1=evaluation.f1 or 0.0,
        )

    def _run_tuned_bm25_reader(
        self,
        case: CaseProfile,
        config: BatchRunConfig,
    ) -> PredictionRecord:
        self._validate_reader_checkpoint(config)
        retrieved = self.retriever.retrieve(
            case.retrieval_query,
            top_k=config.evidence_top_k,
        )
        predicted_answer = tuned_bm25_reader_prediction(
            case,
            [document.text for document in retrieved],
            model_path=config.finetuned_reader_path,
            max_seq_length=config.reader_max_seq_length,
            doc_stride=config.reader_doc_stride,
            max_answer_length=config.reader_max_answer_length,
        )
        evaluation = self.evaluator.evaluate_answer(case, predicted_answer)
        return PredictionRecord(
            case_id=case.case_id,
            split=config.split_name,
            method="tuned_bm25_reader",
            question=case.question,
            gold_answer=case.answer or "",
            predicted_answer=predicted_answer,
            exact_match=evaluation.exact_match or 0.0,
            f1=evaluation.f1 or 0.0,
        )

    @staticmethod
    def _validate_reader_checkpoint(config: BatchRunConfig) -> None:
        manifest_path = Path(config.finetuned_reader_path) / "checkpoint_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Reader reproducibility blocker: missing {manifest_path}")
        provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_ids = set(provenance.get("train_case_ids", []))
        # The run manifest only carries the immutable split hash; training IDs may
        # not overlap an evaluation split supplied by callers that opt in here.
        if not train_ids:
            raise ValueError("Reader checkpoint manifest has no train_case_ids")
        overlap = train_ids.intersection(config.evaluation_case_ids or [])
        if overlap:
            raise ValueError(f"Reader checkpoint training split overlaps evaluation cases: {sorted(overlap)[:10]}")

    @staticmethod
    def _expand_methods(method: MethodName) -> list[str]:
        if method == "both":
            return ["direct", "debate"]
        if method == "all":
            return [
                "direct",
                "cot",
                "self_debate_single_call",
                "unstructured_multi_agent",
                "debate",
                "extractive_qa",
                "bm25_reader",
                "finetuned_reader",
                "tuned_bm25_reader",
            ]
        if method == "vanilla":
            import warnings
            warnings.warn("vanilla is deprecated; use self_debate_single_call", DeprecationWarning, stacklevel=2)
            return ["self_debate_single_call"]
        return [method]

    def _make_role_clients(self, config: BatchRunConfig) -> dict[str, LLMClient]:
        if self.llm_client is not None:
            return {
                role: self.llm_client
                for role in (
                    "direct",
                    "cot",
                    "vanilla",
                    "proponent",
                    "opponent",
                    "judge",
                    "evaluator",
                )
            }
        return create_role_llm_clients(self._role_configs(config))

    def _client_for(self, role: str) -> LLMClient:
        try:
            return self._role_clients[role]
        except KeyError as exc:
            raise KeyError(f"No LLM client configured for role/method: {role}") from exc

    def _role_configs(self, config: BatchRunConfig) -> dict[str, LLMConfig]:
        model = config.model or config.gemini_model
        base_config = LLMConfig(
            backend=config.llm_backend,
            model=model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            top_p=config.top_p,
        )
        if config.role_llm_configs:
            role_configs = dict(config.role_llm_configs)
            for role in (
                "direct",
                "cot",
                "vanilla",
                "proponent",
                "opponent",
                "judge",
                "evaluator",
            ):
                role_configs.setdefault(role, base_config)
            return role_configs

        return {
            "direct": base_config,
            "cot": base_config,
            "vanilla": base_config,
            "proponent": base_config,
            "opponent": base_config,
            "judge": base_config,
            "evaluator": base_config,
        }

    def _debate_agent_kwargs(self, config: BatchRunConfig) -> dict[str, object]:
        return {
            "argument_max_tokens": config.argument_max_tokens,
            "max_context_chars": config.max_context_chars,
            "max_evidence_docs": config.max_evidence_docs,
            "max_evidence_chars": config.max_evidence_chars,
            "max_history_turns": config.max_history_turns,
            "max_history_chars": config.max_history_chars,
        }

    def _build_retriever(self, config: BatchRunConfig) -> LegalRetriever:
        extra_documents = (
            load_uts_vlc_documents(
                dataset_name=config.uts_vlc_dataset,
                split=config.uts_vlc_split,
                limit=config.uts_vlc_limit,
            )
            if config.include_uts_vlc
            else []
        )
        reranker = None
        if config.retrieval_method == "bm25_rerank":
            reranker = SemanticReranker(
                SemanticRerankerConfig(model_name=config.retrieval_reranker_model)
            )
        return LegalRetriever.from_cases(
            self.train_cases,
            extra_documents=extra_documents,
            reranker=reranker,
            method=config.retrieval_method,
            rough_top_n=config.retrieval_rough_top_n,
        )

    def _configure_memory_store(self, config: BatchRunConfig) -> None:
        self.memory_store.mode = config.memory_mode
        self.memory_store.retrieval = config.memory_retrieval
        self.memory_store.max_entries_per_bucket = config.memory_max_entries
        self.memory_store.embedding_model = config.memory_embedding_model
        if config.update_memory and self.memory_store.reflection_llm is None:
            self.memory_store.reflection_llm = self._client_for("evaluator")

    def _run_llm_evaluator(
        self,
        case: CaseProfile,
        result: DebateResult,
        config: BatchRunConfig,
    ) -> EvalResult | None:
        if not config.enable_llm_evaluator or result.verdict is None:
            return None
        evaluator_agent = EvaluatorAgent(self._client_for("evaluator"))
        return evaluator_agent.evaluate(
            case=case,
            transcript=result.transcript,
            verdict=result.verdict,
        )

    @staticmethod
    def _merge_evaluations(
        automated: EvalResult,
        llm_evaluation: EvalResult | None,
    ) -> EvalResult:
        if llm_evaluation is None:
            return automated
        return EvalResult(
            exact_match=automated.exact_match,
            f1=automated.f1,
            legal_accuracy=llm_evaluation.legal_accuracy,
            argument_quality=llm_evaluation.argument_quality,
            logical_consistency=llm_evaluation.logical_consistency,
            notes=(
                f"Automated: {automated.notes or ''} "
                f"LLM evaluator: {llm_evaluation.notes or ''}"
            ).strip(),
        )

    @staticmethod
    def _save_case_evaluation(
        case: CaseProfile,
        method: str,
        automated_evaluation: EvalResult,
        llm_evaluation: EvalResult | None,
        path: Path,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "case": case.agent_view(),
            "method": method,
            "automated_metrics": automated_evaluation.model_dump(),
            "llm_rubric_metrics": (
                llm_evaluation.model_dump() if llm_evaluation else None
            ),
            "human_eval_subset": False,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _make_run_dir(config: BatchRunConfig) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = config.output_dir / f"{timestamp}_{config.split_name}_{config.method}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _save_predictions(records: list[PredictionRecord], path: Path) -> None:
        fieldnames = list(PredictionRecord.model_fields.keys())
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record.model_dump())

    def _save_metrics(
        self,
        records: list[PredictionRecord],
        path: Path,
        config: BatchRunConfig,
    ) -> None:
        grouped: dict[str, list[PredictionRecord]] = {}
        for record in records:
            grouped.setdefault(record.method, []).append(record)

        metrics = {
            "split": config.split_name,
            "method": config.method,
            "num_cases": len({record.case_id for record in records}),
            "num_predictions": len(records),
            "models_by_method": self._models_by_method(config),
            "ablation_settings": {
                "retrieval": config.retrieval_method,
                "memory": config.memory_mode,
                "memory_retrieval": config.memory_retrieval,
                "rounds": config.rounds,
                "judge": "on" if "debate" in self._expand_methods(config.method) else "off",
                "orchestrator": config.orchestrator,
                "roles": "proponent-opponent",
                "closing_statements": config.include_closing_statements,
                "judge_question": config.enable_judge_question,
                "early_stop_confidence": config.early_stop_confidence,
            },
            "fallbacks": self._fallback_payload(
                self._fallback_counts,
                self._parse_attempt_counts,
            ),
            "metrics_by_method": {
                method: {
                    "num_predictions": len(method_records),
                    "exact_match": mean(record.exact_match for record in method_records),
                    "f1": mean(record.f1 for record in method_records),
                }
                for method, method_records in grouped.items()
            },
            "raw_metrics": {
                method: {"exact_match": mean(json.loads(record.output_path).get("raw_exact_match") or 0.0 for record in method_records if record.output_path and record.output_path.startswith("{")), "f1": mean(json.loads(record.output_path).get("raw_f1") or 0.0 for record in method_records if record.output_path and record.output_path.startswith("{"))}
                for method, method_records in grouped.items() if any(record.output_path and record.output_path.startswith("{") for record in method_records)
            },
            "normalized_metrics": {method: {"exact_match": mean(record.exact_match for record in method_records), "f1": mean(record.f1 for record in method_records), "postprocess_changed_rate": mean(float((record.raw_prediction or record.predicted_answer) != record.predicted_answer) for record in method_records if record.raw_prediction is not None)} for method, method_records in grouped.items() if any(record.raw_prediction is not None for record in method_records)},
            "reliability_by_method": {method: {"llm_calls": sum(record.llm_calls or 0 for record in method_records), "parse_retries": sum(record.parse_retries or 0 for record in method_records), "fallback_count": sum(record.fallback_count or 0 for record in method_records), "latency_ms": sum(record.latency_ms or 0 for record in method_records), "input_tokens": sum(record.input_tokens or 0 for record in method_records), "output_tokens": sum(record.output_tokens or 0 for record in method_records)} for method, method_records in grouped.items() if any(record.llm_calls is not None for record in method_records)},
        }
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def _models_by_method(self, config: BatchRunConfig) -> dict[str, object]:
        if self.llm_client is not None:
            injected = {"backend": "injected_client", "model": type(self.llm_client).__name__}
            return {
                "direct": injected,
                "cot": injected,
                "vanilla": injected,
                "debate": {
                    "proponent": injected,
                    "opponent": injected,
                    "judge": injected,
                },
                "extractive_qa": {"model": config.extractive_qa_model},
                "bm25_reader": {
                    "retriever": "lightweight_bm25",
                    "reader": config.extractive_qa_model,
                },
                "finetuned_reader": {"model": config.finetuned_reader_path},
                "tuned_bm25_reader": {
                    "retriever": "lightweight_bm25",
                    "reader": config.finetuned_reader_path,
                },
            }

        role_configs = self._role_configs(config)
        return {
            "direct": self._public_llm_config(role_configs["direct"]),
            "cot": self._public_llm_config(role_configs["cot"]),
            "vanilla": self._public_llm_config(role_configs["vanilla"]),
            "debate": {
                "proponent": self._public_llm_config(role_configs["proponent"]),
                "opponent": self._public_llm_config(role_configs["opponent"]),
                "judge": self._public_llm_config(role_configs["judge"]),
            },
            "extractive_qa": {"model": config.extractive_qa_model},
            "bm25_reader": {
                "retriever": "lightweight_bm25",
                "reader": config.extractive_qa_model,
            },
            "finetuned_reader": {"model": config.finetuned_reader_path},
            "tuned_bm25_reader": {
                "retriever": "lightweight_bm25",
                "reader": config.finetuned_reader_path,
            },
        }

    @staticmethod
    def _public_llm_config(config: LLMConfig) -> dict[str, object]:
        payload: dict[str, object] = {
            "backend": config.backend,
            "model": config.model or "",
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "top_p": config.top_p,
        }
        if config.endpoint:
            payload["endpoint"] = config.endpoint
        return payload

    @staticmethod
    def _fallback_payload(
        fallback_counts: dict[str, int],
        parse_attempt_counts: dict[str, int],
    ) -> dict[str, object]:
        total_fallbacks = sum(fallback_counts.values())
        total_attempts = sum(parse_attempt_counts.values())
        return {
            "total_fallbacks": total_fallbacks,
            "total_parse_attempts": total_attempts,
            "fallback_rate": (
                total_fallbacks / total_attempts if total_attempts else 0.0
            ),
            "by_method": {
                method: {
                    "fallbacks": fallback_counts.get(method, 0),
                    "parse_attempts": attempts,
                    "fallback_rate": (
                        fallback_counts.get(method, 0) / attempts if attempts else 0.0
                    ),
                }
                for method, attempts in parse_attempt_counts.items()
            },
        }

    @staticmethod
    def _save_config(
        config: BatchRunConfig,
        path: Path,
        num_cases: int,
    ) -> None:
        payload = {
            "split_name": config.split_name,
            "method": config.method,
            "limit": config.limit,
            "resolved_num_cases": num_cases,
            "rounds": config.rounds,
            "evidence_top_k": config.evidence_top_k,
            "memory_top_k": config.memory_top_k,
            "output_dir": str(config.output_dir),
            "memory_path": str(config.memory_path),
            "update_memory": config.update_memory,
            "save_debate_artifacts": config.save_debate_artifacts,
            "llm_backend": config.llm_backend,
            "model": config.model,
            "gemini_model": config.gemini_model,
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "top_p": config.top_p,
            "role_llm_configs": {
                role: BaselineBatchRunner._public_llm_config(role_config)
                for role, role_config in (config.role_llm_configs or {}).items()
            },
            "extractive_qa_model": config.extractive_qa_model,
            "retrieval_method": config.retrieval_method,
            "retrieval_rough_top_n": config.retrieval_rough_top_n,
            "retrieval_reranker_model": config.retrieval_reranker_model,
            "include_uts_vlc": config.include_uts_vlc,
            "uts_vlc_dataset": config.uts_vlc_dataset,
            "uts_vlc_split": config.uts_vlc_split,
            "uts_vlc_limit": config.uts_vlc_limit,
            "memory_mode": config.memory_mode,
            "memory_retrieval": config.memory_retrieval,
            "memory_max_entries": config.memory_max_entries,
            "memory_embedding_model": config.memory_embedding_model,
            "include_closing_statements": config.include_closing_statements,
            "enable_judge_question": config.enable_judge_question,
            "early_stop_confidence": config.early_stop_confidence,
            "enable_llm_evaluator": config.enable_llm_evaluator,
            "finetuned_reader_path": config.finetuned_reader_path,
            "reader_max_seq_length": config.reader_max_seq_length,
            "reader_doc_stride": config.reader_doc_stride,
            "reader_max_answer_length": config.reader_max_answer_length,
            "orchestrator": config.orchestrator,
            "dataset_sha256": config.dataset_sha256,
            "split_manifest_hash": config.split_manifest_hash,
            "memory_snapshot_hash": config.memory_snapshot_hash,
            "contaminated": config.contaminated,
            "resolved_config": config.resolved_config or {},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
