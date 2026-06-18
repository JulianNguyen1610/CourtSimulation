"""Smoke-test entry point for the baseline system."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.agents.debate_agent import DebateAgent
from src.agents.defendant import DefendantAgent
from src.agents.defense import DefenseAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.prosecutor import ProsecutorAgent
from src.artifacts import save_debate_result
from src.courtroom.session import CourtroomSession
from src.data_loader import load_court_case_json, load_vilqa_csv, split_cases
from src.evaluation.ljp_evaluator import LJPEvaluator
from src.experiment_runner import BatchRunConfig, BaselineBatchRunner, select_split
from src.llm import (
    LLMConfig,
    create_role_llm_clients,
    llm_config_from_mapping,
)
from src.memory.memory_store import MemoryStore
from src.orchestrator import DebateOrchestrator
from src.retrieval.legal_retriever import LegalRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load ViLQA/ALQAC data and verify baseline scaffold."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Experiment YAML config with role-specific LLM settings.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/ALQAC.csv"),
        help="Path to ViLQA/ALQAC CSV file.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-courtroom",
        action="store_true",
        help="Run one Phase 3 courtroom LJP session on a structured court case.",
    )
    parser.add_argument(
        "--courtroom-case",
        type=Path,
        default=Path("data/processed/case_01_theft.json"),
        help="Path to structured courtroom case JSON for --run-courtroom.",
    )
    parser.add_argument(
        "--courtroom-config",
        type=Path,
        default=Path("configs/courtroom.yaml"),
        help="Courtroom protocol YAML for --run-courtroom.",
    )
    parser.add_argument(
        "--run-debate",
        action="store_true",
        help="Run one deterministic mock debate after loading the dataset.",
    )
    parser.add_argument(
        "--run-batch",
        action="store_true",
        help="Run a batch baseline experiment and save metrics/predictions.",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="Dataset row index to use for --run-debate.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--evidence-top-k", type=int, default=5)
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=Path("memory-bank/baseline_memory.json"),
        help="Path to JSON memory store.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save debate transcript JSON under the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vilqa_multi_agent_baseline"),
    )
    parser.add_argument(
        "--update-memory",
        action="store_true",
        help="Append post-debate memory entries to the JSON memory store.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Dataset split for --run-batch.",
    )
    parser.add_argument(
        "--method",
        choices=[
            "direct",
            "cot",
            "vanilla",
            "debate",
            "both",
            "all",
            "extractive_qa",
            "bm25_reader",
        ],
        default="both",
        help="Baseline method for --run-batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum cases for --run-batch; use 0 for the full split.",
    )
    parser.add_argument(
        "--save-debate-artifacts",
        action="store_true",
        help="Save per-case debate JSON artifacts in batch mode.",
    )
    parser.add_argument(
        "--llm",
        choices=["mock", "openai", "gemini", "local"],
        default=None,
        help="LLM backend override for LLM-backed runs.",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Gemini model name when --llm gemini is selected.",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Ollama/local model name when --llm local is selected (e.g. qwen3.5:9b).",
    )
    parser.add_argument(
        "--local-endpoint",
        default=None,
        help="OpenAI-compatible local endpoint (default Ollama: http://localhost:11434/v1/chat/completions).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Provider API key override. Prefer env vars such as GEMINI_API_KEY.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--extractive-qa-model",
        default=None,
        help="Hugging Face QA model for extractive_qa and bm25_reader.",
    )
    parser.add_argument(
        "--retrieval-method",
        choices=["off", "bm25_only", "bm25_rerank"],
        default=None,
        help="Retrieval ablation mode.",
    )
    parser.add_argument("--retrieval-rough-top-n", type=int, default=None)
    parser.add_argument("--retrieval-reranker-model", default=None)
    parser.add_argument("--include-uts-vlc", action="store_true")
    parser.add_argument("--uts-vlc-limit", type=int, default=None)
    parser.add_argument(
        "--memory-mode",
        choices=["off", "read_only", "read_update"],
        default=None,
    )
    parser.add_argument(
        "--memory-retrieval",
        choices=["lexical", "embedding"],
        default=None,
    )
    parser.add_argument("--memory-max-entries", type=int, default=None)
    parser.add_argument("--memory-embedding-model", default=None)
    parser.add_argument(
        "--disable-closing-statements",
        action="store_true",
        help="Ablation flag to remove closing statements before verdict.",
    )
    parser.add_argument("--enable-judge-question", action="store_true")
    parser.add_argument("--early-stop-confidence", type=float, default=None)
    parser.add_argument("--enable-llm-evaluator", action="store_true")
    parser.add_argument(
        "--local-timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds for --llm local (default 600).",
    )
    return parser


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML config if present."""

    if not path.exists():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def resolve_role_llm_configs(
    raw_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, LLMConfig]:
    """Resolve role-specific LLM configs from YAML plus CLI/env overrides."""

    llm_section = raw_config.get("llm", {})
    if not isinstance(llm_section, dict):
        llm_section = {}
    default_backend = str(args.llm or llm_section.get("backend") or "mock")
    if default_backend not in ("mock", "openai", "gemini", "local"):
        raise ValueError(f"Unsupported LLM backend: {default_backend}")

    default_mapping = llm_section.get("default", {})
    if not isinstance(default_mapping, dict):
        default_mapping = {}
    provider_defaults = llm_section.get(default_backend, {})
    if isinstance(provider_defaults, dict):
        default_mapping = {**provider_defaults, **default_mapping}

    role_section = llm_section.get("roles", {})
    if not isinstance(role_section, dict):
        role_section = {}

    roles = (
        "direct",
        "cot",
        "vanilla",
        "proponent",
        "opponent",
        "judge",
        "evaluator",
        "prosecutor",
        "defense",
        "defendant",
    )
    resolved: dict[str, LLMConfig] = {}
    for role in roles:
        role_mapping = role_section.get(role, {})
        if not isinstance(role_mapping, dict):
            role_mapping = {}
        mapping = {**default_mapping, **role_mapping}
        mapping["backend"] = args.llm or mapping.get("backend") or default_backend
        if args.gemini_model and mapping["backend"] == "gemini":
            mapping["model"] = args.gemini_model
        if args.local_model and mapping["backend"] == "local":
            mapping["model"] = args.local_model
        if args.local_endpoint and mapping["backend"] == "local":
            mapping["endpoint"] = args.local_endpoint
        if args.temperature is not None:
            mapping["temperature"] = args.temperature
        if args.max_output_tokens is not None:
            mapping["max_output_tokens"] = args.max_output_tokens
        if args.top_p is not None:
            mapping["top_p"] = args.top_p
        if args.local_timeout is not None and mapping["backend"] == "local":
            mapping["timeout"] = args.local_timeout
        config = llm_config_from_mapping(
            mapping,
            default_backend=default_backend,  # type: ignore[arg-type]
            role=role,
        )
        if args.api_key:
            config = replace(config, api_key=args.api_key)
        resolved[role] = config

    fallbacks = {
        "prosecutor": "proponent",
        "defense": "opponent",
        "defendant": "proponent",
    }
    for role, fallback_role in fallbacks.items():
        if role not in role_section and fallback_role in resolved:
            resolved[role] = resolved[fallback_role]
    return resolved


def resolve_extractive_qa_model(
    raw_config: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    baselines = raw_config.get("baselines", {})
    configured_model = None
    if isinstance(baselines, dict):
        reader = baselines.get("extractive_qa", {})
        if isinstance(reader, dict):
            configured_model = reader.get("model")
    return (
        args.extractive_qa_model
        or configured_model
        or "deepset/xlm-roberta-base-squad2"
    )


def get_section(raw_config: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw_config.get(name, {})
    return section if isinstance(section, dict) else {}


def resolve_retrieval_settings(raw_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    retrieval = get_section(raw_config, "retrieval")
    corpus = retrieval.get("legal_corpus", {})
    if not isinstance(corpus, dict):
        corpus = {}
    return {
        "retrieval_method": args.retrieval_method or retrieval.get("method", "bm25_only"),
        "retrieval_rough_top_n": args.retrieval_rough_top_n
        or int(retrieval.get("rough_top_n", 100)),
        "retrieval_reranker_model": args.retrieval_reranker_model
        or retrieval.get("reranker_model", "BAAI/bge-m3"),
        "include_uts_vlc": bool(args.include_uts_vlc or corpus.get("include_uts_vlc", False)),
        "uts_vlc_dataset": corpus.get("dataset", "VietnamAIHub/UTS_VLC"),
        "uts_vlc_split": corpus.get("split", "train"),
        "uts_vlc_limit": args.uts_vlc_limit or corpus.get("limit"),
    }


def resolve_memory_settings(raw_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    memory = get_section(raw_config, "memory")
    return {
        "memory_mode": args.memory_mode or memory.get("mode", "read_only"),
        "memory_retrieval": args.memory_retrieval or memory.get("retrieval", "lexical"),
        "memory_max_entries": args.memory_max_entries
        or int(memory.get("max_entries_per_bucket", 1000)),
        "memory_embedding_model": args.memory_embedding_model
        or memory.get("embedding_model", "intfloat/multilingual-e5-large"),
    }


def resolve_debate_settings(raw_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    debate = get_section(raw_config, "debate")
    limits = debate.get("local_prompt_limits", {})
    if not isinstance(limits, dict):
        limits = {}
    return {
        "include_closing_statements": not args.disable_closing_statements
        and bool(debate.get("include_closing_statements", True)),
        "enable_judge_question": bool(
            args.enable_judge_question or debate.get("enable_judge_question", False)
        ),
        "early_stop_confidence": args.early_stop_confidence
        if args.early_stop_confidence is not None
        else debate.get("early_stop_confidence"),
        "argument_max_tokens": int(debate.get("argument_max_tokens", 500)),
        "max_context_chars": limits.get("max_context_chars"),
        "max_evidence_docs": limits.get("max_evidence_docs"),
        "max_evidence_chars": limits.get("max_evidence_chars"),
        "max_history_turns": limits.get("max_history_turns"),
        "max_history_chars": limits.get("max_history_chars"),
    }


def resolve_evaluation_settings(raw_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    evaluation = get_section(raw_config, "evaluation")
    return {
        "enable_llm_evaluator": bool(
            args.enable_llm_evaluator or evaluation.get("enable_llm_evaluator", False)
        )
    }


def resolve_courtroom_agent_kwargs(
    courtroom_config_path: Path,
    debate_settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    courtroom_raw = load_yaml_config(courtroom_config_path)
    courtroom_section = courtroom_raw.get("courtroom", courtroom_raw)
    agents = courtroom_section.get("agents", {})
    if not isinstance(agents, dict):
        agents = {}
    shared = {
        "argument_max_tokens": int(
            agents.get("argument_max_tokens", debate_settings["argument_max_tokens"])
        ),
        "max_context_chars": agents.get("max_context_chars")
        or debate_settings.get("max_context_chars"),
        "max_evidence_docs": agents.get("max_evidence_docs")
        or debate_settings.get("max_evidence_docs"),
        "max_evidence_chars": agents.get("max_evidence_chars")
        or debate_settings.get("max_evidence_chars"),
        "max_history_turns": agents.get("max_history_turns")
        or debate_settings.get("max_history_turns"),
        "max_history_chars": agents.get("max_history_chars")
        or debate_settings.get("max_history_chars"),
    }
    judge_kwargs = {
        key: shared[key]
        for key in ("max_context_chars", "max_history_turns", "max_history_chars")
        if shared.get(key) is not None
    }
    return shared, judge_kwargs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    raw_config = load_yaml_config(args.config)
    role_llm_configs = resolve_role_llm_configs(raw_config, args)
    extractive_qa_model = resolve_extractive_qa_model(raw_config, args)
    retrieval_settings = resolve_retrieval_settings(raw_config, args)
    memory_settings = resolve_memory_settings(raw_config, args)
    debate_settings = resolve_debate_settings(raw_config, args)
    evaluation_settings = resolve_evaluation_settings(raw_config, args)
    cases = load_vilqa_csv(args.dataset)
    split = split_cases(
        cases,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    if args.run_courtroom:
        court_case = load_court_case_json(args.courtroom_case)
        role_clients = create_role_llm_clients(role_llm_configs)
        agent_kwargs, judge_kwargs = resolve_courtroom_agent_kwargs(
            args.courtroom_config,
            debate_settings,
        )
        judge = JudgeAgent(role_clients["judge"], **judge_kwargs)
        session = CourtroomSession.from_config(
            args.courtroom_config,
            prosecutor=ProsecutorAgent(role_clients["prosecutor"], **agent_kwargs),
            defense=DefenseAgent(role_clients["defense"], **agent_kwargs),
            defendant=DefendantAgent(role_clients["defendant"], **agent_kwargs),
            judge=judge,
            legal_retriever=LegalRetriever.from_cases(split.train),
            memory_store=MemoryStore.load(args.memory_path),
        )
        courtroom_result = session.run(court_case)
        print("\nCourtroom session completed.")
        print(f"Case id: {courtroom_result.case_id}")
        print(f"Phases: {', '.join(courtroom_result.phases_completed)}")
        print(f"Transcript turns: {len(courtroom_result.transcript)}")
        if courtroom_result.legal_judgment is not None:
            print(f"Predicted charge: {courtroom_result.legal_judgment.charge}")
            print(f"Predicted sentence: {courtroom_result.legal_judgment.sentence}")
        if court_case.ground_truth and courtroom_result.legal_judgment is not None:
            evaluation = LJPEvaluator().evaluate(
                courtroom_result.legal_judgment,
                court_case.ground_truth,
                valid_evidence_ids={item.evidence_id for item in court_case.evidence},
            )
            courtroom_result.evaluation = evaluation
            print(f"LJP charge accuracy: {evaluation.charge_accuracy}")
            print(f"LJP article accuracy: {evaluation.article_accuracy}")
        return

    first_case = cases[0]
    print("Baseline scaffold loaded successfully.")
    print(f"Dataset: {args.dataset}")
    print(
        "Split sizes: "
        f"train={len(split.train)}, "
        f"validation={len(split.validation)}, "
        f"test={len(split.test)}"
    )
    print(f"First case id: {first_case.case_id}")
    print(f"First question: {first_case.question}")
    print(f"First answer: {first_case.answer}")

    if args.run_batch:
        memory_store = MemoryStore.load(args.memory_path)
        selected_split = select_split(
            args.split,
            train=split.train,
            validation=split.validation,
            test=split.test,
        )
        runner = BaselineBatchRunner(
            train_cases=split.train,
            memory_store=memory_store,
        )
        run_dir = runner.run(
            selected_split,
            BatchRunConfig(
                split_name=args.split,
                method=args.method,
                limit=args.limit,
                rounds=args.rounds,
                evidence_top_k=args.evidence_top_k,
                memory_top_k=args.memory_top_k,
                output_dir=args.output_dir,
                memory_path=args.memory_path,
                update_memory=args.update_memory,
                save_debate_artifacts=args.save_debate_artifacts,
                llm_backend=role_llm_configs["judge"].backend,
                model=role_llm_configs["judge"].model,
                gemini_model=role_llm_configs["judge"].model or "gemini-2.0-flash",
                api_key=args.api_key,
                temperature=role_llm_configs["judge"].temperature,
                max_output_tokens=role_llm_configs["judge"].max_output_tokens,
                top_p=role_llm_configs["judge"].top_p,
                role_llm_configs=role_llm_configs,
                extractive_qa_model=extractive_qa_model,
                **retrieval_settings,
                **memory_settings,
                **debate_settings,
                **evaluation_settings,
            ),
        )
        print("\nBatch baseline completed.")
        print(f"Run directory: {run_dir}")
        print(f"Metrics: {run_dir / 'metrics.json'}")
        print(f"Predictions: {run_dir / 'predictions.csv'}")

    if args.run_debate:
        if args.case_index < 0 or args.case_index >= len(cases):
            raise IndexError(
                f"case-index {args.case_index} is out of range for "
                f"{len(cases)} loaded cases."
            )

        role_clients = create_role_llm_clients(role_llm_configs)
        legal_retriever = LegalRetriever.from_cases(split.train)
        memory_store = MemoryStore.load(args.memory_path)
        orchestrator = DebateOrchestrator(
            proponent=DebateAgent("proponent", role_clients["proponent"]),
            opponent=DebateAgent("opponent", role_clients["opponent"]),
            judge=JudgeAgent(role_clients["judge"]),
            rounds=args.rounds,
            legal_retriever=legal_retriever,
            memory_store=memory_store,
            evidence_top_k=args.evidence_top_k,
            memory_top_k=args.memory_top_k,
        )
        result = orchestrator.run(cases[args.case_index])
        print("\nDebate completed.")
        print(f"Debate case id: {result.case_id}")
        print(f"Retrieved evidence: {len(result.legal_evidence)}")
        print(
            "Retrieved memory: "
            f"R={len(result.memory_context.regulations)}, "
            f"E={len(result.memory_context.experiences)}, "
            f"C={len(result.memory_context.cases)}"
        )
        print(f"Transcript turns: {len(result.transcript)}")
        print(f"Belief updates: {len(result.belief_history)}")
        if result.verdict is not None:
            print(f"Verdict answer: {result.verdict.answer}")
            print(f"Verdict confidence: {result.verdict.confidence}")

        if args.update_memory:
            memory_store.update_from_debate(cases[args.case_index], result)
            memory_store.save()
            print(f"Updated memory: {args.memory_path}")

        if args.save_result:
            output_path = save_debate_result(
                result,
                cases[args.case_index],
                output_dir=args.output_dir,
            )
            print(f"Saved debate result: {output_path}")


if __name__ == "__main__":
    main()
