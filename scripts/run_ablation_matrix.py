"""Generate or execute P1 ablation matrix runs.

Default mode is dry-run: it writes commands without calling any LLM APIs.
Use --execute only after the validation config is finalized.

Verify server sync:
    python scripts/run_ablation_matrix.py --help | grep pending-only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "2026-06-26-p1"

OLLAMA_DEFAULTS = {
    "local_model": "qwen3.5:9b",
    "local_endpoint": "http://localhost:11434/v1/chat/completions",
    "local_timeout": 1200.0,
}

PENDING_VARIANTS = frozenset(
    {
        "retrieval_off",
        "retrieval_bm25_rerank",
        "memory_read_only",
        "memory_read_update",
        "closing_off",
        "judge_question_on",
    }
)

VARIANT_GROUPS: dict[str, frozenset[str]] = {
    "retrieval": frozenset(
        {"retrieval_off", "retrieval_bm25", "retrieval_bm25_rerank"}
    ),
    "memory": frozenset(
        {"memory_off", "memory_read_only", "memory_read_update"}
    ),
    "rounds": frozenset({"rounds_3", "rounds_5"}),
    "judge": frozenset({"judge_off_vanilla"}),
    "features": frozenset({"closing_off", "judge_question_on"}),
}


@dataclass(frozen=True)
class AblationVariant:
    name: str
    retrieval: str
    memory: str
    rounds: int
    judge: str
    roles: str
    closing: bool = True
    judge_question: bool = False
    orchestrator: str = "fixed"  # historical P1 ablations used fixed turn order

    @property
    def method(self) -> str:
        return "vanilla" if self.judge == "off" else "debate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controlled P1 ablations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Dry-run pending variants\n"
            "  python scripts/run_ablation_matrix.py --pending-only --include-heavy-rerank\n"
            "\n"
            "  # Server (Ollama qwen3.5:9b, validation 53)\n"
            "  export LOCAL_LLM_REASONING_EFFORT=none\n"
            "  python scripts/run_ablation_matrix.py --config configs/ollama.yaml "
            "--llm local --split validation --limit 0 "
            "--include-heavy-rerank --pending-only --execute --continue-on-error\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--llm",
        default="mock",
        choices=["mock", "gemini", "openai", "local"],
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/p1_ablation_matrix")
    parser.add_argument(
        "--local-model",
        default=None,
        help="Local/Ollama model when --llm local (default: qwen3.5:9b).",
    )
    parser.add_argument(
        "--local-endpoint",
        default=None,
        help="OpenAI-compatible local endpoint (default: http://localhost:11434/v1/chat/completions).",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Gemini model when --llm gemini.",
    )
    parser.add_argument(
        "--local-timeout",
        type=float,
        default=None,
        help="Per-request timeout for --llm local (default: 1200s).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run generated commands (otherwise dry-run only).",
    )
    parser.add_argument(
        "--include-heavy-rerank",
        action="store_true",
        help="Include retrieval_bm25_rerank (loads sentence-transformers / BGE-m3).",
    )
    parser.add_argument(
        "--groups",
        default="retrieval,memory,features",
        help="Comma-separated groups: retrieval, memory, rounds, features, judge.",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Run only variants not yet measured (6 pending ablations).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining variants if one subprocess fails.",
    )
    parser.add_argument(
        "--summary-csv",
        default="docs/experiments/p1_ablation_pending_results.csv",
        help="CSV path for metrics (appended when --execute).",
    )
    return parser


def make_variants(include_heavy_rerank: bool) -> list[AblationVariant]:
    # Reference: structured debate r=1, bm25_only, memory=off (EM=0.4906 val 53)
    variants: list[AblationVariant] = [
        AblationVariant("retrieval_off", "off", "off", 1, "on", "proponent-opponent"),
        AblationVariant("retrieval_bm25", "bm25_only", "off", 1, "on", "proponent-opponent"),
        AblationVariant("memory_off", "bm25_only", "off", 1, "on", "proponent-opponent"),
        AblationVariant(
            "memory_read_only", "bm25_only", "read_only", 1, "on", "proponent-opponent"
        ),
        AblationVariant(
            "memory_read_update", "bm25_only", "read_update", 1, "on", "proponent-opponent"
        ),
        AblationVariant("rounds_3", "bm25_only", "off", 3, "on", "proponent-opponent"),
        AblationVariant("rounds_5", "bm25_only", "off", 5, "on", "proponent-opponent"),
        AblationVariant("judge_off_vanilla", "bm25_only", "off", 1, "off", "proponent-opponent"),
        AblationVariant(
            "closing_off",
            "bm25_only",
            "off",
            1,
            "on",
            "proponent-opponent",
            closing=False,
        ),
        AblationVariant(
            "judge_question_on",
            "bm25_only",
            "off",
            1,
            "on",
            "proponent-opponent",
            judge_question=True,
        ),
    ]
    if include_heavy_rerank:
        variants.insert(
            2,
            AblationVariant(
                "retrieval_bm25_rerank", "bm25_rerank", "off", 1, "on", "proponent-opponent"
            ),
        )
    return variants


def filter_variants(
    variants: list[AblationVariant],
    groups: set[str],
) -> list[AblationVariant]:
    if not groups:
        return variants
    unknown = sorted(group for group in groups if group not in VARIANT_GROUPS)
    if unknown:
        valid = ", ".join(sorted(VARIANT_GROUPS))
        raise SystemExit(f"Unknown --groups value(s): {', '.join(unknown)}. Valid: {valid}")
    allowed = set().union(*(VARIANT_GROUPS[group] for group in groups))
    return [variant for variant in variants if variant.name in allowed]


def resolve_local_defaults(args: argparse.Namespace) -> None:
    if args.llm != "local":
        return
    if args.local_model is None:
        args.local_model = OLLAMA_DEFAULTS["local_model"]
    if args.local_endpoint is None:
        args.local_endpoint = OLLAMA_DEFAULTS["local_endpoint"]
    if args.local_timeout is None:
        args.local_timeout = OLLAMA_DEFAULTS["local_timeout"]


def build_command(
    args: argparse.Namespace,
    output_root: Path,
    variant: AblationVariant,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.main",
        "--config",
        args.config,
        "--run-batch",
        "--llm",
        args.llm,
        "--split",
        args.split,
        "--method",
        variant.method,
        "--limit",
        str(args.limit),
        "--rounds",
        str(variant.rounds),
        "--retrieval-method",
        variant.retrieval,
        "--memory-mode",
        variant.memory,
        "--output-dir",
        str(output_root / variant.name),
    ]
    if args.local_model:
        command += ["--local-model", args.local_model]
    if args.local_endpoint:
        command += ["--local-endpoint", args.local_endpoint]
    if args.gemini_model:
        command += ["--gemini-model", args.gemini_model]
    if args.local_timeout is not None:
        command += ["--local-timeout", str(args.local_timeout)]
    if variant.memory == "read_update":
        command += [
            "--update-memory",
            "--memory-path",
            f"memory-bank/ablation_{variant.name}.json",
        ]
    if not variant.closing:
        command.append("--disable-closing-statements")
    if variant.judge_question:
        command.append("--enable-judge-question")
    if variant.orchestrator != "judge_mediated":
        command += ["--orchestrator", variant.orchestrator]
    return command


def read_latest_metrics(
    output_root: Path,
    variant: AblationVariant,
    *,
    run_id: str,
) -> list[dict[str, str]]:
    run_dirs = sorted((output_root / variant.name).glob("*"))
    if not run_dirs:
        return []
    metrics_path = run_dirs[-1] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for method, values in metrics.get("metrics_by_method", {}).items():
        rows.append(
            {
                "run_id": run_id,
                "variant": variant.name,
                "method": method,
                "retrieval": variant.retrieval,
                "memory": variant.memory,
                "rounds": str(variant.rounds),
                "judge": variant.judge,
                "closing": str(variant.closing),
                "judge_question": str(variant.judge_question),
                "orchestrator": variant.orchestrator,
                "roles": variant.roles,
                "exact_match": str(values.get("exact_match", "")),
                "f1": str(values.get("f1", "")),
                "fallback_rate": str(metrics.get("fallbacks", {}).get("fallback_rate", "")),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    resolve_local_defaults(args)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    groups = {group.strip() for group in args.groups.split(",") if group.strip()}
    variants = filter_variants(
        make_variants(include_heavy_rerank=args.include_heavy_rerank),
        groups,
    )
    if args.pending_only:
        variants = [variant for variant in variants if variant.name in PENDING_VARIANTS]

    if not variants:
        print("No ablation variants selected. Check --groups / --pending-only / --include-heavy-rerank.")
        return 1

    command_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    failed: list[str] = []

    print(f"run_ablation_matrix {SCRIPT_VERSION} | variants={len(variants)} | execute={args.execute}")
    for variant in variants:
        command = build_command(args, output_root, variant)
        command_rows.append({"variant": variant.name, "command": " ".join(command)})
        print(f"  [{variant.name}] {' '.join(command)}")
        if not args.execute:
            continue
        try:
            subprocess.run(command, check=True)
            summary_rows.extend(
                read_latest_metrics(output_root, variant, run_id=run_id)
            )
        except subprocess.CalledProcessError as exc:
            failed.append(variant.name)
            print(f"Variant {variant.name} failed: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    commands_path = output_root / "commands.csv"
    write_rows(commands_path, command_rows, fieldnames=["variant", "command"])
    print(f"Wrote ablation commands: {commands_path}")

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        append_rows(Path(args.summary_csv), summary_rows, fieldnames=fieldnames)
        print(f"Appended metrics summary: {args.summary_csv}")

    if failed:
        print(f"Failed variants ({len(failed)}): {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
