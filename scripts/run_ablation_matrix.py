"""Generate or execute P1 ablation matrix runs.

Default mode is dry-run: it writes commands without calling any LLM APIs.
Use --execute only after the validation config is finalized.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import timezone, datetime
from pathlib import Path


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

    @property
    def method(self) -> str:
        return "vanilla" if self.judge == "off" else "debate"


PENDING_VARIANTS = {
    "retrieval_off",
    "retrieval_bm25_rerank",
    "memory_read_only",
    "memory_read_update",
    "closing_off",
    "judge_question_on",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled P1 ablations.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--llm", default="mock", choices=["mock", "gemini", "openai", "local"])
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/p1_ablation_matrix")
    parser.add_argument("--local-model", default=None, help="Local/Ollama model when --llm local.")
    parser.add_argument("--local-endpoint", default=None, help="OpenAI-compatible local endpoint.")
    parser.add_argument("--gemini-model", default=None, help="Gemini model when --llm gemini.")
    parser.add_argument("--local-timeout", type=float, default=None, help="Per-request timeout for --llm local.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-heavy-rerank", action="store_true")
    parser.add_argument(
        "--groups",
        default="retrieval,memory,features",
        help="Comma-separated variant groups: retrieval, memory, rounds, features, judge.",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Skip reference variants already measured on validation 53.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining variants if one subprocess fails.",
    )
    parser.add_argument(
        "--summary-csv",
        default="docs/experiments/p1_ablation_summary.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    command_rows = []
    summary_rows = []
    for variant in variants:
        command = build_command(args, output_root, variant)
        command_rows.append({"variant": variant.name, "command": " ".join(command)})
        if args.execute:
            try:
                subprocess.run(command, check=True)
                summary_rows.extend(read_latest_metrics(output_root, variant))
            except subprocess.CalledProcessError as exc:
                print(f"Variant {variant.name} failed: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise

    commands_path = output_root / "commands.csv"
    write_rows(commands_path, command_rows, fieldnames=["variant", "command"])
    if summary_rows:
        write_rows(Path(args.summary_csv), summary_rows, fieldnames=list(summary_rows[0]))
    print(f"Wrote ablation commands: {commands_path}")
    if args.execute:
        print(f"Wrote metrics summary: {args.summary_csv}")


VARIANT_GROUPS: dict[str, set[str]] = {
    "retrieval": {
        "retrieval_off",
        "retrieval_bm25",
        "retrieval_bm25_rerank",
    },
    "memory": {
        "memory_off",
        "memory_read_only",
        "memory_read_update",
    },
    "rounds": {"rounds_3", "rounds_5"},
    "judge": {"judge_off_vanilla"},
    "features": {"closing_off", "judge_question_on"},
}


def make_variants(include_heavy_rerank: bool) -> list[AblationVariant]:
    # Reference: structured debate r=1 bm25_only memory=off (matches baseline runs)
    variants = [
        # --- Retrieval ablation (B.2.7 / B.2.8) ---
        AblationVariant("retrieval_off", "off", "off", 1, "on", "proponent-opponent"),
        AblationVariant("retrieval_bm25", "bm25_only", "off", 1, "on", "proponent-opponent"),
        # --- Memory ablation (B.3.7) ---
        AblationVariant("memory_off", "bm25_only", "off", 1, "on", "proponent-opponent"),
        AblationVariant("memory_read_only", "bm25_only", "read_only", 1, "on", "proponent-opponent"),
        AblationVariant("memory_read_update", "bm25_only", "read_update", 1, "on", "proponent-opponent"),
        # --- Rounds ablation (already done, keep for reference) ---
        AblationVariant("rounds_3", "bm25_only", "off", 3, "on", "proponent-opponent"),
        AblationVariant("rounds_5", "bm25_only", "off", 5, "on", "proponent-opponent"),
        # --- Judge ablation (ABL-10) ---
        AblationVariant("judge_off_vanilla", "bm25_only", "off", 1, "off", "proponent-opponent"),
        # --- Closing statement ablation (ABL-11) ---
        AblationVariant("closing_off", "bm25_only", "off", 1, "on", "proponent-opponent", closing=False),
        # --- Judge question ablation (ABL-12) ---
        AblationVariant("judge_question_on", "bm25_only", "off", 1, "on", "proponent-opponent", judge_question=True),
    ]
    if include_heavy_rerank:
        variants.append(
            AblationVariant("retrieval_bm25_rerank", "bm25_rerank", "off", 1, "on", "proponent-opponent")
        )
    return variants


def filter_variants(
    variants: list[AblationVariant],
    groups: set[str],
) -> list[AblationVariant]:
    if not groups:
        return variants
    allowed = set().union(*(VARIANT_GROUPS[group] for group in groups if group in VARIANT_GROUPS))
    return [variant for variant in variants if variant.name in allowed]


def build_command(args: argparse.Namespace, output_root: Path, variant: AblationVariant) -> list[str]:
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
    return command


def read_latest_metrics(output_root: Path, variant: AblationVariant) -> list[dict[str, str]]:
    run_dirs = sorted((output_root / variant.name).glob("*"))
    if not run_dirs:
        return []
    metrics_path = run_dirs[-1] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    for method, values in metrics.get("metrics_by_method", {}).items():
        rows.append(
            {
                "variant": variant.name,
                "method": method,
                "retrieval": variant.retrieval,
                "memory": variant.memory,
                "rounds": str(variant.rounds),
                "judge": variant.judge,
                "closing": str(variant.closing),
                "judge_question": str(variant.judge_question),
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


if __name__ == "__main__":
    main()
