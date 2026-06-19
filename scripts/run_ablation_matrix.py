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
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class AblationVariant:
    name: str
    retrieval: str
    memory: str
    rounds: int
    judge: str
    roles: str

    @property
    def method(self) -> str:
        return "vanilla" if self.judge == "off" else "debate"


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
    parser.add_argument("--summary-csv", default="docs/experiments/p1_ablation_summary.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    variants = make_variants(include_heavy_rerank=args.include_heavy_rerank)
    command_rows = []
    summary_rows = []
    for variant in variants:
        command = build_command(args, output_root, variant)
        command_rows.append({"variant": variant.name, "command": " ".join(command)})
        if args.execute:
            subprocess.run(command, check=True)
            summary_rows.extend(read_latest_metrics(output_root, variant))

    commands_path = output_root / "commands.csv"
    write_rows(commands_path, command_rows, fieldnames=["variant", "command"])
    if summary_rows:
        write_rows(Path(args.summary_csv), summary_rows, fieldnames=list(summary_rows[0]))
    print(f"Wrote ablation commands: {commands_path}")
    if args.execute:
        print(f"Wrote metrics summary: {args.summary_csv}")


def make_variants(include_heavy_rerank: bool) -> list[AblationVariant]:
    retrieval_values = ["off", "bm25_only"]
    if include_heavy_rerank:
        retrieval_values.append("bm25_rerank")

    variants = [
        AblationVariant("full_3r_bm25_memory_judge", "bm25_only", "read_only", 3, "on", "proponent-opponent"),
        AblationVariant("retrieval_off", "off", "read_only", 3, "on", "proponent-opponent"),
        AblationVariant("memory_off", "bm25_only", "off", 3, "on", "proponent-opponent"),
        AblationVariant("memory_update_on", "bm25_only", "read_update", 3, "on", "proponent-opponent"),
        AblationVariant("rounds_1", "bm25_only", "read_only", 1, "on", "proponent-opponent"),
        AblationVariant("rounds_5", "bm25_only", "read_only", 5, "on", "proponent-opponent"),
        AblationVariant("judge_off_vanilla", "bm25_only", "read_only", 3, "off", "proponent-opponent"),
    ]
    if "bm25_rerank" in retrieval_values:
        variants.append(
            AblationVariant("bm25_plus_rerank", "bm25_rerank", "read_only", 3, "on", "proponent-opponent")
        )
    return variants


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
        command.append("--update-memory")
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
