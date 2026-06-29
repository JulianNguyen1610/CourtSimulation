"""Compare fixed vs judge-mediated Phase 1 debate orchestrators.

Fair ablation (paper secondary config):
  - method: debate
  - rounds: 1
  - retrieval: off
  - memory: read_only
  - closing: on
  - split: validation (53 cases when --limit 0)

Default mode is dry-run. Use --execute to call the LLM backend.

Examples:
  # Print commands (no API)
  python scripts/run_orchestrator_ablation.py

  # Mock smoke on 2 cases
  python scripts/run_orchestrator_ablation.py --llm mock --limit 2 --execute

  # Server (Ollama qwen3.5:9b, full validation)
  export LOCAL_LLM_REASONING_EFFORT=none
  python scripts/run_orchestrator_ablation.py --config configs/ollama.yaml \\
      --llm local --split validation --limit 0 --execute --continue-on-error
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

SCRIPT_VERSION = "2026-06-29-orchestrator"

OLLAMA_DEFAULTS = {
    "local_model": "qwen3.5:9b",
    "local_endpoint": "http://localhost:11434/v1/chat/completions",
    "local_timeout": 1200.0,
}

REFERENCE_EM = 0.6038
REFERENCE_F1 = 0.8412
REFERENCE_NOTE = (
    "structured debate fixed orchestrator, retrieval=off, memory=read_only "
    "(rerun 2026-06-29)"
)


@dataclass(frozen=True)
class OrchestratorVariant:
    name: str
    orchestrator: str
    expected_method_key: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablation: fixed vs judge-mediated debate orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max cases; 0 = full split.",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--retrieval-method", default="off")
    parser.add_argument("--memory-mode", default="read_only")
    parser.add_argument("--output-root", default="outputs/orchestrator_ablation")
    parser.add_argument("--local-model", default=None)
    parser.add_argument("--local-endpoint", default=None)
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--local-timeout", type=float, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run generated commands (otherwise dry-run only).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running if one variant subprocess fails.",
    )
    parser.add_argument(
        "--summary-csv",
        default="docs/experiments/orchestrator_ablation_results.csv",
        help="CSV path for metrics (appended when --execute).",
    )
    parser.add_argument(
        "--variants",
        default="fixed,judge_mediated",
        help="Comma-separated orchestrator modes to run.",
    )
    return parser


def make_variants(selected: set[str]) -> list[OrchestratorVariant]:
    all_variants = [
        OrchestratorVariant(
            name="orchestrator_fixed",
            orchestrator="fixed",
            expected_method_key="debate",
        ),
        OrchestratorVariant(
            name="orchestrator_judge_mediated",
            orchestrator="judge_mediated",
            expected_method_key="debate_judge_mediated",
        ),
    ]
    return [variant for variant in all_variants if variant.orchestrator in selected]


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
    variant: OrchestratorVariant,
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
        "debate",
        "--limit",
        str(args.limit),
        "--rounds",
        str(args.rounds),
        "--retrieval-method",
        args.retrieval_method,
        "--memory-mode",
        args.memory_mode,
        "--orchestrator",
        variant.orchestrator,
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
    return command


def read_latest_metrics(
    output_root: Path,
    variant: OrchestratorVariant,
    *,
    run_id: str,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    run_dirs = sorted((output_root / variant.name).glob("*"))
    if not run_dirs:
        return []
    metrics_path = run_dirs[-1] / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    method_metrics = metrics.get("metrics_by_method", {}).get(
        variant.expected_method_key, {}
    )
    if not method_metrics:
        return []
    ablation = metrics.get("ablation_settings", {})
    fallbacks = metrics.get("fallbacks", {})
    return [
        {
            "run_id": run_id,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "variant": variant.name,
            "orchestrator": variant.orchestrator,
            "method": variant.expected_method_key,
            "split": args.split,
            "n_cases": str(metrics.get("num_cases", "")),
            "rounds": str(args.rounds),
            "retrieval": args.retrieval_method,
            "memory": args.memory_mode,
            "llm": args.llm,
            "exact_match": str(method_metrics.get("exact_match", "")),
            "f1": str(method_metrics.get("f1", "")),
            "fallback_rate": str(fallbacks.get("fallback_rate", "")),
            "delta_em_vs_fixed_ref": "",
            "metrics_path": str(metrics_path),
            "notes": "",
        }
    ]


def append_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_commands(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["variant", "orchestrator", "command"])
        writer.writeheader()
        writer.writerows(rows)


def annotate_delta_em(rows: list[dict[str, str]]) -> None:
    for row in rows:
        try:
            em = float(row["exact_match"])
        except (TypeError, ValueError):
            continue
        row["delta_em_vs_fixed_ref"] = f"{em - REFERENCE_EM:+.4f}"


def main() -> int:
    args = build_parser().parse_args()
    resolve_local_defaults(args)

    selected = {item.strip() for item in args.variants.split(",") if item.strip()}
    variants = make_variants(selected)
    if not variants:
        print("No variants selected. Use --variants fixed,judge_mediated")
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    command_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    failed: list[str] = []

    print(f"run_orchestrator_ablation {SCRIPT_VERSION}")
    print(f"  variants={len(variants)} execute={args.execute} split={args.split} limit={args.limit}")
    print(f"  reference fixed EM={REFERENCE_EM:.4f} ({REFERENCE_NOTE})")
    print("")

    for variant in variants:
        command = build_command(args, output_root, variant)
        command_rows.append(
            {
                "variant": variant.name,
                "orchestrator": variant.orchestrator,
                "command": " ".join(command),
            }
        )
        print(f"[{variant.name}] {' '.join(command)}")
        if not args.execute:
            continue
        try:
            subprocess.run(command, check=True)
            summary_rows.extend(
                read_latest_metrics(output_root, variant, run_id=run_id, args=args)
            )
        except subprocess.CalledProcessError as exc:
            failed.append(variant.name)
            print(f"Variant {variant.name} failed: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    commands_path = output_root / "commands.csv"
    write_commands(commands_path, command_rows)
    print(f"\nWrote commands: {commands_path}")

    if summary_rows:
        annotate_delta_em(summary_rows)
        fieldnames = list(summary_rows[0].keys())
        append_rows(Path(args.summary_csv), summary_rows, fieldnames=fieldnames)
        print(f"Appended metrics: {args.summary_csv}")
        print("\n| Variant | EM | F1 | ΔEM vs ref |")
        print("|---|---:|---:|---:|")
        for row in summary_rows:
            print(
                f"| {row['variant']} | {row['exact_match']} | {row['f1']} | "
                f"{row['delta_em_vs_fixed_ref']} |"
            )

    if failed:
        print(f"\nFailed variants: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
