#!/usr/bin/env python3
"""Fine-tune an extractive QA reader on ALQAC/ViLQA training data.

Usage:
    # Fine-tune with default XLM-RoBERTa-base-squad2
    python scripts/train_reader.py

    # Fine-tune with a different base model
    python scripts/train_reader.py --base-model vinai/phobert-base-v2

    # Custom output and hyperparameters
    python scripts/train_reader.py \
        --output-dir checkpoints/my_legal_reader \
        --learning-rate 2e-5 \
        --epochs 5 \
        --batch-size 4

    # Run evaluation with a fine-tuned reader
    python scripts/train_reader.py --eval-only \
        --model-path checkpoints/legal_qa_reader/best_model

The script ONLY uses the train split for training and the validation
split for early stopping. The test split is never touched during
training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import hashlib
import importlib.metadata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune extractive QA reader on Vietnamese legal data.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to ALQAC CSV file.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for splitting.",
    )
    parser.add_argument(
        "--base-model",
        default="deepset/xlm-roberta-base-squad2",
        help="HuggingFace base model (pretrained QA).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/legal_qa_reader"),
        help="Output directory for checkpoints.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only evaluate a saved model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to fine-tuned model directory for --eval-only.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="Max validation cases for evaluation (0=all).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers (0=main process).",
    )
    return parser


def run_training(args: argparse.Namespace) -> Path:
    """Execute the fine-tuning pipeline."""
    from src.config import ExperimentConfig, SplitManifest, sha256_file
    from src.data_loader import load_vilqa_csv, split_cases_from_manifest
    from src.reader.finetune_reader import (
        ReaderConfig,
        check_reader_training_dependencies,
        finetune_reader,
    )

    dep_versions = check_reader_training_dependencies()
    logger.info(
        "Verified training stack: torch=%s transformers=%s accelerate=%s",
        dep_versions["torch"],
        dep_versions["transformers"],
        dep_versions["accelerate"],
    )

    experiment = ExperimentConfig.from_yaml(args.config)
    dataset = args.dataset or experiment.dataset.path
    seed = args.seed if args.seed is not None else experiment.seed
    manifest = SplitManifest.load(args.split_manifest or experiment.dataset.split_manifest)
    cases = load_vilqa_csv(dataset)
    split = split_cases_from_manifest(cases, manifest, dataset)

    config = ReaderConfig(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        output_dir=str(args.output_dir),
        seed=seed,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
    )

    logger.info(
        "Fine-tuning reader: base=%s, train=%d, val=%d, epochs=%d, lr=%s",
        args.base_model,
        len(split.train),
        len(split.validation),
        args.epochs,
        args.learning_rate,
    )
    best_model_dir = finetune_reader(
        train_cases=split.train,
        validation_cases=split.validation,
        config=config,
    )
    logger.info("Fine-tuned reader saved to: %s", best_model_dir)
    checkpoint = Path(best_model_dir)
    checkpoint_hash = hashlib.sha256("".join(sorted(p.name + str(p.stat().st_size) for p in checkpoint.rglob("*") if p.is_file())).encode()).hexdigest()
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "accelerate") if importlib.util.find_spec(name)}
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({"base_model": args.base_model, "base_model_revision": None, "tokenizer_revision": None, "dataset_sha256": sha256_file(dataset), "train_validation_manifest_hash": manifest.sha256, "train_case_ids": manifest.train_case_ids, "validation_case_ids": manifest.validation_case_ids, "hyperparameters": config.__dict__, "epochs": args.epochs, "seed": seed, "package_versions": versions, "checkpoint_hash": checkpoint_hash}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return best_model_dir


def run_evaluation(args: argparse.Namespace) -> None:
    """Evaluate a fine-tuned reader on validation split."""
    from src.config import ExperimentConfig, SplitManifest
    from src.data_loader import load_vilqa_csv, split_cases_from_manifest
    from src.evaluation.evaluator import ViLQAEvaluator
    from src.reader.finetune_reader import load_finetuned_reader

    experiment = ExperimentConfig.from_yaml(args.config); dataset = args.dataset or experiment.dataset.path
    manifest = SplitManifest.load(args.split_manifest or experiment.dataset.split_manifest)
    split = split_cases_from_manifest(load_vilqa_csv(dataset), manifest, dataset)

    model_path = args.model_path or Path("checkpoints/legal_qa_reader/best_model")
    checkpoint_manifest = model_path / "checkpoint_manifest.json"
    if not checkpoint_manifest.exists():
        raise FileNotFoundError(f"Reader reproducibility blocker: missing {checkpoint_manifest}")
    provenance = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    if set(provenance.get("train_case_ids", [])) & set(manifest.validation_case_ids + manifest.test_case_ids):
        raise ValueError("Reader checkpoint training split overlaps current validation/test manifest.")
    reader = load_finetuned_reader(model_path)
    evaluator = ViLQAEvaluator()

    val_cases = split.validation
    if args.eval_limit > 0:
        val_cases = val_cases[: args.eval_limit]

    em_scores = []
    f1_scores = []
    for case in val_cases:
        result = reader.predict(question=case.question, context=case.context)
        eval_result = evaluator.evaluate_answer(case, result.answer)
        if eval_result.exact_match is not None:
            em_scores.append(eval_result.exact_match)
        if eval_result.f1 is not None:
            f1_scores.append(eval_result.f1)

    em_mean = sum(em_scores) / len(em_scores) if em_scores else 0.0
    f1_mean = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    print(f"\nFine-tuned Reader Evaluation on {len(val_cases)} validation cases:")
    print(f"  Model: {model_path}")
    print(f"  Exact Match: {em_mean:.4f}")
    print(f"  Token F1:    {f1_mean:.4f}")

    results_path = Path(model_path) / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "split": "validation",
                "num_cases": len(val_cases),
                "exact_match": em_mean,
                "f1": f1_mean,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Results saved to: {results_path}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()

    if not args.eval_only:
        from src.reader.finetune_reader import (

            READER_TRAINING_INSTALL_HINT,
            check_reader_training_dependencies,
        )

        try:
            versions = check_reader_training_dependencies()
        except ImportError as exc:
            print("ERROR: Reader training dependencies are missing or too old.", file=sys.stderr)
            print(exc, file=sys.stderr)
            print(f"\nRun:\n  {READER_TRAINING_INSTALL_HINT}", file=sys.stderr)
            print("\nThen verify:\n  python scripts/verify_reader_deps.py", file=sys.stderr)
            raise SystemExit(1) from exc

        logger.info(
            "Training dependencies OK: torch=%s transformers=%s accelerate=%s sentencepiece=%s",
            versions["torch"],
            versions["transformers"],
            versions["accelerate"],
            versions["sentencepiece"],
        )

    if args.eval_only:
        run_evaluation(args)
    else:
        best_model_dir = run_training(args)
        print(f"\nTraining complete. Best model: {best_model_dir}")
        print("To evaluate, run:")
        print(f"  python scripts/train_reader.py --eval-only --model-path {best_model_dir}")


if __name__ == "__main__":
    main()
