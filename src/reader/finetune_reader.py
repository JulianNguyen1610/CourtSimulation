"""Fine-tuned extractive QA reader for Vietnamese legal domain.

This module fine-tunes a multilingual transformer (XLM-RoBERTa, PhoBERT,
or similar) on the ALQAC/ViLQA training split for extractive question
answering. The fine-tuned reader replaces the generic SQuAD2-pretrained
model, adapting it to Vietnamese legal text patterns.

Key design decisions:
- Training data comes ONLY from train split; validation is used for
  early stopping; test is never touched during training.
- Gold answer spans are located in context via character matching;
  unlabeled/impossible cases get ``is_impossible=True`` (SQuAD2-style).
- Output checkpoint is a Hugging Face model directory loadable by
  ``pipeline("question-answering", model=path)``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models import CaseProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReaderConfig:
    """Configuration for fine-tuning the extractive QA reader."""

    base_model: str = "deepset/xlm-roberta-base-squad2"
    max_seq_length: int = 384
    doc_stride: int = 128
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    output_dir: str = "checkpoints/legal_qa_reader"
    seed: int = 42
    fp16: bool = False
    dataloader_num_workers: int = 0


@dataclass
class FinetunedQAResult:
    """Result from the fine-tuned reader prediction."""

    answer: str
    score: float
    start_index: int = -1
    end_index: int = -1


class LegalQADataset:
    """Convert ALQAC CaseProfile list into SQuAD2-format training data.

    The core logic maps (context, question, answer) triples into SQuAD2
    entries, locating the answer span in context. When the gold answer
    is not found verbatim in context, the sample is marked as
    ``is_impossible=True``, enabling SQuAD2-style training where the
    model learns when no answer exists.

    Only train-split cases should be passed here. Validation cases are
    used for early stopping during training; test cases must never be
    used during any training phase.
    """

    def __init__(self, cases: list[CaseProfile], *, split_name: str = "train") -> None:
        if split_name not in ("train", "validation"):
            raise ValueError(
                f"LegalQADataset only accepts train or validation splits, "
                f"got {split_name!r}. Test split must never be used for training."
            )
        self.cases = cases
        self.split_name = split_name

    def to_squad_examples(self) -> list[dict[str, Any]]:
        """Convert case profiles to SQuAD2-compatible training examples."""
        examples: list[dict[str, Any]] = []
        impossible_count = 0

        for case in self.cases:
            answer_text = (case.answer or "").strip()
            is_impossible = False
            start_position = 0

            if answer_text and answer_text in case.context:
                start_position = case.context.index(answer_text)
            elif answer_text:
                is_impossible = True
                impossible_count += 1
            else:
                is_impossible = True
                impossible_count += 1

            example = {
                "id": case.case_id,
                "title": f"_legal_{case.case_id}",
                "context": case.context,
                "question": case.question,
                "answers": (
                    {
                        "text": [answer_text],
                        "answer_start": [start_position],
                    }
                    if not is_impossible
                    else {"text": [], "answer_start": []}
                ),
                "is_impossible": is_impossible,
            }
            examples.append(example)

        logger.info(
            "Converted %d cases to SQuAD2 format (%d possible, %d impossible) "
            "for split=%s",
            len(self.cases),
            len(self.cases) - impossible_count,
            impossible_count,
            self.split_name,
        )
        return examples

    def to_squad_dict(self) -> dict[str, Any]:
        """Return a full SQuAD2 data dictionary."""
        examples = self.to_squad_examples()
        paragraph_map: dict[str, list[dict[str, Any]]] = {}
        for example in examples:
            context_key = example["context"]
            paragraph_map.setdefault(context_key, []).append(example)

        paragraphs = []
        for context_text, context_examples in paragraph_map.items():
            qas = []
            for example in context_examples:
                qas.append(
                    {
                        "id": example["id"],
                        "question": example["question"],
                        "answers": example["answers"],
                        "is_impossible": example["is_impossible"],
                    }
                )
            paragraphs.append({"context": context_text, "qas": qas})

        return {
            "version": "v2.0",
            "data": [
                {
                    "title": "ViLQA_ALQAC_Legal",
                    "paragraphs": paragraphs,
                }
            ],
        }


class LegalQAReader:
    """Fine-tuned extractive QA reader for Vietnamese legal text.

    Wraps a Hugging Face QA pipeline backed by a fine-tuned model.
    The pipeline is lazy-loaded so imports only happen on first use.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_answer_length: int = 50,
        n_best_size: int = 20,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size
        self._pipeline = None

    def predict(self, question: str, context: str) -> FinetunedQAResult:
        """Extract answer from context given a question."""
        pipeline = self._get_pipeline()
        output = pipeline(
            question=question,
            context=context,
            max_seq_len=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_answer_len=self.max_answer_length,
            top_k=self.n_best_size,
        )

        if isinstance(output, list) and output:
            best = output[0]
        elif isinstance(output, dict):
            best = output
        else:
            return FinetunedQAResult(answer="", score=0.0)

        answer = (best.get("answer") or "").strip()
        score = float(best.get("score", 0.0))
        start = int(best.get("start", -1))
        end = int(best.get("end", -1))

        return FinetunedQAResult(
            answer=answer,
            score=score,
            start_index=start,
            end_index=end,
        )

    def predict_with_retrieved_context(
        self,
        question: str,
        context: str,
        retrieved_contexts: list[str],
        top_k_answers: int = 5,
    ) -> FinetunedQAResult:
        """Predict using both original context and retrieved contexts.

        Runs the reader on each context (original + retrieved) and
        returns the highest-confidence answer span.
        """
        all_contexts = [context] + [c for c in retrieved_contexts if c.strip()]
        best_result = FinetunedQAResult(answer="", score=0.0)

        pipeline = self._get_pipeline()

        for ctx in all_contexts:
            if not ctx.strip():
                continue
            try:
                output = pipeline(
                    question=question,
                    context=ctx,
                    max_seq_len=self.max_seq_length,
                    doc_stride=self.doc_stride,
                    max_answer_len=self.max_answer_length,
                    top_k=top_k_answers,
                )
                if isinstance(output, list) and output:
                    for candidate in output:
                        answer = (candidate.get("answer") or "").strip()
                        score = float(candidate.get("score", 0.0))
                        if answer and score > best_result.score:
                            best_result = FinetunedQAResult(
                                answer=answer,
                                score=score,
                                start_index=int(candidate.get("start", -1)),
                                end_index=int(candidate.get("end", -1)),
                            )
                elif isinstance(output, dict):
                    answer = (output.get("answer") or "").strip()
                    score = float(output.get("score", 0.0))
                    if answer and score > best_result.score:
                        best_result = FinetunedQAResult(
                            answer=answer,
                            score=score,
                            start_index=int(output.get("start", -1)),
                            end_index=int(output.get("end", -1)),
                        )
            except Exception:
                logger.debug("Reader failed on context (len=%d), skipping", len(ctx))
                continue

        return best_result

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "LegalQAReader requires `transformers` and a PyTorch backend."
                ) from exc
            self._pipeline = pipeline(
                "question-answering",
                model=str(self.model_path),
                tokenizer=str(self.model_path),
            )
        return self._pipeline


def finetune_reader(
    train_cases: list[CaseProfile],
    validation_cases: list[CaseProfile],
    config: ReaderConfig | None = None,
) -> Path:
    """Fine-tune an extractive QA reader on ALQAC training data.

    This function creates SQuAD2-format training data from ALQAC cases,
    then runs Hugging Face Trainer to fine-tune the base model.

    Returns the path to the best checkpoint directory.
    """
    try:
        from transformers import (
            AutoModelForQuestionAnswering,
            AutoTokenizer,
            DefaultDataCollator,
            Trainer,
            TrainingArguments,
        )
        from transformers.tokenization_utils_base import BatchEncoding
    except ImportError as exc:
        raise ImportError(
            "Fine-tuning requires `transformers` and PyTorch. "
            "Install requirements first."
        ) from exc

    config = config or ReaderConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert to SQuAD2 format and save as JSON
    train_dataset_obj = LegalQADataset(train_cases, split_name="train")
    val_dataset_obj = LegalQADataset(validation_cases, split_name="validation")

    train_dict = train_dataset_obj.to_squad_dict()
    val_dict = val_dataset_obj.to_squad_dict()

    train_json_path = output_dir / "train_squad.json"
    val_json_path = output_dir / "val_squad.json"
    train_json_path.write_text(json.dumps(train_dict, ensure_ascii=False), encoding="utf-8")
    val_json_path.write_text(json.dumps(val_dict, ensure_ascii=False), encoding="utf-8")

    # 2. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForQuestionAnswering.from_pretrained(config.base_model)

    # 3. Tokenize using SQuAD-style preprocessing
    train_features = _tokenize_squad_data(train_dict, tokenizer, config)
    val_features = _tokenize_squad_data(val_dict, tokenizer, config)

    # 4. Set up Trainer
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        report_to="none",
    )

    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=val_features,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_squad_metrics,
    )

    # 5. Train
    logger.info("Starting fine-tuning with %d train, %d validation samples",
                len(train_features), len(val_features))
    trainer.train()

    # 6. Save best model
    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    # 7. Save training metadata
    metadata = {
        "base_model": config.base_model,
        "num_train_cases": len(train_cases),
        "num_val_cases": len(validation_cases),
        "num_train_features": len(train_features),
        "num_val_features": len(val_features),
        "config": {
            "max_seq_length": config.max_seq_length,
            "doc_stride": config.doc_stride,
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_train_epochs,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
        },
    }
    metadata_path = best_model_dir / "training_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Fine-tuned model saved to %s", best_model_dir)
    return best_model_dir


def _tokenize_squad_data(
    squad_dict: dict[str, Any],
    tokenizer,
    config: ReaderConfig,
) -> list[dict[str, Any]]:
    """Tokenize SQuAD2-format data and return features for training.

    Uses Hugging Face's SQuAD-style tokenization approach:
    - Split long contexts into overlapping windows (doc_stride)
    - Locate answer start/end token positions
    - Mark impossible questions (no answer span)
    """
    features: list[dict[str, Any]] = []

    for article in squad_dict.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context_text = paragraph["context"]
            for qa in paragraph.get("qas", []):
                question_text = qa["question"]
                is_impossible = qa.get("is_impossible", False)
                answer_list = qa.get("answers", [])

                # Tokenize with truncation and stride
                tokenized = tokenizer(
                    question_text,
                    context_text,
                    max_length=config.max_seq_length,
                    stride=config.doc_stride,
                    truncation="only_second",
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                )

                # Find answer span in context
                if not is_impossible and answer_list:
                    answer_text = answer_list[0]["text"]
                    answer_start_char = answer_list[0]["answer_start"]
                    answer_end_char = answer_start_char + len(answer_text)

                    offset_mapping = tokenized.pop("offset_mapping", None)
                    sample_ids = tokenized.pop("overflow_to_sample_mapping", None)

                    input_ids_list = tokenized["input_ids"]
                    attention_mask_list = tokenized["attention_mask"]

                    if isinstance(input_ids_list, (list, tuple)) and len(input_ids_list) > 0:
                        for i in range(len(input_ids_list)):
                            start_position = 0
                            end_position = 0

                            if offset_mapping and i < len(offset_mapping):
                                offsets = offset_mapping[i]
                                cls_token_id = tokenizer.cls_token_id

                                # Find token positions that overlap with answer
                                for idx, (os_start, os_end) in enumerate(offsets):
                                    if os_start == 0 and os_end == 0:
                                        continue
                                    if os_start <= answer_start_char < os_end:
                                        start_position = idx
                                    if os_start < answer_end_char <= os_end:
                                        end_position = idx

                            feature = {
                                "input_ids": input_ids_list[i],
                                "attention_mask": attention_mask_list[i],
                                "start_positions": start_position,
                                "end_positions": end_position,
                            }
                            features.append(feature)
                    else:
                        # Single feature (short context)
                        offsets = offset_mapping[0] if offset_mapping else []
                        start_position = 0
                        end_position = 0
                        for idx, (os_start, os_end) in enumerate(offsets):
                            if os_start == 0 and os_end == 0:
                                continue
                            if os_start <= answer_start_char < os_end:
                                start_position = idx
                            if os_start < answer_end_char <= os_end:
                                end_position = idx
                        feature = {
                            "input_ids": tokenized["input_ids"],
                            "attention_mask": tokenized["attention_mask"],
                            "start_positions": start_position,
                            "end_positions": end_position,
                        }
                        features.append(feature)
                else:
                    # is_impossible: no answer in this context
                    if isinstance(tokenized["input_ids"], list) and tokenized["input_ids"]:
                        for i in range(len(tokenized["input_ids"])):
                            feature = {
                                "input_ids": tokenized["input_ids"][i],
                                "attention_mask": tokenized["attention_mask"][i],
                                "start_positions": 0,
                                "end_positions": 0,
                            }
                            features.append(feature)
                    else:
                        feature = {
                            "input_ids": tokenized["input_ids"],
                            "attention_mask": tokenized["attention_mask"],
                            "start_positions": 0,
                            "end_positions": 0,
                        }
                        features.append(feature)

    return features


def _compute_squad_metrics(eval_pred) -> dict[str, float]:
    """Compute exact match and F1 for SQuAD-style evaluation during training."""
    try:
        from transformers import EvalPrediction
    except ImportError:
        pass

    predictions, labels = eval_pred
    start_preds = predictions[0].argmax(axis=-1)
    end_preds = predictions[1].argmax(axis=-1)

    exact_matches = 0
    f1_scores = []
    total = 0

    for start_pred, end_pred, start_label, end_label in zip(
        start_preds, end_preds, labels[0], labels[1], strict=False
    ):
        if start_label == 0 and end_label == 0:
            # Impossible question: correct if pred is also [0,0]
            if start_pred == 0 and end_pred == 0:
                exact_matches += 1
                f1_scores.append(1.0)
            else:
                f1_scores.append(0.0)
            total += 1
            continue

        pred_start = min(start_pred, end_pred)
        pred_end = max(start_pred, end_pred)
        label_start = min(start_label, end_label)
        label_end = max(start_label, end_label)

        if pred_start == label_start and pred_end == label_end:
            exact_matches += 1

        pred_tokens = set(range(pred_start, pred_end + 1))
        label_tokens = set(range(label_start, label_end + 1))
        common = pred_tokens & label_tokens
        if common:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(label_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)
        total += 1

    em = exact_matches / total if total else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {"exact_match": em, "f1": avg_f1}


def load_finetuned_reader(
    model_path: str | Path,
    *,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    max_answer_length: int = 50,
) -> LegalQAReader:
    """Load a fine-tuned reader from a checkpoint directory."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Fine-tuned reader not found at: {path}")
    return LegalQAReader(
        model_path=path,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_answer_length=max_answer_length,
    )
