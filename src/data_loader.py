"""Dataset loading utilities for ViLQA/ALQAC-style legal QA data."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models import (
    CaseProfile,
    CourtCase,
    EvidenceItem,
    JudgmentGroundTruth,
    Testimony,
)


REQUIRED_VILQA_COLUMNS = {"context", "question", "answer"}


@dataclass(frozen=True)
class DatasetSplit:
    """Reproducible split of normalized case profiles."""

    train: list[CaseProfile]
    validation: list[CaseProfile]
    test: list[CaseProfile]


def load_vilqa_csv(path: str | Path) -> list[CaseProfile]:
    """Load ViLQA/ALQAC CSV data into normalized case profiles."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    missing_columns = REQUIRED_VILQA_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset {dataset_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame = frame.dropna(subset=list(REQUIRED_VILQA_COLUMNS)).reset_index(drop=True)
    cases: list[CaseProfile] = []
    for row_index, row in frame.iterrows():
        cases.append(
            CaseProfile(
                case_id=f"vilqa-{row_index}",
                context=str(row["context"]).strip(),
                question=str(row["question"]).strip(),
                answer=str(row["answer"]).strip(),
                metadata={
                    "dataset": "ViLQA/ALQAC",
                    "source_path": str(dataset_path),
                    "row_index": int(row_index),
                },
            )
        )

    if not cases:
        raise ValueError(f"Dataset {dataset_path} has no valid QA rows.")
    return cases


def split_cases(
    cases: list[CaseProfile],
    train_count: int,
    test_count: int,
    seed: int,
) -> DatasetSplit:
    """Split cases into train/validation/test with fixed counts.
    
    Args:
        cases: All cases to split
        train_count: Number of cases for training (e.g., 200)
        test_count: Number of cases for testing (e.g., 200)
        seed: Random seed for reproducibility
        
    Returns:
        DatasetSplit with train/validation/test splits.
        Validation gets all remaining cases.
    """
    total = len(cases)
    if train_count + test_count >= total:
        raise ValueError(
            f"train_count ({train_count}) + test_count ({test_count}) "
            f"must be less than total cases ({total})"
        )
    if train_count <= 0 or test_count <= 0:
        raise ValueError("train_count and test_count must be positive")

    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)

    train_end = train_count
    validation_end = total - test_count

    return DatasetSplit(
        train=shuffled[:train_end],
        validation=shuffled[train_end:validation_end],
        test=shuffled[validation_end:],
    )
def load_court_case_json(path: str | Path) -> CourtCase:
    """Load one structured courtroom LJP case from JSON."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Court case file not found: {dataset_path}")

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Court case JSON must be an object: {dataset_path}")

    ground_truth_payload = payload.get("ground_truth")
    ground_truth = None
    if isinstance(ground_truth_payload, dict):
        ground_truth = JudgmentGroundTruth(
            charge=str(ground_truth_payload.get("charge", "")).strip(),
            articles=[
                str(article).strip()
                for article in ground_truth_payload.get("articles", [])
            ],
            sentence=str(ground_truth_payload.get("sentence", "")).strip(),
            reasoning=ground_truth_payload.get("reasoning"),
        )

    evidence = [
        EvidenceItem(
            evidence_id=str(item.get("evidence_id", f"evidence-{index}")),
            text=str(item.get("text", "")).strip(),
            source_type=str(item.get("source_type", "document")),
            metadata=item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
        )
        for index, item in enumerate(payload.get("evidence", []))
        if isinstance(item, dict)
    ]
    testimonies = [
        Testimony(
            speaker=str(item.get("speaker", "unknown")),
            text=str(item.get("text", "")).strip(),
            role=str(item.get("role", "witness")),
            metadata=item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
        )
        for item in payload.get("testimonies", [])
        if isinstance(item, dict)
    ]

    case_id = str(payload.get("case_id", dataset_path.stem))
    facts = str(payload.get("facts", "")).strip()
    if not facts:
        raise ValueError(f"Court case {dataset_path} is missing non-empty facts.")

    return CourtCase(
        case_id=case_id,
        case_type=str(payload.get("case_type", "courtroom_ljp")),
        facts=facts,
        evidence=evidence,
        testimonies=testimonies,
        ground_truth=ground_truth,
        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
    )


def load_simucourt(
    split: str = "train",
    limit: int | None = None,
    cache_dir: str | Path | None = None,
) -> list[CourtCase]:
    """Load SimuCourt/AgentsCourt cases when the HF dataset is available."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "load_simucourt requires the `datasets` package. "
            "Install dependencies from requirements.txt."
        ) from exc

    dataset = load_dataset(
        "law-ai/SimuCourt",
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
    )
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in SimuCourt dataset.")

    cases: list[CourtCase] = []
    rows = dataset[split]
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        facts = str(
            row.get("facts")
            or row.get("case_facts")
            or row.get("description")
            or ""
        ).strip()
        if not facts:
            continue

        articles_raw = row.get("articles") or row.get("law_articles") or []
        if isinstance(articles_raw, str):
            articles = [part.strip() for part in articles_raw.split(",") if part.strip()]
        elif isinstance(articles_raw, list):
            articles = [str(item).strip() for item in articles_raw]
        else:
            articles = []

        ground_truth = JudgmentGroundTruth(
            charge=str(row.get("charge") or row.get("crime") or "").strip(),
            articles=articles,
            sentence=str(row.get("sentence") or row.get("penalty") or "").strip(),
            reasoning=str(row.get("reasoning") or row.get("judgment") or "").strip() or None,
        )
        cases.append(
            CourtCase(
                case_id=str(row.get("case_id") or row.get("id") or f"simucourt-{split}-{index}"),
                facts=facts,
                evidence=[
                    EvidenceItem(
                        evidence_id=f"simucourt-{split}-{index}-facts",
                        text=facts,
                        source_type="dataset_fact",
                    )
                ],
                ground_truth=ground_truth,
                metadata={"dataset": "SimuCourt", "split": split, "row_index": index},
            )
        )

    if not cases:
        raise ValueError(f"No valid SimuCourt rows found for split={split}.")
    return cases


def load_vlegal(
    task: str = "court_decision_prediction",
    split: str = "test",
    limit: int | None = None,
    cache_dir: str | Path | None = None,
) -> list[CourtCase]:
    """Load VLegal-Bench courtroom-related tasks when HF dataset is available."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "load_vlegal requires the `datasets` package. "
            "Install dependencies from requirements.txt."
        ) from exc

    dataset = load_dataset(
        "CMC-OPENAI/VLegal-Bench",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in VLegal-Bench dataset.")

    cases: list[CourtCase] = []
    for index, row in enumerate(dataset[split]):
        if limit is not None and index >= limit:
            break
        row_task = str(row.get("task") or row.get("task_name") or task)
        if task not in row_task and row_task not in task:
            continue

        facts = str(
            row.get("input")
            or row.get("context")
            or row.get("question")
            or ""
        ).strip()
        if not facts:
            continue

        answer = str(row.get("answer") or row.get("output") or "").strip()
        ground_truth = JudgmentGroundTruth(
            charge=answer,
            articles=[],
            sentence=answer,
            reasoning=None,
        )
        cases.append(
            CourtCase(
                case_id=str(row.get("id") or f"vlegal-{split}-{index}"),
                facts=facts,
                ground_truth=ground_truth,
                metadata={
                    "dataset": "VLegal-Bench",
                    "task": row_task,
                    "split": split,
                    "row_index": index,
                },
            )
        )

    if not cases:
        raise ValueError(
            f"No VLegal-Bench rows matched task={task!r} for split={split}."
        )
    return cases
