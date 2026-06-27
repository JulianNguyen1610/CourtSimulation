"""Utilities for saving experiment artifacts."""

from __future__ import annotations

from datetime import datetime, timezone

# Python 3.10 compatibility: UTC alias
UTC = timezone.utc
from pathlib import Path

from src.models import CaseProfile, CourtCase, CourtroomResult, DebateResult


def save_debate_result(
    result: DebateResult,
    case: CaseProfile,
    output_dir: str | Path = "outputs/vilqa_multi_agent_baseline",
) -> Path:
    """Save a debate result as JSON without exposing hidden runtime state."""

    root = Path(output_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    case_dir = root / f"{timestamp}_{case.case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)

    output_path = case_dir / "debate_result.json"
    payload = {
        "case": case.agent_view(),
        "result": result.model_dump(mode="json"),
    }
    output_path.write_text(
        _to_json(payload),
        encoding="utf-8",
    )
    return output_path


def save_courtroom_result(
    result: CourtroomResult,
    court_case: CourtCase,
    output_dir: str | Path = "outputs/courtroom_pilot",
) -> Path:
    """Save a courtroom session transcript and LJP verdict as JSON."""

    root = Path(output_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    case_dir = root / f"{timestamp}_{court_case.case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)

    output_path = case_dir / "courtroom_result.json"
    payload = {
        "case": court_case.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
    }
    output_path.write_text(_to_json(payload), encoding="utf-8")

    if result.evaluation is not None:
        metrics_path = case_dir / "ljp_metrics.json"
        metrics_path.write_text(
            _to_json(result.evaluation.model_dump(mode="json")),
            encoding="utf-8",
        )

    return output_path


def _to_json(payload: object) -> str:
    # Keep json import local to avoid exposing a wider serialization API.
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)
