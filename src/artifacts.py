"""Utilities for saving experiment artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.models import CaseProfile, DebateResult


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


def _to_json(payload: object) -> str:
    # Keep json import local to avoid exposing a wider serialization API.
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)
