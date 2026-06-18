"""LLM-as-judge evaluator for debate quality dimensions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.llm import LLMClient
from src.models import AgentOutput, CaseProfile, EvalResult, Verdict


class EvaluatorAgent:
    """Evaluate legal reasoning quality without seeing gold answers."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_dir: str | Path = "configs/prompts",
    ) -> None:
        self.llm = llm
        self.prompt_dir = Path(prompt_dir)

    def evaluate(
        self,
        case: CaseProfile,
        transcript: list[AgentOutput],
        verdict: Verdict,
    ) -> EvalResult:
        """Return rubric scores for non-gold qualitative dimensions."""

        prompt = self._render_prompt(
            "evaluator.txt",
            case_profile=case.agent_view(),
            transcript=[turn.model_dump() for turn in transcript],
            verdict=verdict.model_dump(),
        )
        raw_output = self.llm.generate(prompt)
        data = self._loads_json_or_empty(raw_output)
        try:
            return EvalResult(
                legal_accuracy=self._score(data.get("legal_accuracy")),
                argument_quality=self._score(data.get("argument_quality")),
                logical_consistency=self._score(data.get("logical_consistency")),
                notes=str(data.get("notes") or "LLM evaluator completed."),
            )
        except (TypeError, ValueError, ValidationError):
            return EvalResult(
                legal_accuracy=None,
                argument_quality=None,
                logical_consistency=None,
                notes="LLM evaluator output failed validation.",
            )

    def _render_prompt(self, template_name: str, **values: object) -> str:
        template_path = self.prompt_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        return template_path.read_text(encoding="utf-8").format(**values)

    @staticmethod
    def _loads_json_or_empty(raw_output: str) -> dict[str, Any]:
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw_output,
            flags=re.DOTALL | re.IGNORECASE,
        )
        json_text = fence_match.group(1) if fence_match else None
        if json_text is None:
            block_match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
            json_text = block_match.group(0) if block_match else raw_output.strip()
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _score(value: object) -> float | None:
        if value is None:
            return None
        score = float(value)
        return min(1.0, max(0.0, score))
