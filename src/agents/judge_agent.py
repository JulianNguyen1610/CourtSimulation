"""Judge agent with belief tracking for debate rounds."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.llm import LLMClient, extract_candidate_from_context, is_mock_llm
from src.models import AgentOutput, BeliefState, CaseProfile, CourtCase, JudgeControlDecision, LegalJudgment, Verdict
from src.utils.prompt_compact import compact_agent_view, compact_history, truncate_text


class JudgeAgent:
    """Judge that updates beliefs after each round and renders a verdict."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_dir: str | Path = "configs/prompts",
        max_context_chars: int | None = None,
        max_history_turns: int | None = None,
        max_history_chars: int | None = None,
    ) -> None:
        self.llm = llm
        self.prompt_dir = Path(prompt_dir)
        self.max_context_chars = max_context_chars
        self.max_history_turns = max_history_turns
        self.max_history_chars = max_history_chars
        self.belief_history: list[BeliefState] = []
        self.fallback_count = 0
        self.parse_attempt_count = 0
        self._courtroom_mode = False

    def enable_courtroom_mode(self) -> None:
        """Switch judge prompts to courtroom templates."""

        self._courtroom_mode = True

    def update_belief(
        self,
        case: CaseProfile,
        proponent_argument: AgentOutput,
        opponent_argument: AgentOutput,
        round_index: int,
    ) -> BeliefState:
        """Update judge belief after one complete debate round."""

        prompt = self._render_prompt(
            "judge_belief.txt",
            case_profile=self._case_view(case),
            proponent_argument=truncate_text(
                proponent_argument.public_argument,
                self.max_history_chars or 10_000,
            ),
            opponent_argument=truncate_text(
                opponent_argument.public_argument,
                self.max_history_chars or 10_000,
            ),
            previous_belief=[
                belief.model_dump() for belief in self.belief_history
            ],
            round_index=round_index,
        )
        raw_output = self._generate_json_with_optional_retry(prompt)
        belief = self._parse_belief(raw_output, case, round_index)
        self.belief_history.append(belief)
        return belief

    def render_verdict(
        self,
        case: CaseProfile,
        transcript: list[AgentOutput],
    ) -> Verdict:
        """Render final verdict after debate ends."""

        prompt = self._render_prompt(
            "judge_verdict.txt",
            case_profile=self._case_view(case),
            transcript=self._compact_transcript(transcript),
            belief_history=[belief.model_dump() for belief in self.belief_history],
        )
        raw_output = self._generate_json_with_optional_retry(prompt)
        return self._parse_verdict(raw_output, case)

    def ask_follow_up(
        self,
        case: CaseProfile,
        transcript: list[AgentOutput],
    ) -> str | None:
        """Ask one optional clarification question before closing statements."""

        prompt = self._render_prompt(
            "judge_question.txt",
            case_profile=self._case_view(case),
            transcript=self._compact_transcript(transcript),
            belief_history=[belief.model_dump() for belief in self.belief_history],
        )
        question = self.llm.generate(prompt).strip()
        if not question or question.upper().startswith("NO_QUESTION"):
            return None
        return question

    def decide_control_action(
        self,
        case: CaseProfile,
        transcript: list[AgentOutput],
        completed_rounds: int,
        max_rounds: int,
        last_speaker_role: str | None = None,
    ) -> JudgeControlDecision:
        """Choose the next debate action as presiding judge."""

        if is_mock_llm(self.llm):
            return self._fallback_control_action(
                completed_rounds=completed_rounds,
                max_rounds=max_rounds,
                last_speaker_role=last_speaker_role,
            )

        prompt = self._render_prompt(
            "judge_control.txt",
            case_profile=self._case_view(case),
            transcript=self._compact_transcript(transcript),
            belief_history=[belief.model_dump() for belief in self.belief_history],
            completed_rounds=completed_rounds,
            max_rounds=max_rounds,
            last_speaker_role=last_speaker_role or "none",
        )
        raw_output = self._generate_json_with_optional_retry(prompt)
        return self._parse_control_decision(
            raw_output,
            completed_rounds=completed_rounds,
            max_rounds=max_rounds,
            last_speaker_role=last_speaker_role,
        )

    def open_session(
        self,
        court_case: CourtCase,
        round_index: int = 0,
    ) -> AgentOutput:
        """Courtroom opening statement by the presiding judge."""

        prompt = self._render_courtroom_prompt(
            "judge_opening.txt",
            case_profile=self._court_case_view(court_case),
            round_index=round_index,
        )
        opening = self.llm.generate(prompt).strip()
        return AgentOutput(
            role="judge",
            round_index=round_index,
            private_strategy="opening",
            public_argument=opening,
        )

    def update_courtroom_belief(
        self,
        court_case: CourtCase,
        prosecutor_argument: AgentOutput,
        defense_argument: AgentOutput,
        round_index: int,
    ) -> BeliefState:
        """Update belief after one prosecutor/defense debate round."""

        prompt = self._render_courtroom_prompt(
            "judge_belief.txt",
            case_profile=self._court_case_view(court_case),
            proponent_argument=truncate_text(
                prosecutor_argument.public_argument,
                self.max_history_chars or 10_000,
            ),
            opponent_argument=truncate_text(
                defense_argument.public_argument,
                self.max_history_chars or 10_000,
            ),
            previous_belief=[belief.model_dump() for belief in self.belief_history],
            round_index=round_index,
        )
        raw_output = self._generate_json_with_optional_retry(prompt)
        belief = self._parse_belief(
            raw_output,
            court_case.to_case_profile(),
            round_index,
        )
        self.belief_history.append(belief)
        return belief

    def ask_courtroom_question(
        self,
        court_case: CourtCase,
        transcript: list[AgentOutput],
    ) -> str | None:
        """Optional judge clarification during courtroom debate."""

        prompt = self._render_courtroom_prompt(
            "judge_question.txt",
            case_profile=self._court_case_view(court_case),
            transcript=self._compact_transcript(transcript),
            belief_history=[belief.model_dump() for belief in self.belief_history],
        )
        question = self.llm.generate(prompt).strip()
        if not question or question.upper().startswith("NO_QUESTION"):
            return None
        return question

    def deliberate(
        self,
        court_case: CourtCase,
        transcript: list[AgentOutput],
    ) -> str:
        """Private judicial deliberation before final ruling."""

        prompt = self._render_courtroom_prompt(
            "judge_deliberation.txt",
            case_profile=self._court_case_view(court_case),
            transcript=self._compact_transcript(transcript),
            belief_history=[belief.model_dump() for belief in self.belief_history],
            deliberation="",
        )
        return self.llm.generate(prompt).strip()

    def render_ljp_verdict(
        self,
        court_case: CourtCase,
        transcript: list[AgentOutput],
        deliberation: str | None = None,
    ) -> LegalJudgment:
        """Render structured legal judgment for courtroom LJP."""

        prompt = self._render_courtroom_prompt(
            "judge_ljp_verdict.txt",
            case_profile=self._court_case_view(court_case),
            transcript=self._compact_transcript(transcript),
            belief_history=[belief.model_dump() for belief in self.belief_history],
            deliberation=deliberation or "",
        )
        raw_output = self._generate_json_with_optional_retry(prompt)
        return self._parse_ljp_verdict(raw_output, court_case)

    def _court_case_view(self, court_case: CourtCase) -> dict[str, object]:
        return compact_agent_view(court_case.agent_view(), self.max_context_chars)

    def _render_courtroom_prompt(self, template_name: str, **values: object) -> str:
        template_path = self.prompt_dir / "courtroom" / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        return template.format(**values)

    def _parse_ljp_verdict(self, raw_output: str, court_case: CourtCase) -> LegalJudgment:
        self.parse_attempt_count += 1
        data = self._loads_json_or_empty(raw_output)
        profile = court_case.to_case_profile()
        fallback_charge = self.belief_history[-1].prediction if self.belief_history else "unknown"
        if not data:
            self._record_fallback()
        try:
            return LegalJudgment(
                charge=str(data.get("charge") or fallback_charge),
                articles=self._coerce_string_list(data.get("articles", [])),
                sentence=str(data.get("sentence") or extract_candidate_from_context(profile)),
                reasoning=str(
                    data.get("reasoning")
                    or "Fallback LJP verdict because judge output was not valid JSON."
                ),
                confidence=self._coerce_confidence(data.get("confidence", 50.0)),
                cited_evidence_ids=self._coerce_string_list(
                    data.get("cited_evidence_ids", [])
                ),
            )
        except (TypeError, ValueError, ValidationError):
            self._record_fallback()
            return LegalJudgment(
                charge=fallback_charge,
                articles=[],
                sentence=extract_candidate_from_context(profile),
                reasoning="Fallback LJP verdict because judge output failed validation.",
                confidence=50.0,
            )

    def _case_view(self, case: CaseProfile) -> dict[str, object]:
        return compact_agent_view(case.agent_view(), self.max_context_chars)

    def _compact_transcript(self, transcript: list[AgentOutput]) -> list[dict[str, object]]:
        return compact_history(
            [turn.model_dump() for turn in transcript],
            self.max_history_turns,
            self.max_history_chars,
        )

    def reset(self) -> None:
        """Clear belief history before a new case."""

        self.belief_history.clear()
        self.fallback_count = 0
        self.parse_attempt_count = 0

    def _render_prompt(self, template_name: str, **values: object) -> str:
        template_path = self.prompt_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        return template.format(**values)

    def _parse_control_decision(
        self,
        raw_output: str,
        completed_rounds: int,
        max_rounds: int,
        last_speaker_role: str | None,
    ) -> JudgeControlDecision:
        self.parse_attempt_count += 1
        data = self._loads_json_or_empty(raw_output)
        if not data:
            self._record_fallback()
            return self._fallback_control_action(
                completed_rounds=completed_rounds,
                max_rounds=max_rounds,
                last_speaker_role=last_speaker_role,
            )
        try:
            action = str(data.get("action", "")).strip()
            allowed = {
                "call_proponent",
                "call_opponent",
                "ask_question",
                "request_closing",
                "end_debate",
            }
            if action not in allowed:
                raise ValueError(f"Unsupported control action: {action}")
            return JudgeControlDecision(
                action=action,  # type: ignore[arg-type]
                message=str(data.get("message") or ""),
                confidence=self._coerce_confidence(data.get("confidence", 50.0)),
                reasoning=str(
                    data.get("reasoning")
                    or "Judge selected the next debate action."
                ),
            )
        except (TypeError, ValueError, ValidationError):
            self._record_fallback()
            return self._fallback_control_action(
                completed_rounds=completed_rounds,
                max_rounds=max_rounds,
                last_speaker_role=last_speaker_role,
            )

    @staticmethod
    def _fallback_control_action(
        completed_rounds: int,
        max_rounds: int,
        last_speaker_role: str | None,
    ) -> JudgeControlDecision:
        """Deterministic control schedule used by MockLLM and JSON fallbacks."""

        if completed_rounds >= max_rounds:
            return JudgeControlDecision(
                action="request_closing",
                reasoning="Fallback control: debate rounds complete.",
            )
        if last_speaker_role in {None, "judge", "opponent"}:
            return JudgeControlDecision(
                action="call_proponent",
                reasoning="Fallback control: start or resume with Proponent.",
            )
        if last_speaker_role == "proponent":
            return JudgeControlDecision(
                action="call_opponent",
                reasoning="Fallback control: Opponent rebuts Proponent.",
            )
        return JudgeControlDecision(
            action="call_proponent",
            reasoning="Fallback control: continue debate with Proponent.",
        )

    def _parse_belief(
        self,
        raw_output: str,
        case: CaseProfile,
        round_index: int,
    ) -> BeliefState:
        self.parse_attempt_count += 1
        data = self._loads_json_or_empty(raw_output)
        fallback_prediction = (
            self._parse_text_candidate(raw_output)
            or extract_candidate_from_context(case)
        )
        if not data:
            self._record_fallback()
        try:
            return BeliefState(
                round_index=round_index,
                prediction=str(
                    data.get("prediction") or fallback_prediction
                ),
                confidence=self._coerce_confidence(data.get("confidence", 50.0)),
                reasoning=str(
                    data.get("reasoning")
                    or "Fallback belief because judge output was not valid JSON."
                ),
            )
        except (TypeError, ValueError, ValidationError):
            self._record_fallback()
            return BeliefState(
                round_index=round_index,
                prediction=fallback_prediction,
                confidence=50.0,
                reasoning="Fallback belief because judge output failed validation.",
            )

    def _parse_verdict(self, raw_output: str, case: CaseProfile) -> Verdict:
        self.parse_attempt_count += 1
        data = self._loads_json_or_empty(raw_output)
        fallback_answer = self._fallback_verdict_answer(raw_output, case)
        if not data:
            self._record_fallback()
        try:
            return Verdict(
                prediction=str(data.get("prediction") or fallback_answer),
                answer=str(data.get("answer") or fallback_answer),
                confidence=self._coerce_confidence(data.get("confidence", 50.0)),
                reasoning=str(
                    data.get("reasoning")
                    or "Fallback verdict because judge output was not valid JSON."
                ),
                cited_evidence_ids=self._coerce_string_list(
                    data.get("cited_evidence_ids", [])
                ),
            )
        except (TypeError, ValueError, ValidationError):
            self._record_fallback()
            return Verdict(
                prediction=fallback_answer,
                answer=fallback_answer,
                confidence=50.0,
                reasoning="Fallback verdict because judge output failed validation.",
            )

    def _generate_json_with_optional_retry(self, prompt: str) -> str:
        raw_output = self.llm.generate(prompt)
        if self._loads_json_or_empty(raw_output) or is_mock_llm(self.llm):
            return raw_output
        retry_prompt = (
            f"{prompt}\n\n"
            "Your previous response was not valid JSON. Retry once and return "
            "valid JSON only, with no markdown or extra commentary."
        )
        return self.llm.generate(retry_prompt)

    def _fallback_verdict_answer(self, raw_output: str, case: CaseProfile) -> str:
        if self.belief_history:
            prediction = self.belief_history[-1].prediction.strip()
            if prediction:
                return prediction
        recovered = self._recover_json_field(raw_output)
        if recovered:
            return recovered
        parsed_text = self._parse_text_candidate(raw_output)
        if parsed_text:
            return parsed_text
        return extract_candidate_from_context(case)

    @staticmethod
    def _recover_json_field(raw_output: str) -> str | None:
        """Recover answer/prediction value from JSON truncated by token limits."""

        for key in ("answer", "prediction"):
            match = re.search(
                rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"',
                raw_output,
                flags=re.IGNORECASE,
            )
            if match and match.group(1).strip():
                return match.group(1).strip()
        return None

    def _record_fallback(self) -> None:
        self.fallback_count += 1

    @classmethod
    def _loads_json_or_empty(cls, raw_output: str) -> dict[str, Any]:
        json_text = cls._extract_json_text(raw_output)
        if not json_text:
            return {}
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return parsed

    @staticmethod
    def _extract_json_text(raw_output: str) -> str | None:
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            raw_output,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fence_match:
            return fence_match.group(1)
        block_match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
        if block_match:
            return block_match.group(0)
        stripped = raw_output.strip()
        return stripped if stripped.startswith("{") and stripped.endswith("}") else None

    @staticmethod
    def _parse_text_candidate(raw_output: str) -> str | None:
        labeled_match = re.search(
            r"(?:answer|prediction|final answer|đáp án|câu trả lời)\s*[:：]\s*(.+)",
            raw_output,
            flags=re.IGNORECASE,
        )
        if labeled_match:
            return labeled_match.group(1).strip().strip('"`')
        for line in raw_output.splitlines():
            cleaned = line.strip().strip("-*` ")
            if cleaned and not cleaned.startswith(("{", "}", "```")):
                return cleaned
        return None

    @staticmethod
    def _coerce_confidence(value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 50.0
        return min(100.0, max(0.0, confidence))

    @staticmethod
    def _coerce_string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]
