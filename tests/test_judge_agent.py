"""Tests for robust judge output parsing and fallback accounting."""

from __future__ import annotations

import unittest

from src.agents.judge_agent import JudgeAgent
from src.llm import MockLLM
from src.models import BeliefState, CaseProfile


class JudgeAgentParsingTest(unittest.TestCase):
    def test_verdict_parser_extracts_json_from_markdown_fence(self) -> None:
        case = CaseProfile(
            case_id="case-1",
            context="Người vi phạm có thể bị phạt tù đến 07 năm.",
            question="Có thể bị phạt tù đến bao nhiêu năm?",
            answer="07 năm",
        )
        judge = JudgeAgent(MockLLM())

        verdict = judge._parse_verdict(
            '```json\n{"prediction": "07 năm", "answer": "07 năm", '
            '"confidence": 80, "reasoning": "Có trong ngữ cảnh."}\n```',
            case,
        )

        self.assertEqual(verdict.answer, "07 năm")
        self.assertEqual(judge.fallback_count, 0)

    def test_invalid_verdict_fallback_prefers_latest_belief(self) -> None:
        case = CaseProfile(
            case_id="case-1",
            context="Người vi phạm có thể bị phạt tù đến 07 năm.",
            question="Có thể bị phạt tù đến bao nhiêu năm?",
            answer="07 năm",
        )
        judge = JudgeAgent(MockLLM())
        judge.belief_history.append(
            BeliefState(
                round_index=1,
                prediction="belief answer",
                confidence=60,
                reasoning="Prior debate belief.",
            )
        )

        verdict = judge._parse_verdict("not json", case)

        self.assertEqual(verdict.answer, "belief answer")
        self.assertEqual(judge.fallback_count, 1)


if __name__ == "__main__":
    unittest.main()
