"""Sanity checks for Phase 2 debate orchestration."""

from __future__ import annotations

import unittest

from src.agents.debate_agent import DebateAgent
from src.agents.judge_agent import JudgeAgent
from src.llm import MockLLM
from src.models import CaseProfile
from src.orchestrator import DebateOrchestrator


class Phase2OrchestratorTest(unittest.TestCase):
    def test_mock_debate_has_expected_round_artifacts(self) -> None:
        case = CaseProfile(
            case_id="test-case",
            context="Người vi phạm có thể bị phạt tù đến 07 năm.",
            question="Có thể bị phạt tù đến bao nhiêu năm?",
            answer="07 năm",
        )
        mock_llm = MockLLM()
        orchestrator = DebateOrchestrator(
            proponent=DebateAgent("proponent", mock_llm),
            opponent=DebateAgent("opponent", mock_llm),
            judge=JudgeAgent(mock_llm),
            rounds=2,
        )

        result = orchestrator.run(case)

        self.assertEqual(len(result.transcript), 6)
        self.assertEqual(len(result.belief_history), 2)
        self.assertIsNotNone(result.verdict)
        self.assertEqual(result.transcript[-2].private_strategy, "closing_statement")
        self.assertEqual(result.transcript[-1].private_strategy, "closing_statement")

    def test_agent_view_excludes_gold_answer(self) -> None:
        case = CaseProfile(
            case_id="test-case",
            context="Context",
            question="Question",
            answer="Gold answer",
        )

        self.assertNotIn("answer", case.agent_view())


if __name__ == "__main__":
    unittest.main()
