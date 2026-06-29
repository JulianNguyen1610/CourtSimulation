"""Tests for judge-mediated Phase 1 debate orchestration."""

from __future__ import annotations

import unittest

from src.agents.debate_agent import DebateAgent
from src.agents.judge_agent import JudgeAgent
from src.judge_mediated_orchestrator import JudgeMediatedOrchestrator
from src.llm import MockLLM
from src.models import CaseProfile
from src.orchestrator import DebateOrchestrator, create_debate_orchestrator


class JudgeMediatedOrchestratorTest(unittest.TestCase):
    def _build_case(self) -> CaseProfile:
        return CaseProfile(
            case_id="test-case",
            context="Người vi phạm có thể bị phạt tù đến 07 năm.",
            question="Có thể bị phạt tù đến bao nhiêu năm?",
            answer="07 năm",
        )

    def _build_orchestrator(self, rounds: int = 2) -> JudgeMediatedOrchestrator:
        mock_llm = MockLLM()
        return JudgeMediatedOrchestrator(
            proponent=DebateAgent("proponent", mock_llm),
            opponent=DebateAgent("opponent", mock_llm),
            judge=JudgeAgent(mock_llm),
            rounds=rounds,
        )

    def test_mock_judge_mediated_debate_completes_rounds(self) -> None:
        result = self._build_orchestrator(rounds=2).run(self._build_case())

        self.assertEqual(len(result.belief_history), 2)
        self.assertIsNotNone(result.verdict)
        control_turns = [
            turn
            for turn in result.transcript
            if turn.private_strategy.startswith("control:")
        ]
        self.assertEqual(len(control_turns), 4)
        self.assertEqual(result.transcript[-2].private_strategy, "closing_statement")
        self.assertEqual(result.transcript[-1].private_strategy, "closing_statement")

    def test_judge_fallback_control_schedule(self) -> None:
        judge = JudgeAgent(MockLLM())

        first = judge.decide_control_action(
            case=self._build_case(),
            transcript=[],
            completed_rounds=0,
            max_rounds=2,
            last_speaker_role=None,
        )
        self.assertEqual(first.action, "call_proponent")

        after_proponent = judge.decide_control_action(
            case=self._build_case(),
            transcript=[],
            completed_rounds=0,
            max_rounds=2,
            last_speaker_role="proponent",
        )
        self.assertEqual(after_proponent.action, "call_opponent")

        after_rounds = judge.decide_control_action(
            case=self._build_case(),
            transcript=[],
            completed_rounds=2,
            max_rounds=2,
            last_speaker_role="opponent",
        )
        self.assertEqual(after_rounds.action, "request_closing")

    def test_create_debate_orchestrator_factory(self) -> None:
        mock_llm = MockLLM()
        fixed = create_debate_orchestrator(
            "fixed",
            proponent=DebateAgent("proponent", mock_llm),
            opponent=DebateAgent("opponent", mock_llm),
            judge=JudgeAgent(mock_llm),
            rounds=1,
        )
        mediated = create_debate_orchestrator(
            "judge_mediated",
            proponent=DebateAgent("proponent", mock_llm),
            opponent=DebateAgent("opponent", mock_llm),
            judge=JudgeAgent(mock_llm),
            rounds=1,
        )

        self.assertIsInstance(fixed, DebateOrchestrator)
        self.assertIsInstance(mediated, JudgeMediatedOrchestrator)
        self.assertNotIsInstance(fixed, JudgeMediatedOrchestrator)


if __name__ == "__main__":
    unittest.main()
