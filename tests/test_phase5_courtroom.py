"""Phase 3 courtroom protocol, data model, and LJP evaluator tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agents.compat import ProponentAgent, create_phase1_debate_pair
from src.agents.defendant import DefendantAgent
from src.agents.defense import DefenseAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.prosecutor import ProsecutorAgent
from src.courtroom.protocol import CourtroomProtocol, ProtocolConfig
from src.courtroom.session import CourtroomSession
from src.data_loader import load_court_case_json
from src.evaluation.ljp_evaluator import (
    LJPEvaluator,
    normalize_label,
    parse_sentence_years,
    sentence_bucket,
)
from src.llm import MockLLM
from src.models import CourtCase, EvidenceItem, JudgmentGroundTruth, LegalJudgment


class CourtroomDataModelTest(unittest.TestCase):
    def test_court_case_round_trip_json(self) -> None:
        case = CourtCase(
            case_id="test-1",
            facts="Bi cao lay cap tai san.",
            evidence=[
                EvidenceItem(evidence_id="ev-1", text="Camera ghi nhan hanh vi.")
            ],
            ground_truth=JudgmentGroundTruth(
                charge="Trom cap tai san",
                articles=["Dieu 173"],
                sentence="12 thang tu",
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "case.json"
            path.write_text(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "facts": case.facts,
                        "evidence": [item.model_dump() for item in case.evidence],
                        "ground_truth": case.ground_truth.model_dump(),
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_court_case_json(path)
            self.assertEqual(loaded.case_id, "test-1")
            self.assertEqual(loaded.evidence[0].evidence_id, "ev-1")

    def test_court_case_to_case_profile(self) -> None:
        case = CourtCase(
            case_id="test-2",
            facts="Su viec xay ra tai Ha Noi.",
            ground_truth=JudgmentGroundTruth(
                charge="Trom cap",
                articles=[],
                sentence="6 thang",
            ),
        )
        profile = case.to_case_profile()
        self.assertIn("Su viec xay ra", profile.context)
        self.assertTrue(profile.metadata.get("courtroom_case"))


class CourtroomProtocolTest(unittest.TestCase):
    def test_opening_turn_order(self) -> None:
        court_case = CourtCase(case_id="p-1", facts="Mot vu an thu nghiem.")
        llm = MockLLM()
        protocol = CourtroomProtocol(ProtocolConfig(enable_debate=False, enable_closing=False))
        judge = JudgeAgent(llm)
        prosecutor = ProsecutorAgent(llm)
        defense = DefenseAgent(llm)
        defendant = DefendantAgent(llm)

        turns, phases = protocol.opening(
            court_case=court_case,
            judge=judge,
            prosecutor=prosecutor,
            defendant=defendant,
            defense=defense,
            legal_evidence=[],
            past_memory=__import__("src.models", fromlist=["MemoryContext"]).MemoryContext(),
            transcript=[],
        )

        self.assertEqual(len(turns), 4)
        self.assertEqual([turn.role for turn in turns], ["judge", "prosecutor", "defendant", "defense"])
        self.assertIn("opening_judge", phases)
        self.assertIn("opening_defense", phases)


class LJPEvaluatorTest(unittest.TestCase):
    def test_charge_and_sentence_metrics(self) -> None:
        predicted = LegalJudgment(
            charge="Trom cap tai san",
            articles=["Dieu 173"],
            sentence="12 thang tu",
            reasoning="Co du bang chung.",
            cited_evidence_ids=["ev-1"],
        )
        gold = JudgmentGroundTruth(
            charge="trom cap tai san",
            articles=["dieu 173"],
            sentence="1 nam tu",
        )
        result = LJPEvaluator().evaluate(
            predicted,
            gold,
            valid_evidence_ids={"ev-1", "ev-2"},
        )
        self.assertEqual(result.charge_accuracy, 1.0)
        self.assertEqual(result.article_accuracy, 1.0)
        self.assertEqual(result.citation_validity, 1.0)
        self.assertIsNotNone(result.sentence_mae_years)

    def test_sentence_bucket_helpers(self) -> None:
        self.assertEqual(parse_sentence_years("12 thang tu"), 1.0)
        self.assertEqual(sentence_bucket(2.5), "1_3y")
        self.assertEqual(normalize_label("Điều 173"), "dieu 173")


class CourtroomCompatTest(unittest.TestCase):
    def test_phase1_aliases(self) -> None:
        llm = MockLLM()
        proponent, opponent = create_phase1_debate_pair(llm, llm)
        self.assertEqual(proponent.role, "proponent")
        self.assertEqual(ProponentAgent(llm).role, "proponent")


class CourtroomSessionTest(unittest.TestCase):
    def test_session_runs_with_mock_llm(self) -> None:
        court_case = load_court_case_json("data/processed/case_01_theft.json")
        llm = MockLLM()
        agent_kwargs = {"argument_max_tokens": 128}
        judge_kwargs = {}
        session = CourtroomSession.from_config(
            "configs/courtroom.yaml",
            prosecutor=ProsecutorAgent(llm, **agent_kwargs),
            defense=DefenseAgent(llm, **agent_kwargs),
            defendant=DefendantAgent(llm, **agent_kwargs),
            judge=JudgeAgent(llm, **judge_kwargs),
        )
        result = session.run(court_case)
        self.assertGreaterEqual(len(result.transcript), 4)
        self.assertIn("final_ruling", result.phases_completed)
        self.assertIsNotNone(result.legal_judgment)


if __name__ == "__main__":
    unittest.main()
