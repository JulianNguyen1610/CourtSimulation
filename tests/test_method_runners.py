from __future__ import annotations
import unittest
from src.llm import MockLLM
from src.models import CaseProfile
from src.methods import ContextBundle, MethodPrediction, run_llm_method

class RecordingLLM(MockLLM):
    def __init__(self): super().__init__(); self.prompts = []
    def generate(self, prompt: str) -> str: self.prompts.append(prompt); return super().generate(prompt)

class MethodRunnerTest(unittest.TestCase):
    def setUp(self): self.case = CaseProfile(case_id="x", context="Án phạt là 02 năm.", question="Bao nhiêu năm?", answer="02 năm")
    def test_prediction_contract_and_context(self):
        llm = RecordingLLM(); bundle = ContextBundle.from_case(self.case)
        result = run_llm_method("direct", self.case, bundle, llm)
        self.assertIsInstance(result, MethodPrediction); self.assertEqual(result.raw_output, result.raw_answer)
        self.assertIn("ANSWER CONTRACT", llm.prompts[0]); self.assertNotIn('"answer"', llm.prompts[0])
    def test_self_debate_one_call_unstructured_turns(self):
        bundle = ContextBundle.from_case(self.case); llm = RecordingLLM()
        self.assertEqual(run_llm_method("self_debate_single_call", self.case, bundle, llm, 3).llm_calls, 1)
        self.assertEqual(run_llm_method("unstructured_multi_agent", self.case, bundle, llm, 2).llm_calls, 9)
    def test_shared_primary_context(self):
        bundle = ContextBundle.from_case(self.case); llm = RecordingLLM()
        run_llm_method("direct", self.case, bundle, llm); run_llm_method("self_debate_single_call", self.case, bundle, llm)
        self.assertTrue(all(self.case.context in prompt for prompt in llm.prompts))
