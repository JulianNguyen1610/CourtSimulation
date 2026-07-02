"""Tests for completed simple baseline evaluation and batch runner."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.evaluator import exact_match_score, token_f1_score
from src.experiment_runner import BaselineBatchRunner, BatchRunConfig
from src.memory.memory_store import MemoryStore
from src.models import CaseProfile


class Phase4EvaluationRunnerTest(unittest.TestCase):
    def test_exact_match_and_f1_normalize_answers(self) -> None:
        self.assertEqual(exact_match_score("07 năm.", "07 năm"), 1.0)
        self.assertAlmostEqual(token_f1_score("phạt tù 07 năm", "07 năm"), 2 / 3)

    def test_batch_runner_saves_metrics_and_predictions(self) -> None:
        cases = [
            CaseProfile(
                case_id="case-1",
                context="Người vi phạm có thể bị phạt tù đến 07 năm.",
                question="Có thể bị phạt tù đến bao nhiêu năm?",
                answer="07 năm",
            ),
            CaseProfile(
                case_id="case-2",
                context="Không áp dụng tù chung thân với người dưới 18 tuổi.",
                question="Không áp dụng tù chung thân với người dưới bao nhiêu tuổi?",
                answer="18 tuổi",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs"
            memory_path = Path(temp_dir) / "memory.json"
            runner = BaselineBatchRunner(
                train_cases=cases,
                memory_store=MemoryStore.load(memory_path),
            )
            run_dir = runner.run(
                cases,
                BatchRunConfig(
                    split_name="validation",
                    method="both",
                    limit=2,
                    rounds=1,
                    evidence_top_k=1,
                    memory_top_k=1,
                    output_dir=output_dir,
                    memory_path=memory_path,
                ),
            )

            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            with (run_dir / "predictions.csv").open("r", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(metrics["num_cases"], 2)
            self.assertEqual(metrics["num_predictions"], 4)
            self.assertEqual(len(rows), 4)
            self.assertIn("direct", metrics["metrics_by_method"])
            self.assertIn("debate_judge_mediated", metrics["metrics_by_method"])


if __name__ == "__main__":
    unittest.main()
