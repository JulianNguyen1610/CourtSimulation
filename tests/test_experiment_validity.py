from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from src.config import ExperimentConfig, SplitConfig, new_split_manifest
from src.data_loader import load_vilqa_csv, split_cases_from_manifest
from src.memory.memory_store import MemoryStore
from src.experiment_runner import BaselineBatchRunner, BatchRunConfig
from src.main import build_parser

class ExperimentValidityTest(unittest.TestCase):
    def test_yaml_defaults_and_cli_style_override(self):
        config = ExperimentConfig.from_yaml("configs/default.yaml")
        self.assertEqual(config.method.rounds, 1)
        parser = build_parser()
        self.assertIsNone(parser.parse_args([]).rounds)
        self.assertEqual(parser.parse_args(["--rounds", "3"]).rounds, 3)
    def test_manifest_deterministic_and_hash_mismatch(self):
        cases = load_vilqa_csv("data/ALQAC.csv"); split = SplitConfig(seed=42)
        one = new_split_manifest("data/ALQAC.csv", [c.case_id for c in cases], split)
        two = new_split_manifest("data/ALQAC.csv", [c.case_id for c in cases], split)
        self.assertEqual(one.train_case_ids, two.train_case_ids)
        self.assertFalse(set(one.train_case_ids) & set(one.validation_case_ids))
        with self.assertRaises(ValueError): split_cases_from_manifest(cases, one.model_copy(update={"dataset_sha256": "bad"}), "data/ALQAC.csv")
    def test_memory_contamination_is_detected(self):
        store = MemoryStore(); store.cases = [{"case_id": "validation-1", "text": "x"}]
        with self.assertRaisesRegex(ValueError, "contamination"): store.validate_case_isolation({"validation-1"})
    def test_reader_overlap_and_no_secret_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); checkpoint = root / "checkpoint"; checkpoint.mkdir()
            (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({"train_case_ids": ["eval-1"]}), encoding="utf-8")
            config = BatchRunConfig(split_name="validation", method="finetuned_reader", limit=1, rounds=1, evidence_top_k=1, memory_top_k=1, output_dir=root, memory_path=root / "m.json", finetuned_reader_path=str(checkpoint), evaluation_case_ids=["eval-1"])
            with self.assertRaisesRegex(ValueError, "overlaps"): BaselineBatchRunner._validate_reader_checkpoint(config)
            path = root / "config.json"; path.write_text(json.dumps({"resolved_config": {"api_key": "[REDACTED]"}}), encoding="utf-8")
            self.assertNotIn("super-secret", path.read_text(encoding="utf-8"))
