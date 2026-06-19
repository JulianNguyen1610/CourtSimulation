"""Tests for cached Hugging Face extractive QA pipeline loading."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from src.baselines import (
    _get_extractive_qa_pipeline,
    clear_extractive_qa_pipeline_cache,
    extractive_qa_prediction,
)
from src.models import CaseProfile


class ExtractiveQAPipelineCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_extractive_qa_pipeline_cache()

    def tearDown(self) -> None:
        clear_extractive_qa_pipeline_cache()

    def test_pipeline_loaded_once_per_model(self) -> None:
        reader = MagicMock(return_value={"answer": "07 năm"})
        fake_transformers = MagicMock()
        fake_transformers.pipeline = MagicMock(return_value=reader)
        case = CaseProfile(
            case_id="vilqa-0",
            context="Có thể bị phạt tù lên đến 07 năm.",
            question="Bị phạt bao nhiêu năm?",
            answer="07 năm",
        )

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            first = extractive_qa_prediction(case, model_name="test-qa-model")
            second = extractive_qa_prediction(case, model_name="test-qa-model")

        self.assertEqual(first, "07 năm")
        self.assertEqual(second, "07 năm")
        fake_transformers.pipeline.assert_called_once_with(
            "question-answering",
            model="test-qa-model",
            tokenizer="test-qa-model",
        )
        self.assertEqual(reader.call_count, 2)

    def test_get_pipeline_caches_by_model_name(self) -> None:
        reader_a = MagicMock()
        reader_b = MagicMock()
        fake_transformers = MagicMock()
        fake_transformers.pipeline = MagicMock(side_effect=[reader_a, reader_b])

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            got_a = _get_extractive_qa_pipeline("model-a")
            got_b = _get_extractive_qa_pipeline("model-b")
            got_a_again = _get_extractive_qa_pipeline("model-a")

        self.assertIs(got_a, reader_a)
        self.assertIs(got_b, reader_b)
        self.assertIs(got_a_again, reader_a)
        self.assertEqual(fake_transformers.pipeline.call_count, 2)


if __name__ == "__main__":
    unittest.main()
