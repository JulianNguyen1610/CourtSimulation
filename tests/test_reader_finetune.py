"""Unit tests for fine-tuned reader dataset and SQuAD parsing."""

from __future__ import annotations

import unittest

from src.models import CaseProfile
from src.reader.finetune_reader import (
    LegalQADataset,
    _extract_squad_answer_span,
    _tokenize_squad_data,
    ReaderConfig,
)


class SquadAnswerParsingTest(unittest.TestCase):
    def test_extract_dict_format_from_legal_qa_dataset(self) -> None:
        case = CaseProfile(
            case_id="qa-1",
            question="Thoi han bao quan la bao lau?",
            context="Thoi han bao quan dat toi da 20 ngay.",
            answer="20 ngay",
        )
        squad = LegalQADataset([case], split_name="train").to_squad_dict()
        qa = squad["data"][0]["paragraphs"][0]["qas"][0]

        text, start = _extract_squad_answer_span(
            qa["answers"],
            is_impossible=qa["is_impossible"],
        )
        self.assertEqual(text, "20 ngay")
        self.assertEqual(start, case.context.index("20 ngay"))

    def test_extract_list_format(self) -> None:
        text, start = _extract_squad_answer_span(
            [{"text": "20 ngay", "answer_start": 28}],
            is_impossible=False,
        )
        self.assertEqual(text, "20 ngay")
        self.assertEqual(start, 28)

    def test_impossible_answer_returns_none(self) -> None:
        text, start = _extract_squad_answer_span(
            {"text": [], "answer_start": []},
            is_impossible=True,
        )
        self.assertIsNone(text)
        self.assertIsNone(start)


class TokenizeSquadDataTest(unittest.TestCase):
    def test_tokenize_legal_qa_dict_without_key_error(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers not installed")

        case = CaseProfile(
            case_id="qa-2",
            question="Muc phat toi thieu la bao nhieu?",
            context="Muc phat toi thieu la 6 thang tu.",
            answer="6 thang tu",
        )
        squad = LegalQADataset([case], split_name="train").to_squad_dict()
        tokenizer = AutoTokenizer.from_pretrained("deepset/xlm-roberta-base-squad2")
        features = _tokenize_squad_data(
            squad,
            tokenizer,
            ReaderConfig(max_seq_length=128, doc_stride=64),
        )
        self.assertGreater(len(features), 0)
        self.assertIn("input_ids", features[0])
        self.assertIn("start_positions", features[0])
        self.assertIn("end_positions", features[0])


if __name__ == "__main__":
    unittest.main()
