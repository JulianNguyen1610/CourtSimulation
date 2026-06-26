"""Sanity checks for Phase 3 retrieval and memory components."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.memory.memory_store import MemoryStore
from src.models import CaseProfile, DebateResult, Verdict
from src.retrieval.legal_retriever import LegalRetriever


class Phase3RetrievalMemoryTest(unittest.TestCase):
    def test_bm25_retriever_returns_relevant_context_without_answer(self) -> None:
        cases = [
            CaseProfile(
                case_id="case-1",
                context="Tội trộm cắp tài sản có thể bị phạt tù.",
                question="Trộm cắp tài sản bị xử lý thế nào?",
                answer="phạt tù",
            ),
            CaseProfile(
                case_id="case-2",
                context="Hợp đồng dân sự vô hiệu khi vi phạm điều cấm.",
                question="Khi nào hợp đồng vô hiệu?",
                answer="vi phạm điều cấm",
            ),
        ]

        retriever = LegalRetriever.from_cases(cases)
        results = retriever.retrieve("hợp đồng dân sự vô hiệu", top_k=1)

        self.assertEqual(results[0].metadata["case_id"], "case-2")
        self.assertNotIn("answer", results[0].metadata)

    def test_memory_round_trip_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            store = MemoryStore.load(memory_path)
            case = CaseProfile(
                case_id="case-1",
                context="Người vi phạm có thể bị phạt tù đến 07 năm.",
                question="Có thể bị phạt tù đến bao nhiêu năm?",
                answer="07 năm",
            )
            result = DebateResult(
                case_id=case.case_id,
                verdict=Verdict(
                    prediction="07 năm",
                    answer="07 năm",
                    confidence=75,
                    reasoning="Dựa trên ngữ cảnh pháp luật.",
                ),
            )

            store.add_regulations(
                [
                    {
                        "id": "article-1",
                        "article_id": "1",
                        "law_name": "Luật mẫu",
                        "source_type": "regulation",
                        "text": "Người vi phạm có thể bị phạt tù.",
                    }
                ]
            )
            store.update_from_debate(case, result)
            store.save()

            reloaded = MemoryStore.load(memory_path)
            similar_case = CaseProfile(
                case_id="case-2",
                context="Một người khác có thể bị phạt tù đến 07 năm.",
                question="Có thể bị phạt tù đến bao nhiêu năm?",
            )
            context = reloaded.query(similar_case, top_k=3)

            self.assertEqual(len(context.regulations), 1)
            self.assertEqual(len(context.experiences), 1)
            self.assertEqual(len(context.cases), 1)

            same_case_context = reloaded.query(case, top_k=3)
            self.assertEqual(len(same_case_context.cases), 0)


class MemoryLeakPreventionTest(unittest.TestCase):
    """B.3.9: Verify that gold answers never leak into memory entries."""

    def test_sanitize_entry_removes_answer_keys(self) -> None:
        store = MemoryStore(mode="read_update")
        case = CaseProfile(
            case_id="case-x",
            context="Context text here.",
            question="What is the penalty?",
            answer="07 năm",
        )
        entry = {
            "id": "test-1",
            "answer": "07 năm",
            "gold_answer": "07 năm",
            "ground_truth": "07 năm",
            "label": "07 năm",
            "text": "Some text with 07 năm in it",
            "other_field": "safe value",
        }
        sanitized = store._sanitize_entry(entry, case)

        self.assertNotIn("answer", sanitized)
        self.assertNotIn("gold_answer", sanitized)
        self.assertNotIn("ground_truth", sanitized)
        self.assertNotIn("label", sanitized)
        self.assertNotIn("07 năm", sanitized.get("text", ""))
        self.assertIn("[REDACTED_ANSWER]", sanitized.get("text", ""))
        self.assertEqual(sanitized["other_field"], "safe value")

    def test_default_memories_no_gold_answer(self) -> None:
        """_append_default_memories should not persist gold answer."""
        store = MemoryStore(mode="read_update")
        case = CaseProfile(
            case_id="case-y",
            context="Người vi phạm có thể bị phạt tù đến 07 năm.",
            question="Mức phạt tối đa?",
            answer="07 năm",
        )
        result = DebateResult(
            case_id=case.case_id,
            verdict=Verdict(
                prediction="07 năm",
                answer="07 năm",
                confidence=80,
                reasoning="Based on context.",
            ),
        )

        store._append_default_memories(case, result)

        for bucket_name in ("regulations", "experiences", "cases"):
            bucket = getattr(store, bucket_name)
            for entry in bucket:
                entry_json = json.dumps(entry, ensure_ascii=False)
                self.assertNotIn(
                    "07 năm",
                    entry_json,
                    f"Gold answer leaked into {bucket_name}: {entry}",
                )

    def test_extend_from_reflection_sanitizes(self) -> None:
        """_extend_from_reflection must sanitize LLM output even if LLM
        tries to persist the gold answer."""
        store = MemoryStore(mode="read_update")
        case = CaseProfile(
            case_id="case-z",
            context="Test context.",
            question="What is the answer?",
            answer="42 months",
        )
        reflected = {
            "regulations": [
                {"id": "ref-1", "text": "The answer is 42 months"},
            ],
            "experiences": [
                {"id": "exp-1", "answer": "42 months", "text": "experience text"},
            ],
            "cases": [
                {"id": "cas-1", "gold_answer": "42 months", "text": "case text 42 months"},
            ],
        }

        store._extend_from_reflection(case, reflected)

        for bucket_name in ("regulations", "experiences", "cases"):
            bucket = getattr(store, bucket_name)
            for entry in bucket:
                entry_json = json.dumps(entry, ensure_ascii=False)
                self.assertNotIn(
                    "42 months",
                    entry_json,
                    f"Gold answer leaked into {bucket_name} via reflection: {entry}",
                )

    def test_verdict_dict_in_reflection_prompt_does_not_leak(self) -> None:
        """Verify that verdict answer field passed to the reflection prompt
        is handled by _sanitize_entry so stored entries are clean."""
        store = MemoryStore(mode="read_update")
        case = CaseProfile(
            case_id="case-w",
            context="The law says fines up to 500 million.",
            question="Maximum fine?",
            answer="500 triệu",
        )
        # Simulate reflection output that copies verdict answer
        reflected = {
            "experiences": [
                {
                    "id": "exp-w",
                    "text": "Debate concluded with answer 500 triệu",
                    "prediction": "500 triệu",
                },
            ],
        }

        store._extend_from_reflection(case, reflected)

        for entry in store.experiences:
            entry_json = json.dumps(entry, ensure_ascii=False)
            self.assertNotIn(
                "500 triệu",
                entry_json,
                f"Gold answer leaked via prediction field: {entry}",
            )


if __name__ == "__main__":
    unittest.main()
