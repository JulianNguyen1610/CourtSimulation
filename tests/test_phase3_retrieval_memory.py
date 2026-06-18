"""Sanity checks for Phase 3 retrieval and memory components."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
