"""Evaluation metrics for ViLQA-style extractive QA."""

from __future__ import annotations

import re
import string
from collections import Counter

from src.models import CaseProfile, EvalResult, Verdict


VIETNAMESE_PUNCTUATION = string.punctuation + "“”‘’…–—"


def normalize_answer(text: str | None) -> str:
    """Normalize an answer for lightweight exact-match/F1 evaluation."""

    if text is None:
        return ""
    lowered = text.lower().strip()
    lowered = lowered.translate(str.maketrans("", "", VIETNAMESE_PUNCTUATION))
    return " ".join(lowered.split())


def exact_match_score(prediction: str | None, gold_answer: str | None) -> float:
    """Return 1.0 when normalized prediction equals normalized gold."""

    return float(normalize_answer(prediction) == normalize_answer(gold_answer))


def token_f1_score(prediction: str | None, gold_answer: str | None) -> float:
    """Compute token-level F1 for short extractive answers."""

    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_gold_answer(context: str, gold_answer: str | None) -> bool:
    """Check whether an extractive gold answer appears in the context."""

    if not gold_answer:
        return False
    pattern = re.escape(normalize_answer(gold_answer))
    return re.search(pattern, normalize_answer(context)) is not None


class ViLQAEvaluator:
    """Evaluator for answer predictions on ViLQA/ALQAC."""

    def evaluate_answer(
        self,
        case: CaseProfile,
        predicted_answer: str | None,
    ) -> EvalResult:
        """Evaluate a predicted answer against the case gold answer."""

        if case.answer is None:
            raise ValueError(f"Case {case.case_id} does not have a gold answer.")

        exact_match = exact_match_score(predicted_answer, case.answer)
        f1 = token_f1_score(predicted_answer, case.answer)
        answer_in_context = contains_gold_answer(case.context, case.answer)
        return EvalResult(
            exact_match=exact_match,
            f1=f1,
            legal_accuracy=None,
            argument_quality=None,
            logical_consistency=None,
            notes=(
                "Gold answer appears in context."
                if answer_in_context
                else "Gold answer not found in normalized context."
            ),
        )

    def evaluate_verdict(self, case: CaseProfile, verdict: Verdict) -> EvalResult:
        """Evaluate a judge verdict using its answer field."""

        return self.evaluate_answer(case, verdict.answer or verdict.prediction)
