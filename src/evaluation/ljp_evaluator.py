"""Legal judgment prediction evaluation metrics."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Callable

from src.models import JudgmentGroundTruth, LegalJudgment, LJPEvalResult


def normalize_label(value: str) -> str:
    """Normalize Vietnamese/English legal labels for comparison."""

    text = value.strip().lower().replace("đ", "d").replace("Đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_articles(articles: list[str]) -> set[str]:
    """Normalize article identifiers for set comparison."""

    normalized: set[str] = set()
    for article in articles:
        cleaned = normalize_label(article)
        cleaned = re.sub(r"[^a-z0-9./-]", "", cleaned)
        if cleaned:
            normalized.add(cleaned)
    return normalized


def parse_sentence_years(sentence: str) -> float | None:
    """Extract approximate imprisonment duration in years from free text."""

    text = normalize_label(sentence)
    if "hoan" in text or "mien" in text or "acquittal" in text or "vo toi" in text:
        return 0.0

    month_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(thang|month)", text)
    if month_match:
        return float(month_match.group(1).replace(",", ".")) / 12.0

    year_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(nam|year)", text)
    if year_match:
        return float(year_match.group(1).replace(",", "."))

    bare_number = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if bare_number and ("tu" in text or "prison" in text):
        return float(bare_number.group(1).replace(",", "."))
    return None


def sentence_bucket(years: float | None) -> str:
    """Map sentence length to coarse buckets for bucket accuracy."""

    if years is None:
        return "unknown"
    if years <= 0:
        return "none"
    if years < 1:
        return "under_1y"
    if years < 3:
        return "1_3y"
    if years < 7:
        return "3_7y"
    if years < 15:
        return "7_15y"
    return "15y_plus"


class LJPEvaluator:
    """Compute LJP metrics beyond extractive QA EM/F1."""

    def evaluate(
        self,
        predicted: LegalJudgment,
        ground_truth: JudgmentGroundTruth,
        *,
        valid_evidence_ids: set[str] | None = None,
        hallucination_checker: Callable[..., float] | None = None,
        citation_checker: Callable[..., float] | None = None,
        human_scores: dict[str, float] | None = None,
    ) -> LJPEvalResult:
        """Score one predicted judgment against gold labels."""

        charge_accuracy = float(
            normalize_label(predicted.charge) == normalize_label(ground_truth.charge)
        )

        pred_articles = normalize_articles(predicted.articles)
        gold_articles = normalize_articles(ground_truth.articles)
        if gold_articles:
            article_accuracy = len(pred_articles & gold_articles) / len(gold_articles)
        else:
            article_accuracy = 1.0 if not pred_articles else 0.0

        pred_years = parse_sentence_years(predicted.sentence)
        gold_years = parse_sentence_years(ground_truth.sentence)
        sentence_mae: float | None = None
        sentence_rmse: float | None = None
        bucket_accuracy: float | None = None
        if pred_years is not None and gold_years is not None:
            error = abs(pred_years - gold_years)
            sentence_mae = error
            sentence_rmse = math.sqrt(error * error)
            bucket_accuracy = float(
                sentence_bucket(pred_years) == sentence_bucket(gold_years)
            )

        citation_validity: float | None = None
        if citation_checker is not None:
            citation_validity = float(citation_checker(predicted, ground_truth))
        elif valid_evidence_ids is not None and predicted.cited_evidence_ids:
            valid_hits = [
                evidence_id
                for evidence_id in predicted.cited_evidence_ids
                if evidence_id in valid_evidence_ids
            ]
            citation_validity = len(valid_hits) / len(predicted.cited_evidence_ids)

        hallucination_rate: float | None = None
        if hallucination_checker is not None:
            hallucination_rate = float(hallucination_checker(predicted, ground_truth))

        human = human_scores or {}
        return LJPEvalResult(
            charge_accuracy=charge_accuracy,
            article_accuracy=article_accuracy,
            sentence_mae_years=sentence_mae,
            sentence_rmse_years=sentence_rmse,
            sentence_bucket_accuracy=bucket_accuracy,
            citation_validity=citation_validity,
            hallucination_rate=hallucination_rate,
            objectivity=human.get("objectivity"),
            logic_score=human.get("logic_score"),
            citation_score=human.get("citation_score"),
        )

    def evaluate_batch(
        self,
        predictions: list[LegalJudgment],
        ground_truths: list[JudgmentGroundTruth],
        **kwargs,
    ) -> dict[str, float | None]:
        """Aggregate batch metrics as simple means over available scores."""

        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have equal length.")

        per_case = [
            self.evaluate(predicted, gold, **kwargs)
            for predicted, gold in zip(predictions, ground_truths, strict=True)
        ]
        aggregate: dict[str, float | None] = {}
        metric_names = [
            "charge_accuracy",
            "article_accuracy",
            "sentence_mae_years",
            "sentence_rmse_years",
            "sentence_bucket_accuracy",
            "citation_validity",
            "hallucination_rate",
            "objectivity",
            "logic_score",
            "citation_score",
        ]
        for metric_name in metric_names:
            values = [
                getattr(result, metric_name)
                for result in per_case
                if getattr(result, metric_name) is not None
            ]
            aggregate[metric_name] = (
                sum(values) / len(values) if values else None
            )
        aggregate["n_cases"] = float(len(per_case))
        return aggregate
