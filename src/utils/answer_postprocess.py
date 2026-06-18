"""Post-processing helpers to keep debate answers short and extractive.

These helpers do not use the gold answer. They only clean an answer string and,
when the model returns a verbose sentence that buries a single short legal span
(money/duration/age), reduce it to that span. The reduction is intentionally
conservative: it only triggers when exactly one distinct numeric span is present
and that span also appears in the case context, to avoid wrong truncation.
"""

from __future__ import annotations

import re

_SURROUNDING_CHARS = " \t\n\r\"'`*"

# Short legal spans frequently used as ViLQA answers.
_LEGAL_SPAN_PATTERNS = [
    r"\d{1,3}(?:[.,]\d{3})+\s*đồng",
    r"\d+\s*đồng",
    r"\d{1,3}\s*năm",
    r"\d{1,3}\s*tháng",
    r"\d{1,3}\s*ngày",
    r"\d{1,3}\s*tuần",
    r"\d{1,3}\s*tuổi",
]
_COMBINED_SPAN_RE = re.compile("|".join(_LEGAL_SPAN_PATTERNS), flags=re.IGNORECASE)

# Long sentences (in words) are candidates for span extraction.
_MAX_SHORT_WORDS = 8


def clean_answer(answer: str) -> str:
    """Strip surrounding quotes/markdown/whitespace and a trailing period."""

    cleaned = answer.strip().strip(_SURROUNDING_CHARS).strip()
    if cleaned.endswith(".") and not re.search(r"\d\.$", cleaned):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def _normalize_numeric_span(span: str) -> str:
    """Normalize a numeric span so digit grouping differences compare equal."""

    lowered = span.lower()
    lowered = re.sub(r"(?<=\d)[.,](?=\d)", "", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def shorten_legal_answer(answer: str | None, context: str) -> str:
    """Return a short, extractive answer when a single legal span is buried.

    The gold answer is never used. Behavior:
    - Clean the answer string (quotes, markdown, trailing period).
    - If already short (<= 8 words), return the cleaned answer.
    - If verbose but it contains exactly one distinct numeric/duration span that
      also appears in the context, return the context's verbatim form of that
      span (preserving original digit grouping and punctuation).
    - Otherwise, return the cleaned answer unchanged.
    """

    if not answer:
        return ""
    cleaned = clean_answer(answer)
    if not cleaned:
        return ""
    if len(cleaned.split()) <= _MAX_SHORT_WORDS:
        return cleaned

    answer_matches = _COMBINED_SPAN_RE.findall(cleaned)
    if not answer_matches:
        return cleaned
    distinct = {_normalize_numeric_span(match) for match in answer_matches}
    if len(distinct) != 1:
        return cleaned

    target = next(iter(distinct))
    for context_match in _COMBINED_SPAN_RE.findall(context):
        if _normalize_numeric_span(context_match) == target:
            return context_match.strip()

    return cleaned
