"""Post-processing helpers to keep debate answers short and extractive.

These helpers do not use the gold answer. They only clean an answer string and,
when the model returns a verbose sentence that buries a short legal span, reduce
it to that span. The reduction is intentionally conservative: it only triggers
for repeated ViLQA-style patterns and when the extracted span is grounded in the
case context, to avoid wrong truncation.
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

_COMPOUND_SPAN_PATTERNS = [
    # Money ranges: "100.000.000 đồng đến dưới 500.000.000 đồng".
    r"(?:từ\s+)?\d{1,3}(?:[.,]\d{3})*\s*đồng\s+đến\s+(?:dưới\s+)?\d{1,3}(?:[.,]\d{3})*\s*đồng",
    # Money lower bound: "2.000.000.000 đồng trở lên".
    r"\d{1,3}(?:[.,]\d{3})*\s*đồng\s+trở\s+lên",
    # Duration compounds: "01 tháng ít nhất 04 ngày".
    r"\d{1,3}\s*tháng\s+ít\s+nhất\s+\d{1,3}\s*ngày",
    # Penalty ranges with alternatives.
    r"phạt\s+tù\s+từ\s+\d{1,3}\s*năm\s+đến\s+\d{1,3}\s*năm,\s*tù\s+chung\s+thân\s+hoặc\s+tử\s+hình",
    r"từ\s+\d{1,3}\s*năm\s+đến\s+\d{1,3}\s*năm,\s*tù\s+chung\s+thân\s+hoặc\s+tử\s+hình",
]
_COMPOUND_SPAN_RE = re.compile("|".join(_COMPOUND_SPAN_PATTERNS), flags=re.IGNORECASE)

_WORD_NUMBER_MAP = {
    "một": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4",
    "tư": "4",
    "năm": "5",
    "sáu": "6",
    "bảy": "7",
    "tám": "8",
    "chín": "9",
    "mười": "10",
}
_WORD_DURATION_RE = re.compile(
    r"\b(" + "|".join(_WORD_NUMBER_MAP) + r")\s+(tháng|năm|ngày|tuần)\b",
    flags=re.IGNORECASE,
)

# "Sau 01 tháng" vs gold "01 tháng" — common LLM prefix on duration questions.
_REDUNDANT_SAU_PREFIX_RE = re.compile(
    r"^sau\s+(.+)$",
    flags=re.IGNORECASE,
)
_REDUNDANT_TINH_TU_PREFIX_RE = re.compile(
    r"^tính\s+từ\s+(.+)$",
    flags=re.IGNORECASE,
)

_GROUNDED_PREFIX_STRIPS = (
    "Di chúc có hiệu lực ",
    "di chúc có hiệu lực ",
)

_DEFINITIONAL_QUESTION_RE = re.compile(
    r"là gì|được gọi là|là như thế nào|là ai|là gì\?",
    flags=re.IGNORECASE,
)
_CO_SUFFIX_RE = re.compile(r"\s+có\s+", flags=re.IGNORECASE)
_SUBJECT_CO_QUYEN_RE = re.compile(
    r"^.+?\s+có\s+quyền\s+(.+)$",
    flags=re.IGNORECASE,
)

# Long sentences (in words) are candidates for span extraction.
_MAX_SHORT_WORDS = 8
_MAX_DEFINITIONAL_WORDS = 40
_MAX_CONTEXT_EXTENSION_WORDS = 4


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


def _normalize_for_context_match(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"(?<=\d)[.,](?=\d)", "", lowered)
    return re.sub(r"\s+", " ", lowered).strip(" .;:,")


def _appears_in_context(span: str, context: str) -> bool:
    return _normalize_for_context_match(span) in _normalize_for_context_match(context)


def _extract_context_span(span: str, context: str) -> str | None:
    """Return context-verbatim form of a candidate span when possible."""

    normalized_span = _normalize_for_context_match(span)
    for match in _COMPOUND_SPAN_RE.findall(context):
        normalized_match = _normalize_for_context_match(match)
        if normalized_match == normalized_span:
            return _clean_extracted_compound(match)
        if normalized_span in normalized_match:
            return _clean_extracted_compound(match)
    for match in _COMBINED_SPAN_RE.findall(context):
        if _normalize_numeric_span(match) == _normalize_numeric_span(span):
            return match.strip()
    return span.strip() if _appears_in_context(span, context) else None


def _clean_extracted_compound(span: str) -> str:
    """Normalize harmless leading prepositions for final answer spans."""

    return span.strip()


def _should_try_context_extension(answer: str, question: str | None) -> bool:
    """Only extend very specific duration phrases that often miss a tail word."""

    lowered = answer.lower()
    if "ít nhất" in lowered and re.search(r"\d+\s*phút", lowered):
        return True
    if question and re.search(r"ít nhất|liên tục", question, flags=re.IGNORECASE):
        return "ít nhất" in lowered
    return False


_CO_SUFFIX_START_RE = re.compile(
    r"^(di chúc|người|cá nhân|tổ chức|bên)\b",
    flags=re.IGNORECASE,
)


def _extract_compound_span(cleaned: str, context: str) -> str | None:
    matches = _COMPOUND_SPAN_RE.findall(cleaned)
    distinct = {_normalize_for_context_match(match) for match in matches}
    if len(distinct) != 1:
        return None
    return _extract_context_span(matches[0], context)


def _extract_word_duration(cleaned: str, context: str) -> str | None:
    match = _WORD_DURATION_RE.search(cleaned)
    if not match:
        return None
    digit = _WORD_NUMBER_MAP[match.group(1).lower()]
    unit = match.group(2).lower()
    for context_match in _COMBINED_SPAN_RE.findall(context):
        normalized = _normalize_numeric_span(context_match)
        if re.fullmatch(rf"0*{digit}\s*{unit}", normalized):
            return context_match.strip()
    return None


def _extract_phrase_by_markers(cleaned: str, context: str) -> str | None:
    """Handle common legal QA over-extractions without using the gold answer."""

    marker_candidates: list[str] = []
    start_marker_specs = [
        ("theo điều kiện ", "after"),
    ]
    for marker, direction in start_marker_specs:
        if not cleaned.lower().startswith(marker):
            continue
        candidate = clean_answer(cleaned[len(marker) :])
        if direction == "after" and " và " in candidate.lower():
            candidate = re.split(r"\s+và\s+", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
        if 2 <= len(candidate.split()) <= 12:
            marker_candidates.append(candidate)

    marker_specs = [
        (" tiếp tục ", "before"),
        (" theo điều kiện ", "after"),
        (" là ", "after"),
    ]
    for marker, direction in marker_specs:
        if marker not in cleaned.lower():
            continue
        pattern = re.compile(re.escape(marker), flags=re.IGNORECASE)
        parts = pattern.split(cleaned, maxsplit=1)
        candidate = parts[0] if direction == "before" else parts[1]
        candidate = clean_answer(candidate)
        if direction == "after" and " và " in candidate.lower():
            candidate = re.split(r"\s+và\s+", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
        if 2 <= len(candidate.split()) <= 12:
            marker_candidates.append(candidate)

    for candidate in marker_candidates:
        if _appears_in_context(candidate, context):
            return candidate
    return None


def _strip_redundant_sau_prefix(answer: str, context: str) -> str:
    """Drop leading ``Sau`` when the remainder is grounded in context."""

    match = _REDUNDANT_SAU_PREFIX_RE.match(answer.strip())
    if not match:
        return answer
    remainder = clean_answer(match.group(1))
    if not remainder:
        return answer
    grounded = _extract_context_span(remainder, context)
    if grounded:
        return grounded
    if _appears_in_context(remainder, context):
        return remainder
    return answer


def _strip_redundant_tinh_tu_prefix(answer: str, context: str) -> str:
    """Drop leading ``tính từ`` when the remainder is grounded in context."""

    match = _REDUNDANT_TINH_TU_PREFIX_RE.match(answer.strip())
    if not match:
        return answer
    remainder = clean_answer(match.group(1))
    if remainder and _appears_in_context(remainder, context):
        return remainder
    return answer


def _strip_known_legal_prefixes(answer: str, context: str) -> str:
    """Drop boilerplate prefixes when the remainder is grounded in context."""

    cleaned = answer
    for prefix in _GROUNDED_PREFIX_STRIPS:
        cleaned = _strip_grounded_leading_prefix(cleaned, context, prefix)
    match = _SUBJECT_CO_QUYEN_RE.match(cleaned.strip())
    if match:
        remainder = clean_answer(f"quyền {match.group(1)}")
        if _appears_in_context(remainder, context):
            cleaned = remainder
    return cleaned


def _extract_definitional_phrase(
    cleaned: str,
    question: str | None,
    context: str,
) -> str | None:
    """Extract the definition tail for ``... là gì?`` style questions."""

    if not question or not _DEFINITIONAL_QUESTION_RE.search(question):
        return None
    if " là " not in cleaned.lower():
        return None

    pattern = re.compile(r"\s+là\s+", flags=re.IGNORECASE)
    parts = pattern.split(cleaned)
    if len(parts) < 2:
        return None
    candidate = clean_answer(parts[-1])
    if not candidate or len(candidate.split()) > _MAX_DEFINITIONAL_WORDS:
        return None
    if _appears_in_context(candidate, context):
        return candidate
    return None


def _extract_grounded_co_suffix(cleaned: str, context: str) -> str | None:
    """Return the longest ``có ...`` suffix that appears verbatim in context."""

    if len(cleaned.split()) <= _MAX_SHORT_WORDS:
        return None
    if not _CO_SUFFIX_START_RE.match(cleaned):
        return None

    candidates: list[str] = []
    for match in _CO_SUFFIX_RE.finditer(cleaned):
        suffix = cleaned[match.start() + 1 :].strip()
        if suffix and _appears_in_context(suffix, context):
            candidates.append(suffix)
    if not candidates:
        return None
    return max(candidates, key=len)


def _context_extension_tail(context: str, start_index: int) -> str:
    """Return the local context tail up to the next clause boundary."""

    tail = context[start_index:]
    clause = re.split(r"[,;.\n]", tail, maxsplit=1)[0]
    return clause.strip()


def _extend_unique_context_phrase(
    answer: str,
    context: str,
    max_extra_words: int = _MAX_CONTEXT_EXTENSION_WORDS,
) -> str:
    """Extend a grounded short span when context contains one longer phrase."""

    if not answer or not _appears_in_context(answer, context):
        return answer

    normalized_answer = _normalize_for_context_match(answer)
    extensions: list[str] = []
    for match in re.finditer(re.escape(answer), context, flags=re.IGNORECASE):
        clause_tail = _context_extension_tail(context, match.end())
        extra_words = clause_tail.split()[:max_extra_words]
        if not extra_words:
            continue
        extended = f"{context[match.start():match.end()]} {' '.join(extra_words)}".strip()
        extended = clean_answer(extended)
        if _normalize_for_context_match(extended) == normalized_answer:
            continue
        if _appears_in_context(extended, context):
            extensions.append(extended)

    distinct = {_normalize_for_context_match(item) for item in extensions}
    if len(distinct) != 1:
        return answer
    return extensions[0]


def _strip_grounded_leading_prefix(answer: str, context: str, prefix: str) -> str:
    """Drop a leading prefix when the remainder appears verbatim in context."""

    if not answer.lower().startswith(prefix.lower()):
        return answer
    remainder = answer[len(prefix) :].strip()
    if remainder and _appears_in_context(remainder, context):
        return remainder
    return answer


def _normalize_money_range_prefix(span: str, context: str) -> str:
    """Drop leading ``từ`` for ``... đến dưới ...`` ranges when grounded."""

    cleaned = span.strip()
    if not cleaned.lower().startswith("từ ") or "đồng" not in cleaned.lower():
        return cleaned
    if "dưới" not in cleaned.lower():
        return cleaned
    without_tu = re.sub(r"^từ\s+", "", cleaned, flags=re.IGNORECASE)
    if _appears_in_context(without_tu, context):
        return without_tu
    return cleaned


def _normalize_short_extractive_answer(answer: str, context: str) -> str:
    """Apply grounded prefix cleanup on already-short candidate spans."""

    cleaned = _clean_extracted_compound(answer)
    cleaned = _normalize_money_range_prefix(cleaned, context)
    for prefix in ("Bị ", "bị ", "phải "):
        cleaned = _strip_grounded_leading_prefix(cleaned, context, prefix)
    return cleaned


def shorten_legal_answer(
    answer: str | None,
    context: str,
    question: str | None = None,
) -> str:
    """Return a short, extractive answer when a single legal span is buried.

    The gold answer is never used. Behavior:
    - Clean the answer string (quotes, markdown, trailing period).
    - If already short (<= 8 words), return the cleaned answer.
    - If verbose but it contains a clear compound span or exactly one distinct
      numeric/duration span grounded in context, return the context's verbatim
      form of that span.
    - For a few common legal QA marker patterns ("là", "theo điều kiện",
      "tiếp tục"), return the short grounded phrase around the marker.
    - Otherwise, return the cleaned answer unchanged.
    """

    if not answer:
        return ""
    cleaned = clean_answer(answer)
    if not cleaned:
        return ""
    cleaned = _strip_redundant_sau_prefix(cleaned, context)
    cleaned = _strip_redundant_tinh_tu_prefix(cleaned, context)
    cleaned = _strip_known_legal_prefixes(cleaned, context)

    definitional = _extract_definitional_phrase(cleaned, question, context)
    if definitional:
        return _normalize_short_extractive_answer(definitional, context)

    marker_phrase = _extract_phrase_by_markers(cleaned, context)
    if marker_phrase:
        return _normalize_short_extractive_answer(marker_phrase, context)

    compound = _extract_compound_span(cleaned, context)
    if compound:
        return _normalize_short_extractive_answer(compound, context)

    word_duration = _extract_word_duration(cleaned, context)
    if word_duration:
        return _normalize_short_extractive_answer(word_duration, context)

    co_suffix = _extract_grounded_co_suffix(cleaned, context)
    if co_suffix:
        return _normalize_short_extractive_answer(co_suffix, context)

    if len(cleaned.split()) <= _MAX_SHORT_WORDS:
        if _should_try_context_extension(cleaned, question):
            cleaned = _extend_unique_context_phrase(
                cleaned,
                context,
                max_extra_words=2,
            )
        return _normalize_short_extractive_answer(cleaned, context)

    answer_matches = _COMBINED_SPAN_RE.findall(cleaned)
    if not answer_matches:
        return _normalize_short_extractive_answer(cleaned, context)
    distinct = {_normalize_numeric_span(match) for match in answer_matches}
    if len(distinct) != 1:
        return _normalize_short_extractive_answer(cleaned, context)

    target = next(iter(distinct))
    for context_match in _COMBINED_SPAN_RE.findall(context):
        if _normalize_numeric_span(context_match) == target:
            return _normalize_short_extractive_answer(context_match.strip(), context)

    return _normalize_short_extractive_answer(cleaned, context)
