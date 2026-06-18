"""Helpers to keep local LLM prompts within practical context limits."""

from __future__ import annotations

from typing import Any


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def compact_agent_view(case_view: dict[str, Any], max_context_chars: int | None) -> dict[str, Any]:
    if not max_context_chars:
        return case_view
    view = dict(case_view)
    context = view.get("context")
    if isinstance(context, str):
        view["context"] = truncate_text(context, max_context_chars)
    return view


def compact_evidence(
    documents: list[dict[str, Any]],
    max_docs: int | None,
    max_chars: int | None,
) -> list[dict[str, Any]]:
    selected = documents[:max_docs] if max_docs else documents
    if not max_chars:
        return selected
    compacted = []
    for document in selected:
        item = dict(document)
        item["text"] = truncate_text(str(item.get("text", "")), max_chars)
        compacted.append(item)
    return compacted


def compact_history(
    history: list[dict[str, Any]],
    max_turns: int | None,
    max_chars: int | None,
) -> list[dict[str, Any]]:
    selected = history[-max_turns:] if max_turns else history
    if not max_chars:
        return selected
    compacted = []
    for turn in selected:
        item = dict(turn)
        argument = item.get("public_argument")
        if isinstance(argument, str):
            item["public_argument"] = truncate_text(argument, max_chars)
        compacted.append(item)
    return compacted
