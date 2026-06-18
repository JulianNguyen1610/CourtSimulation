"""Shared base class for Phase 1 debate and Phase 3 courtroom agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.llm import LLMClient
from src.models import (
    AgentOutput,
    AgentRole,
    CaseProfile,
    CourtCase,
    EvidenceDocument,
    MemoryContext,
)
from src.utils.prompt_compact import (
    compact_agent_view,
    compact_evidence,
    compact_history,
)


class BaseLegalAgent:
    """Base agent with prompt rendering and compact context helpers."""

    def __init__(
        self,
        role: AgentRole,
        llm: LLMClient,
        prompt_dir: str | Path = "configs/prompts",
        *,
        use_courtroom_prompts: bool = False,
        argument_max_tokens: int = 500,
        max_context_chars: int | None = None,
        max_evidence_docs: int | None = None,
        max_evidence_chars: int | None = None,
        max_history_turns: int | None = None,
        max_history_chars: int | None = None,
    ) -> None:
        self.role = role
        self.llm = llm
        self.prompt_root = Path(prompt_dir)
        self.use_courtroom_prompts = use_courtroom_prompts
        self.argument_max_tokens = argument_max_tokens
        self.max_context_chars = max_context_chars
        self.max_evidence_docs = max_evidence_docs
        self.max_evidence_chars = max_evidence_chars
        self.max_history_turns = max_history_turns
        self.max_history_chars = max_history_chars

    @property
    def prompt_dir(self) -> Path:
        if self.use_courtroom_prompts:
            return self.prompt_root / "courtroom"
        return self.prompt_root

    def generate_turn(
        self,
        strategy_template: str,
        public_template: str,
        *,
        case: CaseProfile | None = None,
        court_case: CourtCase | None = None,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        debate_history: list[AgentOutput],
        round_index: int,
        public_template_kwargs: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Generate private strategy and public statement from two templates."""

        prompt_inputs = self._prompt_inputs(
            case=case,
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=debate_history,
        )
        strategy_prompt = self._render_prompt(
            strategy_template,
            round_index=round_index,
            **prompt_inputs,
        )
        private_strategy = self.llm.generate(strategy_prompt)

        public_kwargs = {
            **prompt_inputs,
            "private_strategy": private_strategy,
            "round_index": round_index,
            "argument_max_tokens": self.argument_max_tokens,
        }
        if public_template_kwargs:
            public_kwargs.update(public_template_kwargs)
        public_prompt = self._render_prompt(public_template, **public_kwargs)
        public_argument = self.llm.generate(public_prompt)

        return AgentOutput(
            role=self.role,
            round_index=round_index,
            private_strategy=private_strategy,
            public_argument=public_argument,
            evidence_ids=[document.doc_id for document in legal_evidence],
            memory_ids=self._memory_ids(past_memory),
        )

    def generate_single_turn(
        self,
        template_name: str,
        *,
        case: CaseProfile | None = None,
        court_case: CourtCase | None = None,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        debate_history: list[AgentOutput],
        round_index: int = 0,
        private_strategy_label: str = "single_turn",
        extra_template_kwargs: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Generate one public turn from a single template."""

        prompt_inputs = self._prompt_inputs(
            case=case,
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=debate_history,
        )
        template_kwargs = {
            **prompt_inputs,
            "round_index": round_index,
            "argument_max_tokens": self.argument_max_tokens,
        }
        if extra_template_kwargs:
            template_kwargs.update(extra_template_kwargs)
        public_argument = self.llm.generate(self._render_prompt(template_name, **template_kwargs))
        return AgentOutput(
            role=self.role,
            round_index=round_index,
            private_strategy=private_strategy_label,
            public_argument=public_argument,
            evidence_ids=[document.doc_id for document in legal_evidence],
            memory_ids=self._memory_ids(past_memory),
        )

    def _prompt_inputs(
        self,
        *,
        case: CaseProfile | None,
        court_case: CourtCase | None,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        debate_history: list[AgentOutput],
    ) -> dict[str, object]:
        if court_case is not None:
            case_view = compact_agent_view(
                court_case.agent_view(),
                self.max_context_chars,
            )
        elif case is not None:
            case_view = compact_agent_view(case.agent_view(), self.max_context_chars)
        else:
            raise ValueError("Either case or court_case must be provided.")

        evidence_payload = compact_evidence(
            [document.model_dump() for document in legal_evidence],
            self.max_evidence_docs,
            self.max_evidence_chars,
        )
        history_payload = compact_history(
            [turn.model_dump() for turn in debate_history],
            self.max_history_turns,
            self.max_history_chars,
        )
        return {
            "case_profile": case_view,
            "legal_evidence": evidence_payload,
            "past_memory": past_memory.model_dump(),
            "debate_history": history_payload,
        }

    def _render_prompt(self, template_name: str, **values: object) -> str:
        template_path = self.prompt_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        return template_path.read_text(encoding="utf-8").format(**values)

    @staticmethod
    def _memory_ids(memory_context: MemoryContext) -> list[str]:
        memory_ids: list[str] = []
        for bucket in (
            memory_context.regulations,
            memory_context.experiences,
            memory_context.cases,
        ):
            memory_ids.extend(str(entry["id"]) for entry in bucket if "id" in entry)
        return memory_ids
