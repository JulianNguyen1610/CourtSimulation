"""Debate-side agents for the baseline system."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.llm import LLMClient
from src.models import AgentOutput, CaseProfile, EvidenceDocument, MemoryContext
from src.utils.prompt_compact import (
    compact_agent_view,
    compact_evidence,
    compact_history,
)


PromptRole = Literal["proponent", "opponent"]


class DebateAgent:
    """Agent that prepares a private strategy and a public debate turn."""

    def __init__(
        self,
        role: PromptRole,
        llm: LLMClient,
        prompt_dir: str | Path = "configs/prompts",
        argument_max_tokens: int = 500,
        max_context_chars: int | None = None,
        max_evidence_docs: int | None = None,
        max_evidence_chars: int | None = None,
        max_history_turns: int | None = None,
        max_history_chars: int | None = None,
    ) -> None:
        self.role = role
        self.llm = llm
        self.prompt_dir = Path(prompt_dir)
        self.argument_max_tokens = argument_max_tokens
        self.max_context_chars = max_context_chars
        self.max_evidence_docs = max_evidence_docs
        self.max_evidence_chars = max_evidence_chars
        self.max_history_turns = max_history_turns
        self.max_history_chars = max_history_chars

    def generate_argument(
        self,
        case: CaseProfile,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        debate_history: list[AgentOutput],
        round_index: int,
    ) -> AgentOutput:
        """Generate one private strategy and one public argument."""

        prompt_inputs = self._prompt_inputs(
            case=case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=debate_history,
        )
        strategy_prompt = self._render_prompt(
            self._strategy_template_name,
            round_index=round_index,
            **prompt_inputs,
        )
        private_strategy = self.llm.generate(strategy_prompt)

        argument_prompt = self._render_prompt(
            self._argument_template_name,
            case_profile=prompt_inputs["case_profile"],
            private_strategy=private_strategy,
            debate_history=prompt_inputs["debate_history"],
            round_index=round_index,
            argument_max_tokens=self.argument_max_tokens,
        )
        public_argument = self.llm.generate(argument_prompt)

        return AgentOutput(
            role=self.role,
            round_index=round_index,
            private_strategy=private_strategy,
            public_argument=public_argument,
            evidence_ids=[document.doc_id for document in legal_evidence],
            memory_ids=self._memory_ids(past_memory),
        )

    def generate_closing_statement(
        self,
        case: CaseProfile,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        debate_history: list[AgentOutput],
    ) -> AgentOutput:
        """Generate a final closing statement before the judge verdict."""

        prompt_inputs = self._prompt_inputs(
            case=case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=debate_history,
        )
        prompt = self._render_prompt(
            self._closing_template_name,
            argument_max_tokens=self.argument_max_tokens,
            **prompt_inputs,
        )
        closing_statement = self.llm.generate(prompt)
        return AgentOutput(
            role=self.role,
            round_index=0,
            private_strategy="closing_statement",
            public_argument=closing_statement,
            evidence_ids=[document.doc_id for document in legal_evidence],
            memory_ids=self._memory_ids(past_memory),
        )

    @property
    def _strategy_template_name(self) -> str:
        return f"{self.role}_strategy.txt"

    @property
    def _argument_template_name(self) -> str:
        if self.role == "proponent":
            return "proponent_argument.txt"
        return "opponent_rebuttal.txt"

    @property
    def _closing_template_name(self) -> str:
        return f"closing_{self.role}.txt"

    def _prompt_inputs(
        self,
        case: CaseProfile,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        debate_history: list[AgentOutput],
    ) -> dict[str, object]:
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
            "case_profile": compact_agent_view(
                case.agent_view(),
                self.max_context_chars,
            ),
            "legal_evidence": evidence_payload,
            "past_memory": past_memory.model_dump(),
            "debate_history": history_payload,
        }

    def _render_prompt(self, template_name: str, **values: object) -> str:
        template_path = self.prompt_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        return template.format(**values)

    @staticmethod
    def _memory_ids(memory_context: MemoryContext) -> list[str]:
        memory_ids: list[str] = []
        for bucket in (
            memory_context.regulations,
            memory_context.experiences,
            memory_context.cases,
        ):
            memory_ids.extend(
                str(entry["id"]) for entry in bucket if "id" in entry
            )
        return memory_ids
