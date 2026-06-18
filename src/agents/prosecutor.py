"""Prosecutor agent for courtroom LJP simulation."""

from __future__ import annotations

from src.agents.base_legal_agent import BaseLegalAgent
from src.models import AgentOutput, CourtCase, EvidenceDocument, MemoryContext


class ProsecutorAgent(BaseLegalAgent):
    """Prosecution-side agent emphasizing charges and aggravating factors."""

    def __init__(self, llm, **kwargs) -> None:
        super().__init__(
            role="prosecutor",
            llm=llm,
            use_courtroom_prompts=True,
            **kwargs,
        )

    def present_indictment(
        self,
        court_case: CourtCase,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        transcript: list[AgentOutput],
    ) -> AgentOutput:
        """Opening indictment during courtroom phase 1."""

        return self.generate_turn(
            "prosecutor_strategy.txt",
            "prosecutor_indictment.txt",
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=transcript,
            round_index=0,
        )

    def generate_argument(
        self,
        court_case: CourtCase,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        debate_history: list[AgentOutput],
        round_index: int,
    ) -> AgentOutput:
        """Adversarial prosecution argument during debate phase."""

        return self.generate_turn(
            "prosecutor_strategy.txt",
            "prosecutor_argument.txt",
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=debate_history,
            round_index=round_index,
        )

    def closing_statement(
        self,
        court_case: CourtCase,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        debate_history: list[AgentOutput],
    ) -> AgentOutput:
        """Final prosecution statement before judgment."""

        return self.generate_single_turn(
            "prosecutor_closing.txt",
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=debate_history,
            private_strategy_label="closing_statement",
        )
