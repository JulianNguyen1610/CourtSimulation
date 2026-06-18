"""Defendant agent for first-person courtroom testimony."""

from __future__ import annotations

from src.agents.base_legal_agent import BaseLegalAgent
from src.models import AgentOutput, CourtCase, EvidenceDocument, MemoryContext


class DefendantAgent(BaseLegalAgent):
    """Defendant agent providing first-person testimony."""

    def __init__(self, llm, **kwargs) -> None:
        super().__init__(
            role="defendant",
            llm=llm,
            use_courtroom_prompts=True,
            **kwargs,
        )

    def testify(
        self,
        court_case: CourtCase,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        transcript: list[AgentOutput],
    ) -> AgentOutput:
        """Deliver first-person testimony during opening phase."""

        return self.generate_single_turn(
            "defendant_testimony.txt",
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=transcript,
            round_index=0,
            private_strategy_label="testimony",
        )
