"""Courtroom turn-order protocol for Phase 3 LJP simulation."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.defendant import DefendantAgent
from src.agents.defense import DefenseAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.prosecutor import ProsecutorAgent
from src.models import AgentOutput, CourtCase, EvidenceDocument, MemoryContext


@dataclass(frozen=True)
class ProtocolConfig:
    """Configurable courtroom phase switches."""

    enable_opening: bool = True
    enable_debate: bool = True
    max_debate_rounds: int = 2
    enable_judge_question: bool = False
    enable_closing: bool = True
    enable_deliberation: bool = True
    early_stop_confidence: float | None = None


class CourtroomProtocol:
    """Defines ordered turns for opening, debate, and judgment phases."""

    def __init__(self, config: ProtocolConfig | None = None) -> None:
        self.config = config or ProtocolConfig()

    def opening(
        self,
        court_case: CourtCase,
        judge: JudgeAgent,
        prosecutor: ProsecutorAgent,
        defendant: DefendantAgent,
        defense: DefenseAgent,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        transcript: list[AgentOutput],
    ) -> tuple[list[AgentOutput], list[str]]:
        """Run opening phase: judge, indictment, testimony, defense opening."""

        if not self.config.enable_opening:
            return [], []

        turns: list[AgentOutput] = []
        phases: list[str] = []

        judge_opening = judge.open_session(court_case)
        turns.append(judge_opening)
        phases.append("opening_judge")

        indictment = prosecutor.present_indictment(
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            transcript=transcript + turns,
        )
        turns.append(indictment)
        phases.append("opening_prosecutor")

        testimony = defendant.testify(
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            transcript=transcript + turns,
        )
        turns.append(testimony)
        phases.append("opening_defendant")

        defense_opening = defense.opening_statement(
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            transcript=transcript + turns,
        )
        turns.append(defense_opening)
        phases.append("opening_defense")

        return turns, phases

    def debate_round(
        self,
        court_case: CourtCase,
        prosecutor: ProsecutorAgent,
        defense: DefenseAgent,
        judge: JudgeAgent,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        transcript: list[AgentOutput],
        round_index: int,
    ) -> tuple[list[AgentOutput], list[str]]:
        """Run one adversarial debate round with optional judge question."""

        if not self.config.enable_debate:
            return [], []

        turns: list[AgentOutput] = []
        phases = [f"debate_round_{round_index}"]

        prosecution_turn = prosecutor.generate_argument(
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=transcript + turns,
            round_index=round_index,
        )
        turns.append(prosecution_turn)

        defense_turn = defense.generate_argument(
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=transcript + turns,
            round_index=round_index,
        )
        turns.append(defense_turn)

        judge.update_courtroom_belief(
            court_case=court_case,
            prosecutor_argument=prosecution_turn,
            defense_argument=defense_turn,
            round_index=round_index,
        )
        phases.append(f"debate_belief_{round_index}")

        if self.config.enable_judge_question:
            question = judge.ask_courtroom_question(
                court_case=court_case,
                transcript=transcript + turns,
            )
            if question:
                turns.append(
                    AgentOutput(
                        role="judge",
                        round_index=round_index,
                        private_strategy="follow_up_question",
                        public_argument=question,
                    )
                )
                phases.append(f"debate_judge_question_{round_index}")

        return turns, phases

    def closing(
        self,
        court_case: CourtCase,
        prosecutor: ProsecutorAgent,
        defense: DefenseAgent,
        legal_evidence: list[EvidenceDocument],
        past_memory: MemoryContext,
        transcript: list[AgentOutput],
    ) -> tuple[list[AgentOutput], list[str]]:
        """Run closing statements before judgment phase."""

        if not self.config.enable_closing:
            return [], []

        turns: list[AgentOutput] = []
        phases = ["closing"]

        prosecutor_closing = prosecutor.closing_statement(
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=transcript + turns,
        )
        turns.append(prosecutor_closing)
        phases.append("closing_prosecutor")

        defense_closing = defense.closing_statement(
            court_case=court_case,
            legal_evidence=legal_evidence,
            past_memory=past_memory,
            debate_history=transcript + turns,
        )
        turns.append(defense_closing)
        phases.append("closing_defense")

        return turns, phases
