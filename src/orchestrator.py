"""Debate orchestration for the baseline multi-agent system."""

from __future__ import annotations

from src.agents.debate_agent import DebateAgent
from src.agents.judge_agent import JudgeAgent
from src.memory.memory_store import MemoryStore
from src.models import AgentOutput, DebateResult, EvidenceDocument, MemoryContext, CaseProfile
from src.retrieval.legal_retriever import LegalRetriever


class DebateOrchestrator:
    """Run an n-round Proponent/Opponent/Judge debate."""

    def __init__(
        self,
        proponent: DebateAgent,
        opponent: DebateAgent,
        judge: JudgeAgent,
        rounds: int = 3,
        legal_retriever: LegalRetriever | None = None,
        memory_store: MemoryStore | None = None,
        evidence_top_k: int = 5,
        memory_top_k: int = 5,
        include_closing_statements: bool = True,
        enable_judge_question: bool = False,
        early_stop_confidence: float | None = None,
    ) -> None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1.")
        self.proponent = proponent
        self.opponent = opponent
        self.judge = judge
        self.rounds = rounds
        self.legal_retriever = legal_retriever
        self.memory_store = memory_store
        self.evidence_top_k = evidence_top_k
        self.memory_top_k = memory_top_k
        self.include_closing_statements = include_closing_statements
        self.enable_judge_question = enable_judge_question
        self.early_stop_confidence = early_stop_confidence

    def run(
        self,
        case: CaseProfile,
        legal_evidence: list[EvidenceDocument] | None = None,
        past_memory: MemoryContext | None = None,
    ) -> DebateResult:
        """Run a complete debate and return a structured artifact."""

        self.judge.reset()
        transcript: list[AgentOutput] = []
        memory_context = past_memory or self._retrieve_memory(case)
        evidence = legal_evidence or self._retrieve_evidence(case)

        for round_index in range(1, self.rounds + 1):
            proponent_output = self.proponent.generate_argument(
                case=case,
                legal_evidence=evidence,
                past_memory=memory_context,
                debate_history=transcript,
                round_index=round_index,
            )
            transcript.append(proponent_output)

            opponent_output = self.opponent.generate_argument(
                case=case,
                legal_evidence=evidence,
                past_memory=memory_context,
                debate_history=transcript,
                round_index=round_index,
            )
            transcript.append(opponent_output)

            self.judge.update_belief(
                case=case,
                proponent_argument=proponent_output,
                opponent_argument=opponent_output,
                round_index=round_index,
            )
            latest_belief = self.judge.belief_history[-1]
            if (
                self.early_stop_confidence is not None
                and latest_belief.confidence >= self.early_stop_confidence
            ):
                break

        if self.enable_judge_question:
            question = self.judge.ask_follow_up(case=case, transcript=transcript)
            if question:
                transcript.append(
                    AgentOutput(
                        role="judge",
                        round_index=len(self.judge.belief_history) + 1,
                        private_strategy="follow_up_question",
                        public_argument=question,
                    )
                )

        if self.include_closing_statements:
            proponent_closing = self.proponent.generate_closing_statement(
                case=case,
                legal_evidence=evidence,
                past_memory=memory_context,
                debate_history=transcript,
            )
            transcript.append(proponent_closing)
            opponent_closing = self.opponent.generate_closing_statement(
                case=case,
                legal_evidence=evidence,
                past_memory=memory_context,
                debate_history=transcript,
            )
            transcript.append(opponent_closing)

        verdict = self.judge.render_verdict(case=case, transcript=transcript)
        return DebateResult(
            case_id=case.case_id,
            legal_evidence=evidence,
            memory_context=memory_context,
            transcript=transcript,
            belief_history=list(self.judge.belief_history),
            verdict=verdict,
        )

    def _retrieve_evidence(self, case: CaseProfile) -> list[EvidenceDocument]:
        if self.legal_retriever is None:
            return self._default_context_evidence(case)
        if getattr(self.legal_retriever, "method", None) == "off":
            return []
        retrieved = self.legal_retriever.retrieve(
            case.retrieval_query,
            top_k=self.evidence_top_k,
        )
        if not retrieved:
            return self._default_context_evidence(case)
        return retrieved

    def _retrieve_memory(self, case: CaseProfile) -> MemoryContext:
        if self.memory_store is None:
            return MemoryContext()
        return self.memory_store.query(case, top_k=self.memory_top_k)

    @staticmethod
    def _default_context_evidence(case: CaseProfile) -> list[EvidenceDocument]:
        """Use the provided QA context as evidence until retrieval is added."""

        return [
            EvidenceDocument(
                doc_id=f"{case.case_id}-context",
                text=case.context,
                source="input_context",
                score=1.0,
            )
        ]
