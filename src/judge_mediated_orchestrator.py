"""Judge-mediated debate orchestration for Phase 1 legal QA."""

from __future__ import annotations

from src.agents.debate_agent import DebateAgent
from src.agents.judge_agent import JudgeAgent
from src.models import AgentOutput, CaseProfile, DebateResult, JudgeControlDecision
from src.orchestrator import DebateOrchestrator


class JudgeMediatedOrchestrator(DebateOrchestrator):
    """Run a debate where the judge chooses each next action.

    Flow: Judge decides → Proponent/Opponent speak → belief update after each
    full round → optional judge question → closing → verdict.
    """

    def __init__(
        self,
        *args,
        max_control_steps: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_control_steps = max_control_steps

    def run(
        self,
        case: CaseProfile,
        legal_evidence=None,
        past_memory=None,
    ) -> DebateResult:
        self.judge.reset()
        transcript: list[AgentOutput] = []
        memory_context = past_memory or self._retrieve_memory(case)
        evidence = legal_evidence or self._retrieve_evidence(case)

        round_index = 0
        completed_rounds = 0
        last_proponent_turn: AgentOutput | None = None
        last_speaker_role: str | None = None
        max_steps = self.max_control_steps or max(self.rounds * 5 + 6, 8)

        for _ in range(max_steps):
            if completed_rounds >= self.rounds:
                if self.include_closing_statements:
                    self._append_closings(case, evidence, memory_context, transcript)
                break

            decision = self._normalize_decision(
                self.judge.decide_control_action(
                    case=case,
                    transcript=transcript,
                    completed_rounds=completed_rounds,
                    max_rounds=self.rounds,
                    last_speaker_role=last_speaker_role,
                ),
                completed_rounds=completed_rounds,
            )
            self._append_control_turn(transcript, decision, round_index)

            if decision.action == "end_debate":
                break

            if decision.action == "request_closing":
                if self.include_closing_statements:
                    self._append_closings(case, evidence, memory_context, transcript)
                break

            if decision.action == "ask_question":
                question = decision.message.strip()
                if question:
                    transcript.append(
                        AgentOutput(
                            role="judge",
                            round_index=round_index,
                            private_strategy="follow_up_question",
                            public_argument=question,
                        )
                    )
                    last_speaker_role = "judge"
                continue

            if decision.action == "call_proponent":
                round_index += 1
                proponent_output = self.proponent.generate_argument(
                    case=case,
                    legal_evidence=evidence,
                    past_memory=memory_context,
                    debate_history=transcript,
                    round_index=round_index,
                )
                transcript.append(proponent_output)
                last_proponent_turn = proponent_output
                last_speaker_role = "proponent"
                continue

            if decision.action == "call_opponent":
                if last_proponent_turn is None:
                    continue
                opponent_output = self.opponent.generate_argument(
                    case=case,
                    legal_evidence=evidence,
                    past_memory=memory_context,
                    debate_history=transcript,
                    round_index=round_index,
                )
                transcript.append(opponent_output)
                last_speaker_role = "opponent"

                self.judge.update_belief(
                    case=case,
                    proponent_argument=last_proponent_turn,
                    opponent_argument=opponent_output,
                    round_index=round_index,
                )
                completed_rounds += 1
                last_proponent_turn = None

                if (
                    self.early_stop_confidence is not None
                    and self.judge.belief_history
                    and self.judge.belief_history[-1].confidence
                    >= self.early_stop_confidence
                ):
                    if self.include_closing_statements:
                        self._append_closings(
                            case, evidence, memory_context, transcript
                        )
                    break

        if self.include_closing_statements and not any(
            turn.private_strategy == "closing_statement" for turn in transcript
        ):
            self._append_closings(case, evidence, memory_context, transcript)

        verdict = self.judge.render_verdict(case=case, transcript=transcript)
        return DebateResult(
            case_id=case.case_id,
            legal_evidence=evidence,
            memory_context=memory_context,
            transcript=transcript,
            belief_history=list(self.judge.belief_history),
            verdict=verdict,
        )

    def _normalize_decision(
        self,
        decision: JudgeControlDecision,
        completed_rounds: int,
    ) -> JudgeControlDecision:
        if completed_rounds < self.rounds:
            return decision
        if decision.action in {"call_proponent", "call_opponent", "ask_question"}:
            if self.include_closing_statements:
                return decision.model_copy(
                    update={
                        "action": "request_closing",
                        "reasoning": (
                            f"{decision.reasoning} "
                            "(orchestrator override: max rounds reached)"
                        ).strip(),
                    }
                )
            return decision.model_copy(
                update={
                    "action": "end_debate",
                    "reasoning": (
                        f"{decision.reasoning} "
                        "(orchestrator override: max rounds reached)"
                    ).strip(),
                }
            )
        return decision

    @staticmethod
    def _append_control_turn(
        transcript: list[AgentOutput],
        decision: JudgeControlDecision,
        round_index: int,
    ) -> None:
        transcript.append(
            AgentOutput(
                role="judge",
                round_index=round_index,
                private_strategy=f"control:{decision.action}",
                public_argument=decision.message or decision.reasoning,
            )
        )

    def _append_closings(
        self,
        case: CaseProfile,
        evidence,
        memory_context,
        transcript: list[AgentOutput],
    ) -> None:
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
