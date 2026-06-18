"""Courtroom session lifecycle for Phase 3 LJP simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.agents.defendant import DefendantAgent
from src.agents.defense import DefenseAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.prosecutor import ProsecutorAgent
from src.courtroom.protocol import CourtroomProtocol, ProtocolConfig
from src.memory.memory_store import MemoryStore
from src.models import (
    CourtCase,
    CourtroomResult,
    EvidenceDocument,
    MemoryContext,
)
from src.retrieval.legal_retriever import LegalRetriever


class CourtroomSession:
    """Orchestrate a full courtroom session with three phases."""

    def __init__(
        self,
        prosecutor: ProsecutorAgent,
        defense: DefenseAgent,
        defendant: DefendantAgent,
        judge: JudgeAgent,
        protocol: CourtroomProtocol | None = None,
        legal_retriever: LegalRetriever | None = None,
        memory_store: MemoryStore | None = None,
        evidence_top_k: int = 5,
        memory_top_k: int = 5,
    ) -> None:
        self.prosecutor = prosecutor
        self.defense = defense
        self.defendant = defendant
        self.judge = judge
        self.protocol = protocol or CourtroomProtocol()
        self.legal_retriever = legal_retriever
        self.memory_store = memory_store
        self.evidence_top_k = evidence_top_k
        self.memory_top_k = memory_top_k

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        prosecutor: ProsecutorAgent,
        defense: DefenseAgent,
        defendant: DefendantAgent,
        judge: JudgeAgent,
        legal_retriever: LegalRetriever | None = None,
        memory_store: MemoryStore | None = None,
    ) -> CourtroomSession:
        """Build a session from ``configs/courtroom.yaml``."""

        path = Path(config_path)
        raw: dict[str, Any] = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                raw = loaded

        courtroom_cfg = raw.get("courtroom", raw)
        protocol_cfg = courtroom_cfg.get("protocol", {})
        protocol = CourtroomProtocol(
            ProtocolConfig(
                enable_opening=bool(protocol_cfg.get("enable_opening", True)),
                enable_debate=bool(protocol_cfg.get("enable_debate", True)),
                max_debate_rounds=int(protocol_cfg.get("max_debate_rounds", 2)),
                enable_judge_question=bool(
                    protocol_cfg.get("enable_judge_question", False)
                ),
                enable_closing=bool(protocol_cfg.get("enable_closing", True)),
                enable_deliberation=bool(protocol_cfg.get("enable_deliberation", True)),
                early_stop_confidence=protocol_cfg.get("early_stop_confidence"),
            )
        )
        retrieval_cfg = courtroom_cfg.get("retrieval", {})
        memory_cfg = courtroom_cfg.get("memory", {})
        return cls(
            prosecutor=prosecutor,
            defense=defense,
            defendant=defendant,
            judge=judge,
            protocol=protocol,
            legal_retriever=legal_retriever,
            memory_store=memory_store,
            evidence_top_k=int(retrieval_cfg.get("evidence_top_k", 5)),
            memory_top_k=int(memory_cfg.get("memory_top_k", 5)),
        )

    def run(
        self,
        court_case: CourtCase,
        legal_evidence: list[EvidenceDocument] | None = None,
        past_memory: MemoryContext | None = None,
    ) -> CourtroomResult:
        """Execute opening, debate, and judgment phases."""

        self.judge.enable_courtroom_mode()
        self.judge.reset()

        profile = court_case.to_case_profile()
        transcript: list = []
        phases_completed: list[str] = []
        memory_context = past_memory or self._retrieve_memory(profile)
        evidence = legal_evidence or self._retrieve_evidence(profile)

        opening_turns, opening_phases = self.protocol.opening(
            court_case=court_case,
            judge=self.judge,
            prosecutor=self.prosecutor,
            defendant=self.defendant,
            defense=self.defense,
            legal_evidence=evidence,
            past_memory=memory_context,
            transcript=transcript,
        )
        transcript.extend(opening_turns)
        phases_completed.extend(opening_phases)

        for round_index in range(1, self.protocol.config.max_debate_rounds + 1):
            debate_turns, debate_phases = self.protocol.debate_round(
                court_case=court_case,
                prosecutor=self.prosecutor,
                defense=self.defense,
                judge=self.judge,
                legal_evidence=evidence,
                past_memory=memory_context,
                transcript=transcript,
                round_index=round_index,
            )
            transcript.extend(debate_turns)
            phases_completed.extend(debate_phases)

            if (
                self.protocol.config.early_stop_confidence is not None
                and self.judge.belief_history
                and self.judge.belief_history[-1].confidence
                >= self.protocol.config.early_stop_confidence
            ):
                break

        closing_turns, closing_phases = self.protocol.closing(
            court_case=court_case,
            prosecutor=self.prosecutor,
            defense=self.defense,
            legal_evidence=evidence,
            past_memory=memory_context,
            transcript=transcript,
        )
        transcript.extend(closing_turns)
        phases_completed.extend(closing_phases)

        deliberation: str | None = None
        if self.protocol.config.enable_deliberation:
            deliberation = self.judge.deliberate(
                court_case=court_case,
                transcript=transcript,
            )
            phases_completed.append("deliberation")

        legal_judgment = self.judge.render_ljp_verdict(
            court_case=court_case,
            transcript=transcript,
            deliberation=deliberation,
        )
        phases_completed.append("final_ruling")

        verdict_profile = court_case.to_case_profile()
        verdict = self.judge.render_verdict(case=verdict_profile, transcript=transcript)

        return CourtroomResult(
            case_id=court_case.case_id,
            legal_evidence=evidence,
            memory_context=memory_context,
            transcript=transcript,
            belief_history=list(self.judge.belief_history),
            phases_completed=phases_completed,
            deliberation=deliberation,
            verdict=verdict,
            legal_judgment=legal_judgment,
        )

    def _retrieve_evidence(self, profile) -> list[EvidenceDocument]:
        if self.legal_retriever is None:
            return self._default_context_evidence(profile)
        if getattr(self.legal_retriever, "method", None) == "off":
            return []
        retrieved = self.legal_retriever.retrieve(
            profile.retrieval_query,
            top_k=self.evidence_top_k,
        )
        if not retrieved:
            return self._default_context_evidence(profile)
        return retrieved

    def _retrieve_memory(self, profile) -> MemoryContext:
        if self.memory_store is None:
            return MemoryContext()
        return self.memory_store.query(profile, top_k=self.memory_top_k)

    @staticmethod
    def _default_context_evidence(profile) -> list[EvidenceDocument]:
        return [
            EvidenceDocument(
                doc_id=f"{profile.case_id}-context",
                text=profile.context,
                source="input_context",
                score=1.0,
            )
        ]
