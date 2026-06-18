"""Backward-compatible aliases for Phase 1 QA debate agents."""

from __future__ import annotations

from src.agents.debate_agent import DebateAgent
from src.llm import LLMClient


def ProponentAgent(llm: LLMClient, **kwargs) -> DebateAgent:
    """Phase 1 alias for the supporting debate agent."""

    return DebateAgent("proponent", llm, **kwargs)


def OpponentAgent(llm: LLMClient, **kwargs) -> DebateAgent:
    """Phase 1 alias for the challenging debate agent."""

    return DebateAgent("opponent", llm, **kwargs)


def create_phase1_debate_pair(
    proponent_llm: LLMClient,
    opponent_llm: LLMClient,
    **kwargs,
) -> tuple[DebateAgent, DebateAgent]:
    """Create the legacy proponent/opponent pair used by DebateOrchestrator."""

    return ProponentAgent(proponent_llm, **kwargs), OpponentAgent(opponent_llm, **kwargs)
