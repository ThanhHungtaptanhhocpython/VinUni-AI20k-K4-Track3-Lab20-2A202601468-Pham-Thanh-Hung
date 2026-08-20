"""Supervisor / router implementation."""

from typing import Literal

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

RouteDecision = Literal["researcher", "analyst", "writer", "done"]


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def decide_next_route(self, state: ResearchState) -> RouteDecision:
        """Determine next agent route based on current state and guardrails."""
        # 1. Guardrail: Max iterations reached
        if state.iteration >= self.settings.max_iterations:
            if state.final_answer:
                return "done"
            # If final answer not yet synthesized, attempt writer fallback if notes exist
            if state.research_notes or state.analysis_notes:
                return "writer"
            return "done"

        # 2. Final answer exists -> Stop workflow
        if state.final_answer and len(state.final_answer.strip()) > 0:
            return "done"

        # 3. Analysis notes exist -> Synthesize final report with Writer
        if state.analysis_notes and len(state.analysis_notes.strip()) > 0:
            return "writer"

        # 4. Research notes or sources collected -> Analyze findings with Analyst
        has_research = bool(state.research_notes and state.research_notes.strip())
        if has_research or len(state.sources) > 0:
            return "analyst"

        # 5. Initial state -> Gather evidence with Researcher
        return "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Evaluate state, update route history, record trace and agent result."""
        next_route = self.decide_next_route(state)
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_routing",
            {
                "next_route": next_route,
                "iteration": state.iteration,
                "has_sources": len(state.sources) > 0,
                "has_research_notes": state.research_notes is not None,
                "has_analysis_notes": state.analysis_notes is not None,
                "has_final_answer": state.final_answer is not None,
            },
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Routing decision: proceed to '{next_route}' (step {state.iteration})",
                metadata={"next_route": next_route, "iteration": state.iteration},
            )
        )
        return state
