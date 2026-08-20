"""LangGraph workflow implementation."""

from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph with LangGraph."""

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()
        self._compiled_graph: Any = None

    def build(self) -> Any:
        """Create and compile the LangGraph workflow."""
        builder = StateGraph(dict)

        def _supervisor_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = ResearchState.model_validate(state_dict)
            state = self.supervisor.run(state)
            return state.model_dump()

        def _researcher_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = ResearchState.model_validate(state_dict)
            state = self.researcher.run(state)
            return state.model_dump()

        def _analyst_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = ResearchState.model_validate(state_dict)
            state = self.analyst.run(state)
            return state.model_dump()

        def _writer_node(state_dict: dict[str, Any]) -> dict[str, Any]:
            state = ResearchState.model_validate(state_dict)
            state = self.writer.run(state)
            return state.model_dump()

        builder.add_node("supervisor", _supervisor_node)
        builder.add_node("researcher", _researcher_node)
        builder.add_node("analyst", _analyst_node)
        builder.add_node("writer", _writer_node)

        builder.set_entry_point("supervisor")

        def _route_condition(state_dict: dict[str, Any]) -> str:
            state = ResearchState.model_validate(state_dict)
            if state.route_history:
                last_route = state.route_history[-1]
                if last_route in ["researcher", "analyst", "writer", "done"]:
                    return last_route
            return self.supervisor.decide_next_route(state)

        builder.add_conditional_edges(
            "supervisor",
            _route_condition,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )

        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")

        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the updated final ResearchState."""
        if self._compiled_graph is None:
            self._compiled_graph = self.build()

        initial_dict = state.model_dump()
        final_dict = self._compiled_graph.invoke(initial_dict)
        return ResearchState.model_validate(final_dict)
