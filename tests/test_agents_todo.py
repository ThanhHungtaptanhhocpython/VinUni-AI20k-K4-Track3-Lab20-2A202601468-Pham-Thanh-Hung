"""Unit tests for SupervisorAgent routing policy and MultiAgentWorkflow build."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_supervisor_initial_routing() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent()
    updated_state = supervisor.run(state)
    assert updated_state.route_history == ["researcher"]
    assert updated_state.iteration == 1


def test_supervisor_routes_to_analyst_when_sources_present() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Doc 1", snippet="Snippet content")],
    )
    supervisor = SupervisorAgent()
    updated_state = supervisor.run(state)
    assert updated_state.route_history == ["analyst"]


def test_supervisor_routes_to_writer_when_analysis_present() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        analysis_notes="Key finding 1: Multi-agent improves specialization.",
    )
    supervisor = SupervisorAgent()
    updated_state = supervisor.run(state)
    assert updated_state.route_history == ["writer"]


def test_supervisor_routes_to_done_when_final_answer_present() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        final_answer="Comprehensive synthesis report on multi-agent systems.",
    )
    supervisor = SupervisorAgent()
    updated_state = supervisor.run(state)
    assert updated_state.route_history == ["done"]


def test_supervisor_max_iterations_guard() -> None:
    settings = Settings(max_iterations=3)
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        iteration=3,
    )
    supervisor = SupervisorAgent(settings=settings)
    updated_state = supervisor.run(state)
    assert updated_state.route_history == ["done"]


def test_multi_agent_workflow_build() -> None:
    workflow = MultiAgentWorkflow()
    graph = workflow.build()
    assert graph is not None
