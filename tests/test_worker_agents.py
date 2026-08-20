"""Unit tests for worker agents (Researcher, Analyst, Writer) and SearchClient."""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


class MockLLMClient:
    """Mock LLM client for deterministic testing."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(
            content=f"Mock response for prompt: {user_prompt[:30]}...",
            input_tokens=20,
            output_tokens=30,
        )


class MockSearchClient:
    """Mock search client returning deterministic source documents."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return [
            SourceDocument(
                title="Mock GraphRAG Paper",
                url="https://example.com/graphrag",
                snippet="GraphRAG integrates structured knowledge graphs with LLM generation.",
                metadata={"source_id": "MOCK-01"},
            )
        ]


def test_search_client_offline_fallback() -> None:
    client = SearchClient()
    results = client.search(query="GraphRAG architecture multi-agent", max_results=3)
    assert len(results) > 0
    assert results[0].title
    assert results[0].snippet


def test_researcher_agent_execution() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain GraphRAG state-of-the-art"))
    agent = ResearcherAgent(
        search_client=MockSearchClient(),  # type: ignore[arg-type]
        llm_client=MockLLMClient(),  # type: ignore[arg-type]
    )
    updated = agent.run(state)
    assert len(updated.sources) == 1
    assert updated.research_notes is not None
    assert "Mock response" in updated.research_notes
    assert len(updated.agent_results) == 1


def test_analyst_agent_execution() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain GraphRAG state-of-the-art"),
        research_notes="Key fact 1: Graph-based retrieval outperforms naive RAG on global queries.",
    )
    agent = AnalystAgent(llm_client=MockLLMClient())  # type: ignore[arg-type]
    updated = agent.run(state)
    assert updated.analysis_notes is not None
    assert "Mock response" in updated.analysis_notes
    assert len(updated.agent_results) == 1


def test_writer_agent_execution() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain GraphRAG state-of-the-art"),
        research_notes="Research notes on GraphRAG.",
        analysis_notes="Analysis notes on trade-offs.",
        sources=[SourceDocument(title="Doc 1", snippet="Snippet 1")],
    )
    agent = WriterAgent(llm_client=MockLLMClient())  # type: ignore[arg-type]
    updated = agent.run(state)
    assert updated.final_answer is not None
    assert "Mock response" in updated.final_answer
    assert len(updated.agent_results) == 1
