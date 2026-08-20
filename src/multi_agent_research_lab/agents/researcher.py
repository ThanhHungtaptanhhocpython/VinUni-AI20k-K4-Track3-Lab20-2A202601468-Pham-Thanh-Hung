"""Researcher agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates structured research notes."""

    name = "researcher"

    def __init__(
        self,
        settings: Settings | None = None,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.search_client = search_client or SearchClient(self.settings)
        self.llm_client = llm_client or LLMClient(self.settings)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        # 1. Search for relevant sources
        sources = self.search_client.search(query=query, max_results=max_sources)
        state.sources.extend(sources)

        # 2. Build context string from gathered sources
        sources_text = "\n\n".join(
            f"Source [{idx+1}] Title: {s.title}\nURL/ID: {s.url}\nContent:\n{s.snippet}"
            for idx, s in enumerate(sources)
        )

        system_prompt = (
            "You are a meticulous Researcher Agent. Extract factual findings, core definitions, "
            "and empirical observations from the provided source snippets. "
            "Organize them into clear bullet points with explicit source references "
            "(e.g. [Source 1])."
        )
        user_prompt = (
            f"Research Question: {query}\n\n"
            f"Gathered Sources:\n{sources_text}\n\n"
            "Produce structured Research Notes covering key evidence and facts."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.research_notes = response.content

        # 3. Record trace and agent result
        state.add_trace_event(
            "researcher_completed",
            {
                "num_sources": len(sources),
                "tokens_in": response.input_tokens,
                "tokens_out": response.output_tokens,
            },
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "num_sources": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )

        return state
