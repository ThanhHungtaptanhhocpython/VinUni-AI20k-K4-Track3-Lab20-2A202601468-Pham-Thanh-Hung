"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final structured report with citations from research and analysis notes."""

    name = "writer"

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(self.settings)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        query = state.request.query
        audience = state.request.audience
        research_notes = state.research_notes or "No research notes."
        analysis_notes = state.analysis_notes or "No analysis notes."

        sources_ref = "\n".join(
            f"- [{idx+1}] {s.title} ({s.url or 'local'})"
            for idx, s in enumerate(state.sources)
        ) or "No external sources referenced."

        system_prompt = (
            "You are a professional Technical Writer Agent. Synthesize a comprehensive, rigorous, "
            f"and polished research report for {audience}. "
            "Integrate empirical evidence from research notes and insights from analysis notes. "
            "Include inline source citations (e.g. [1], [2] or [Source ID]) where appropriate."
        )
        user_prompt = (
            f"Research Question: {query}\n\n"
            f"Research Notes:\n{research_notes}\n\n"
            f"Analysis Notes:\n{analysis_notes}\n\n"
            f"Available Sources:\n{sources_ref}\n\n"
            "Structure the report with:\n"
            "# Title\n"
            "## Executive Summary\n"
            "## Technical & Architectural Overview\n"
            "## In-Depth Analysis & Trade-offs\n"
            "## Conclusion & Recommendations\n"
            "## References / Sources"
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.final_answer = response.content

        state.add_trace_event(
            "writer_completed",
            {
                "tokens_in": response.input_tokens,
                "tokens_out": response.output_tokens,
            },
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )

        return state
