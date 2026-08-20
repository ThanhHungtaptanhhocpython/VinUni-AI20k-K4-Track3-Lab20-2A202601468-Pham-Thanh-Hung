"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights, comparing viewpoints and trade-offs."""

    name = "analyst"

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(self.settings)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        query = state.request.query
        research_notes = state.research_notes or "No prior research notes provided."

        system_prompt = (
            "You are an analytical Research Analyst Agent. Your role is to critically evaluate "
            "the research notes provided. Identify key patterns, trade-offs, strengths and "
            "weaknesses of approaches, contradictions, and structural implications for the topic."
        )
        user_prompt = (
            f"Research Topic: {query}\n\n"
            f"Research Notes:\n{research_notes}\n\n"
            "Please analyze these findings. Provide:\n"
            "1. Core Analytical Insights & Mechanisms\n"
            "2. Trade-offs and Comparative Evaluation\n"
            "3. Contradictions or Limitations in Current Evidence\n"
            "4. Strategic Takeaways"
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.analysis_notes = response.content

        state.add_trace_event(
            "analyst_completed",
            {
                "tokens_in": response.input_tokens,
                "tokens_out": response.output_tokens,
            },
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )

        return state
