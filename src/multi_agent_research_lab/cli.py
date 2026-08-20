from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a real single-agent baseline LLM call and display metrics."""

    _init()
    settings = get_settings()
    request = _parse_query(query)
    state = ResearchState(request=request)

    console.print(
        Panel.fit(
            f"[bold cyan]Query:[/bold cyan] {request.query}\n"
            f"[bold cyan]Model:[/bold cyan] {settings.effective_model}\n"
            f"[bold cyan]Endpoint:[/bold cyan] {settings.effective_base_url}",
            title="Single-Agent Baseline (Starting)",
        )
    )

    llm = LLMClient(settings)
    system_prompt = (
        "You are an expert research assistant. Conduct comprehensive analysis on the given topic "
        "and provide a well-structured, clear, and detailed synthesis report."
    )

    start_time = perf_counter()
    response = llm.complete(system_prompt=system_prompt, user_prompt=request.query)
    elapsed_time = perf_counter() - start_time

    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "latency_seconds": elapsed_time,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "model": settings.effective_model,
            },
        )
    )

    console.print(
        Panel.fit(
            f"[bold green]Latency:[/bold green] {elapsed_time:.2f}s | "
            f"[bold green]Input Tokens:[/bold green] {response.input_tokens} | "
            f"[bold green]Output Tokens:[/bold green] {response.output_tokens}",
            title="Baseline Metrics",
            style="green",
        )
    )
    console.print(Panel(state.final_answer, title="Single-Agent Synthesis Output"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow (Supervisor + Researcher + Analyst + Writer)."""

    _init()
    settings = get_settings()
    request = _parse_query(query)
    state = ResearchState(request=request)

    console.print(
        Panel.fit(
            f"[bold cyan]Query:[/bold cyan] {request.query}\n"
            f"[bold cyan]Model:[/bold cyan] {settings.effective_model}\n"
            f"[bold cyan]Max Iterations:[/bold cyan] {settings.max_iterations}",
            title="Multi-Agent Research System (Starting)",
        )
    )

    workflow = MultiAgentWorkflow(settings=settings)
    start_time = perf_counter()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    elapsed_time = perf_counter() - start_time

    # Calculate token usage across agent results
    total_in = sum(
        res.metadata.get("input_tokens") or 0
        for res in result.agent_results
        if isinstance(res.metadata, dict)
    )
    total_out = sum(
        res.metadata.get("output_tokens") or 0
        for res in result.agent_results
        if isinstance(res.metadata, dict)
    )

    # Display routing execution path
    route_display = " -> ".join(result.route_history)
    console.print(
        Panel.fit(
            f"[bold yellow]Route Timeline:[/bold yellow] {route_display}\n"
            f"[bold yellow]Iterations:[/bold yellow] {result.iteration} | "
            f"[bold yellow]Sources Gathered:[/bold yellow] {len(result.sources)}\n"
            f"[bold green]Total Latency:[/bold green] {elapsed_time:.2f}s | "
            f"[bold green]Total Tokens:[/bold green] In: {total_in}, Out: {total_out}",
            title="Multi-Agent Workflow Execution Summary",
            style="green",
        )
    )

    if result.final_answer:
        console.print(Panel(result.final_answer, title="Multi-Agent Final Research Report"))
    else:
        console.print(Panel("No final answer produced.", title="Result", style="yellow"))


if __name__ == "__main__":
    app()
