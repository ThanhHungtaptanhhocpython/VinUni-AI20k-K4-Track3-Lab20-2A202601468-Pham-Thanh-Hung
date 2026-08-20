import os
from pathlib import Path
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
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    # Automatically export LangSmith environment variables if configured
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    # Automatically export Langfuse environment variables if configured
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host



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


@app.command("benchmark")
def benchmark(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Research query for benchmark comparison"),
    ] = "Research GraphRAG state-of-the-art and write a summary",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output markdown report path"),
    ] = "reports/benchmark_report.md",
) -> None:
    """Run benchmark comparing Single-Agent Baseline vs Multi-Agent Workflow."""

    _init()
    settings = get_settings()
    console.print(
        Panel.fit(
            f"[bold cyan]Benchmark Query:[/bold cyan] {query}\n"
            f"[bold cyan]Target Model:[/bold cyan] {settings.effective_model}\n"
            f"[bold cyan]Output Report:[/bold cyan] {output}",
            title="Starting Single vs Multi-Agent Benchmark",
        )
    )

    # 1. Runner for Single-Agent Baseline
    def _run_baseline(q: str) -> ResearchState:
        req = _parse_query(q)
        st = ResearchState(request=req)
        llm = LLMClient(settings)
        sys_p = (
            "You are an expert research assistant. Conduct comprehensive analysis on the topic "
            "and provide a well-structured synthesis report."
        )
        resp = llm.complete(system_prompt=sys_p, user_prompt=req.query)
        st.final_answer = resp.content
        st.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                },
            )
        )
        return st

    # 2. Runner for Multi-Agent Workflow
    def _run_multi_agent(q: str) -> ResearchState:
        req = _parse_query(q)
        st = ResearchState(request=req)
        wf = MultiAgentWorkflow(settings=settings)
        return wf.run(st)

    console.print("[cyan]Running Single-Agent Baseline...[/cyan]")
    _, baseline_metrics = run_benchmark("Single-Agent Baseline", query, _run_baseline)

    console.print("[cyan]Running Multi-Agent Workflow...[/cyan]")
    _, multi_metrics = run_benchmark("Multi-Agent Research System", query, _run_multi_agent)

    # 3. Render and save report
    all_metrics = [baseline_metrics, multi_metrics]
    report_md = render_markdown_report(all_metrics, query=query)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    console.print(
        Panel.fit(
            f"[bold green]Single-Agent:[/bold green] {baseline_metrics.latency_seconds:.2f}s | "
            f"Quality: {baseline_metrics.quality_score:.1f}/10 | "
            f"Cost: ${baseline_metrics.estimated_cost_usd:.6f}\n"
            f"[bold green]Multi-Agent:[/bold green]  {multi_metrics.latency_seconds:.2f}s | "
            f"Quality: {multi_metrics.quality_score:.1f}/10 | "
            f"Cost: ${multi_metrics.estimated_cost_usd:.6f}\n"
            f"[bold yellow]Report Saved:[/bold yellow] {output}",
            title="Benchmark Completed Successfully",
            style="green",
        )
    )


if __name__ == "__main__":
    app()
