"""Benchmark engine for single-agent vs multi-agent research evaluation."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost based on standard blended rates ($0.15/1M input, $0.60/1M output)."""
    cost_in = (input_tokens / 1_000_000) * 0.15
    cost_out = (output_tokens / 1_000_000) * 0.60
    return cost_in + cost_out


def calculate_citation_coverage(text: str, num_sources: int) -> float:
    """Estimate citation coverage based on inline reference patterns."""
    if not text:
        return 0.0
    if num_sources == 0:
        return 0.0

    # Match citations like [1], [2], [Source 1], [A01], [SYN-01]
    matches = set(re.findall(r"\[(?:Source\s*)?([0-9A-Za-z\-_]+)\]", text))
    if not matches:
        return 0.0

    coverage = min(1.0, len(matches) / max(1, num_sources))
    return round(coverage, 2)


def evaluate_quality_score(state: ResearchState) -> float:
    """Heuristic quality score on 0-10 scale evaluating structure, evidence, and depth."""
    text = state.final_answer or ""
    if not text or len(text.strip()) < 50:
        return 0.0

    score = 4.0  # Base score for non-empty response

    # 1. Structure & Headers
    headers = re.findall(r"^#+\s+.+", text, flags=re.MULTILINE)
    if len(headers) >= 4:
        score += 2.0
    elif len(headers) >= 2:
        score += 1.0

    # 2. Content Depth & Length
    word_count = len(text.split())
    if word_count >= 400:
        score += 2.0
    elif word_count >= 200:
        score += 1.0

    # 3. Grounding & Citations
    if len(state.sources) > 0:
        score += 1.0
    citations = re.findall(r"\[(?:Source\s*)?([0-9A-Za-z\-_]+)\]", text)
    if len(citations) >= 3:
        score += 1.0

    return min(10.0, round(score, 1))


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, compute token costs, quality, citation coverage, and failure rate."""

    started = perf_counter()
    try:
        state = runner(query)
        latency = perf_counter() - started
        failure_rate = 0.0 if (state.final_answer and len(state.final_answer) > 20) else 1.0
    except Exception as exc:
        latency = perf_counter() - started
        failure_rate = 1.0
        state = ResearchState(
            request={"query": query},  # type: ignore[arg-type]
            errors=[str(exc)],
        )

    # Token summation
    total_in = 0
    total_out = 0
    for res in state.agent_results:
        if isinstance(res.metadata, dict):
            total_in += res.metadata.get("input_tokens") or 0
            total_out += res.metadata.get("output_tokens") or 0

    cost = estimate_cost(total_in, total_out)
    citation_cov = calculate_citation_coverage(state.final_answer or "", len(state.sources))
    quality = evaluate_quality_score(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=round(cost, 6),
        quality_score=quality,
        citation_coverage=citation_cov,
        failure_rate=failure_rate,
        notes=f"Tokens: {total_in} in / {total_out} out | Iterations: {state.iteration}",
    )
    return state, metrics
