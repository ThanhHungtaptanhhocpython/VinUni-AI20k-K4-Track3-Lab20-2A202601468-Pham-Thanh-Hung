import json
import urllib.parse
import urllib.request
from pathlib import Path

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client with Tavily and Offline Corpus v2 support."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.corpus_dir = Path("ai_agent_offline_research_corpus_v2/topics")

    def _search_tavily(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Query Tavily Search API."""
        api_key = self.settings.tavily_api_key
        if not api_key:
            return []

        try:
            url = "https://api.tavily.com/search"
            headers = {
                "Content-Type": "application/json",
            }
            payload = {
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_snippets": True,
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")

            with urllib.request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
                docs: list[SourceDocument] = []
                for item in result.get("results", [])[:max_results]:
                    docs.append(
                        SourceDocument(
                            title=item.get("title", "Untitled Web Result"),
                            url=item.get("url"),
                            snippet=item.get("content", item.get("snippet", "")),
                            metadata={"score": item.get("score")},
                        )
                    )
                return docs
        except Exception:
            return []

    def _search_offline_corpus(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search inside embedded topics in Offline Research Corpus v2."""
        if not self.corpus_dir.exists():
            return []

        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        scored_docs: list[tuple[float, SourceDocument]] = []

        for json_file in self.corpus_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                kb = data.get("knowledge_base", {})
                articles = kb.get("knowledge_articles", [])
                sources = kb.get("source_documents", [])
                facts = kb.get("atomic_facts", [])

                for art in articles:
                    content = art.get("content", "")
                    title = art.get("title", "")
                    article_id = art.get("article_id", "A0")
                    full_text = f"{title} {content}".lower()
                    score = sum(1.0 for term in query_terms if term in full_text)
                    if score > 0:
                        snippet = content[:500] + "..." if len(content) > 500 else content
                        scored_docs.append(
                            (
                                score,
                                SourceDocument(
                                    title=f"[{article_id}] {title}",
                                    url=f"corpus://{json_file.stem}#{article_id}",
                                    snippet=snippet,
                                    metadata={"source_id": article_id, "type": "article"},
                                ),
                            )
                        )

                for src in sources:
                    content = src.get("content", src.get("summary", ""))
                    title = src.get("title", "")
                    source_id = src.get("source_id", "S0")
                    full_text = f"{title} {content}".lower()
                    score = sum(1.0 for term in query_terms if term in full_text)
                    if score > 0:
                        snippet = content[:500] + "..." if len(content) > 500 else content
                        scored_docs.append(
                            (
                                score,
                                SourceDocument(
                                    title=f"[{source_id}] {title}",
                                    url=src.get("url") or f"corpus://{json_file.stem}#{source_id}",
                                    snippet=snippet,
                                    metadata={"source_id": source_id, "type": "source_document"},
                                ),
                            )
                        )

                for fact in facts:
                    stmt = fact.get("statement", fact.get("fact_statement", ""))
                    fid = fact.get("fact_id", "F0")
                    score = sum(1.0 for term in query_terms if term in stmt.lower())
                    if score > 0:
                        scored_docs.append(
                            (
                                score * 0.8,
                                SourceDocument(
                                    title=f"[{fid}] Fact",
                                    url=f"corpus://{json_file.stem}#{fid}",
                                    snippet=stmt,
                                    metadata={"source_id": fid, "type": "fact"},
                                ),
                            )
                        )
            except Exception:
                continue

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:max_results]]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        # 1. Try Tavily search if configured
        if self.settings.tavily_api_key:
            results = self._search_tavily(query, max_results=max_results)
            if results:
                return results

        # 2. Search local offline corpus
        results = self._search_offline_corpus(query, max_results=max_results)
        if results:
            return results

        # 3. Fallback mock source if neither returned data
        return [
            SourceDocument(
                title="GraphRAG & Multi-Agent State-of-the-Art Overview",
                url="https://arxiv.org/abs/2404.16130",
                snippet=(
                    "GraphRAG combines knowledge graphs with retrieval-augmented generation. "
                    "In multi-agent systems, specialization across supervisor, researcher, "
                    "analyst, and writer roles ensures comprehensive coverage and citation rigor."
                ),
                metadata={"source_id": "SYNTHETIC-01", "type": "overview"},
            )
        ]
