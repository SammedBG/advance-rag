from app.agent.graph import build_rag_graph
from app.agent.nodes import AgentDependencies


class RAGAgent:
    def __init__(
        self,
        dependencies: AgentDependencies,
    ) -> None:
        self.graph = build_rag_graph(
            dependencies=dependencies
        )

    def run(
        self,
        query: str,
    ) -> dict:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        state = self.graph.invoke(
            {
                "query": query,
            }
        )

        return {
            "query": state.get(
                "query"
            ),
            "route": state.get(
                "route",
                "rag",
            ),
            "answer": state.get(
                "answer",
                "",
            ),
            "citations": state.get(
                "citations",
                [],
            ),
            "citation_validation": state.get(
                "citation_validation",
                {},
            ),
            "grounding": state.get(
                "grounding",
                {},
            ),
            "context_stats": state.get(
                "context_stats",
                {},
            ),
            "mcp_result": state.get(
                "mcp_result",
                {},
            ),
            "error": state.get(
                "error",
            ),
        }