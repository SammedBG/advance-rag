from mcp.server.mcpserver import MCPServer

from app.mcp.tools import (
    get_cluster_health,
    get_pod_status,
    search_knowledge_base,
)


mcp = MCPServer(
    "advanced-rag-tools",
)


@mcp.tool()
def get_pod_status_tool(
    pod_name: str,
    namespace: str = "default",
) -> dict:
    """
    Get the current status of a Kubernetes pod.
    """
    return get_pod_status(
        pod_name=pod_name,
        namespace=namespace,
    )


@mcp.tool()
def get_cluster_health_tool() -> dict:
    """
    Get the cluster health status and node counts.
    """
    return get_cluster_health()


@mcp.tool()
def search_knowledge_base_tool(
    query: str,
    limit: int = 3,
    technology: str | None = None,
) -> dict:
    """
    Search the RAG knowledge base for relevant chunks.
    """
    return search_knowledge_base(
        query=query,
        limit=limit,
        technology=technology,
    )


if __name__ == "__main__":
    mcp.run(
        "streamable-http",
        host="127.0.0.1",
        port=8001,
    )