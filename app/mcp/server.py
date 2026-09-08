from mcp.server import MCPServer

from app.mcp.tools import get_pod_status


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


if __name__ == "__main__":
    mcp.run(
        "streamable-http",
        host="127.0.0.1",
        port=8001,
    )