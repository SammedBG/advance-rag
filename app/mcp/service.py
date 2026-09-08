from app.mcp.client import MCPClient, MCPToolResult


class MCPService:
    def __init__(
        self,
        server_url: str,
    ) -> None:
        self.client = MCPClient(
            server_url=server_url,
        )

    def list_tools(self) -> list[str]:
        return self.client.list_tools()

    def get_pod_status(
        self,
        pod_name: str,
        namespace: str = "default",
    ) -> MCPToolResult:
        return self.client.call_tool(
            name="get_pod_status_tool",
            arguments={
                "pod_name": pod_name,
                "namespace": namespace,
            },
        )