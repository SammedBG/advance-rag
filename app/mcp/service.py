import logging
from typing import Any

from app.mcp.client import MCPClient, MCPToolResult
from app.mcp import tools

logger = logging.getLogger(__name__)


class MCPService:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8001/mcp",
    ) -> None:
        self.server_url = server_url
        self.client = MCPClient(
            server_url=server_url,
        ) if server_url else None

    def list_tools(self) -> list[str]:
        if self.client:
            try:
                return self.client.list_tools()
            except Exception as exc:
                logger.debug("Failed to list remote MCP tools (%s); returning local tools", exc)
        return [
            "get_pod_status_tool",
            "get_cluster_health_tool",
            "search_knowledge_base_tool",
        ]

    def get_pod_status(
        self,
        pod_name: str,
        namespace: str = "default",
    ) -> MCPToolResult:
        if self.client:
            try:
                res = self.client.call_tool(
                    name="get_pod_status_tool",
                    arguments={
                        "pod_name": pod_name,
                        "namespace": namespace,
                    },
                )
                if res.success:
                    return res
            except Exception as exc:
                logger.debug("Remote MCP call failed (%s); using local fallback", exc)

        # Local fallback execution
        data = tools.get_pod_status(pod_name=pod_name, namespace=namespace)
        return MCPToolResult(
            tool_name="get_pod_status_tool",
            content=data,
            success=True,
        )

    def get_cluster_health(self) -> MCPToolResult:
        if self.client:
            try:
                res = self.client.call_tool(
                    name="get_cluster_health_tool",
                    arguments={},
                )
                if res.success:
                    return res
            except Exception as exc:
                logger.debug("Remote MCP call failed (%s); using local fallback", exc)

        data = tools.get_cluster_health()
        return MCPToolResult(
            tool_name="get_cluster_health_tool",
            content=data,
            success=True,
        )

    def search_knowledge_base(
        self,
        query: str,
        limit: int = 3,
        technology: str | None = None,
    ) -> MCPToolResult:
        if self.client:
            try:
                res = self.client.call_tool(
                    name="search_knowledge_base_tool",
                    arguments={
                        "query": query,
                        "limit": limit,
                        "technology": technology,
                    },
                )
                if res.success:
                    return res
            except Exception as exc:
                logger.debug("Remote MCP call failed (%s); using local fallback", exc)

        data = tools.search_knowledge_base(
            query=query,
            limit=limit,
            technology=technology,
        )
        return MCPToolResult(
            tool_name="search_knowledge_base_tool",
            content=data,
            success=True,
        )