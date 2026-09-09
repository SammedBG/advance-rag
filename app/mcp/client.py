import asyncio
import json
from dataclasses import dataclass
from typing import Any

from mcp import Client


@dataclass
class MCPToolResult:
    tool_name: str
    content: Any
    success: bool
    error: str | None = None


class MCPClient:
    def __init__(
        self,
        server_url: str,
    ) -> None:
        if not server_url.strip():
            raise ValueError(
                "MCP server URL cannot be empty."
            )

        self.server_url = server_url

    def list_tools(self) -> list[str]:
        return asyncio.run(
            self._list_tools()
        )

    async def _list_tools(self) -> list[str]:
        async with Client(self.server_url) as client:
            tools = await client.list_tools()

            return [
                tool.name
                for tool in tools.tools
            ]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        return asyncio.run(
            self._call_tool(
                name=name,
                arguments=arguments or {},
            )
        )

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        try:
            async with Client(
                self.server_url
            ) as client:
                result = await client.call_tool(
                    name,
                    arguments,
                )

                if getattr(
                    result,
                    "is_error",
                    False,
                ):
                    return MCPToolResult(
                        tool_name=name,
                        content=None,
                        success=False,
                        error=str(result.content),
                    )

                structured_content = getattr(
                    result,
                    "structured_content",
                    None,
                )

                if structured_content is not None:
                    content = structured_content
                else:
                    content = self._parse_content(
                        result.content
                    )

                return MCPToolResult(
                    tool_name=name,
                    content=content,
                    success=True,
                )

        except Exception as exc:
            return MCPToolResult(
                tool_name=name,
                content=None,
                success=False,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    @staticmethod
    def _parse_content(
        content: list[Any],
    ) -> Any:
        parsed_items: list[Any] = []

        for item in content:
            text = getattr(
                item,
                "text",
                None,
            )

            if text is None:
                parsed_items.append(item)
                continue

            try:
                parsed_items.append(
                    json.loads(text)
                )
            except (
                json.JSONDecodeError,
                TypeError,
            ):
                parsed_items.append(text)

        if len(parsed_items) == 1:
            return parsed_items[0]

        return parsed_items