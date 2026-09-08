from typing import TypedDict


class AgentState(TypedDict, total=False):
    query: str
    route: str
    answer: str
    citations: list[int]
    citation_validation: dict
    grounding: dict
    context_stats: dict
    compressed_contexts: list
    mcp_result: dict
    error: str