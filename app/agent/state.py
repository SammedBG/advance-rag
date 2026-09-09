from typing import TypedDict


class AgentState(TypedDict, total=False):
    query: str
    original_query: str
    route: str
    iteration_count: int
    max_iterations: int
    reasoning_trace: list[str]
    answer: str
    citations: list[int]
    citation_validation: dict
    grounding: dict
    context_stats: dict
    compressed_contexts: list
    mcp_result: dict
    error: str