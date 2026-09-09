from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    AgentDependencies,
    generate_node,
    mcp_node,
    refine_node,
    retrieve_node,
    route_node,
)
from app.agent.state import AgentState


def route_after_router(state: AgentState) -> str:
    route = state.get("route")
    if route == "mcp":
        return "mcp"
    if route == "rag":
        return "retrieve"
    raise ValueError(f"Invalid agent route: {route}")


def should_reflect_and_refine(state: AgentState) -> str:
    """
    Self-reflection decision edge:
    If answer grounding is low and we have not exceeded max_iterations, trigger refinement loop.
    """
    grounding = state.get("grounding", {})
    score = grounding.get("score", 1.0)
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 2)
    contexts = state.get("compressed_contexts", [])

    # If no contexts exist, don't loop endlessly; finish immediately
    if not contexts:
        return "end"

    # If grounding is very weak (< 0.2) and we have remaining attempts, refine query
    if score < 0.20 and iteration < max_iter:
        return "refine"

    return "end"


def build_rag_graph(
    dependencies: AgentDependencies,
):
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.add_node(
        "retrieve",
        partial(retrieve_node, dependencies=dependencies),
    )
    graph.add_node(
        "generate",
        partial(generate_node, dependencies=dependencies),
    )
    graph.add_node("refine", refine_node)
    graph.add_node(
        "mcp",
        partial(mcp_node, dependencies=dependencies),
    )

    graph.add_edge(START, "route")

    graph.add_conditional_edges(
        "route",
        route_after_router,
        {
            "retrieve": "retrieve",
            "mcp": "mcp",
        },
    )

    graph.add_edge("retrieve", "generate")

    graph.add_conditional_edges(
        "generate",
        should_reflect_and_refine,
        {
            "refine": "refine",
            "end": END,
        },
    )

    graph.add_edge("refine", "retrieve")
    graph.add_edge("mcp", END)

    return graph.compile()