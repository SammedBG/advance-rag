from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    AgentDependencies,
    generate_node,
    mcp_node,
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

    raise ValueError(
        f"Invalid agent route: {route}"
    )


def build_rag_graph(
    dependencies: AgentDependencies,
):
    graph = StateGraph(AgentState)

    graph.add_node(
        "route",
        route_node,
    )

    graph.add_node(
        "retrieve",
        partial(
            retrieve_node,
            dependencies=dependencies,
        ),
    )

    graph.add_node(
        "generate",
        partial(
            generate_node,
            dependencies=dependencies,
        ),
    )

    graph.add_node(
        "mcp",
        partial(
            mcp_node,
            dependencies=dependencies,
        ),
    )

    graph.add_edge(
        START,
        "route",
    )

    graph.add_conditional_edges(
        "route",
        route_after_router,
        {
            "retrieve": "retrieve",
            "mcp": "mcp",
        },
    )

    graph.add_edge(
        "retrieve",
        "generate",
    )

    graph.add_edge(
        "generate",
        END,
    )

    graph.add_edge(
        "mcp",
        END,
    )

    return graph.compile()