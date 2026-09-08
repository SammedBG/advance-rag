from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    AgentDependencies,
    generate_node,
    retrieve_node,
    route_node,
)
from app.agent.state import AgentState


def route_after_router(state: AgentState) -> str:
    route = state.get("route", "rag")

    if route == "rag":
        return "retrieve"

    if route == "api":
        return "api"

    return "retrieve"


def api_placeholder_node(state: AgentState) -> AgentState:
    state["answer"] = (
        "The API tool route has been selected, "
        "but API integration has not been implemented yet."
    )

    return state


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
        "api",
        api_placeholder_node,
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
            "api": "api",
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
        "api",
        END,
    )

    return graph.compile()