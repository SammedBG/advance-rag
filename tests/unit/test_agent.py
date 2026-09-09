from unittest.mock import MagicMock
import pytest

from app.agent.nodes import (
    AgentDependencies,
    _extract_pod_name,
    generate_node,
    mcp_node,
    refine_node,
    route_node,
)
from app.agent.service import RAGAgent
from app.agent.state import AgentState
from app.context.compressor import CompressedContext


def test_extract_pod_name():
    assert _extract_pod_name("get pod status for auth-service-789") == "auth-service-789"
    assert _extract_pod_name("show status of pod backend-api") == "backend-api"
    assert _extract_pod_name("check pod worker-node-1") == "worker-node-1"
    assert _extract_pod_name("what is kubernetes architecture?") is None


def test_route_node_decisions():
    # MCP pod query
    state_mcp: AgentState = {"query": "get pod status for payment-app"}
    res_mcp = route_node(state_mcp)
    assert res_mcp["route"] == "mcp"
    assert len(res_mcp["reasoning_trace"]) > 0

    # MCP cluster health query
    state_cluster: AgentState = {"query": "check cluster health and node status"}
    res_cluster = route_node(state_cluster)
    assert res_cluster["route"] == "mcp"

    # Standard RAG query
    state_rag: AgentState = {"query": "how does Redis eviction work?"}
    res_rag = route_node(state_rag)
    assert res_rag["route"] == "rag"


def test_refine_node_query_rewrite():
    state: AgentState = {
        "query": "explain crashloopbackoff in k8s",
        "iteration_count": 0,
        "max_iterations": 2,
        "grounding": {"score": 0.1, "grounded": False, "unmatched_terms": ["crashloopbackoff", "pod"]},
        "reasoning_trace": [],
    }
    refined = refine_node(state)
    assert refined["iteration_count"] == 1
    assert "crashloopbackoff" in refined["query"]
    assert len(refined["reasoning_trace"]) == 1


def test_agent_end_to_end_mcp_routing():
    # Mock dependencies
    mock_search = MagicMock()
    mock_expander = MagicMock()
    mock_selector = MagicMock()
    mock_compressor = MagicMock()
    mock_gen = MagicMock()
    mock_grounding = MagicMock()
    
    from app.mcp.service import MCPService
    mcp_service = MCPService(server_url="http://127.0.0.1:9999/test")

    deps = AgentDependencies(
        search_service=mock_search,
        parent_expander=mock_expander,
        context_selector=mock_selector,
        context_compressor=mock_compressor,
        generation_service=mock_gen,
        grounding_validator=mock_grounding,
        mcp_service=mcp_service,
    )

    agent = RAGAgent(dependencies=deps)

    # Run MCP pod status query
    res = agent.run(query="get pod status for web-frontend")
    assert res["route"] == "mcp"
    assert "web-frontend" in res["answer"]
    assert res["mcp_result"]["success"] is True
    assert len(res["reasoning_trace"]) >= 2

    # Run MCP cluster health query
    res_cluster = agent.run(query="check cluster health")
    assert res_cluster["route"] == "mcp"
    assert "Cluster" in res_cluster["answer"]
    assert "Healthy" in res_cluster["answer"]
