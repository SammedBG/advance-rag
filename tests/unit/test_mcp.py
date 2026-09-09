import pytest
from app.mcp import tools
from app.mcp.service import MCPService


def test_mcp_pod_status_tool_running():
    res = tools.get_pod_status(pod_name="auth-service-pod-1", namespace="production")
    assert res["pod_name"] == "auth-service-pod-1"
    assert res["namespace"] == "production"
    assert res["status"] == "Running"
    assert res["ready"] is True
    assert res["restarts"] == 0


def test_mcp_pod_status_tool_failing():
    res = tools.get_pod_status(pod_name="worker-crash-node", namespace="default")
    assert res["status"] == "CrashLoopBackOff"
    assert res["ready"] is False
    assert res["restarts"] > 0


def test_mcp_cluster_health_tool():
    res = tools.get_cluster_health()
    assert res["status"] == "Healthy"
    assert res["nodes_total"] == 5
    assert res["nodes_ready"] == 5


def test_mcp_service_local_fallback():
    service = MCPService(server_url="http://127.0.0.1:9999/nonexistent")
    
    # Tool listing fallback
    tools_list = service.list_tools()
    assert "get_pod_status_tool" in tools_list
    assert "get_cluster_health_tool" in tools_list
    assert "search_knowledge_base_tool" in tools_list

    # Pod status fallback
    pod_res = service.get_pod_status("payment-service", "default")
    assert pod_res.success is True
    assert pod_res.tool_name == "get_pod_status_tool"
    assert pod_res.content["pod_name"] == "payment-service"

    # Cluster health fallback
    cluster_res = service.get_cluster_health()
    assert cluster_res.success is True
    assert cluster_res.content["status"] == "Healthy"
