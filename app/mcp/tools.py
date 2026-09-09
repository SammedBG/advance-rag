from typing import Any


def get_pod_status(
    pod_name: str,
    namespace: str = "default",
) -> dict[str, Any]:
    """
    Return Kubernetes pod status with structured inspection data.
    """
    # Deterministic simulation based on pod_name
    name_lower = pod_name.lower()
    if "crash" in name_lower or "fail" in name_lower or "err" in name_lower:
        status = "CrashLoopBackOff"
        ready = False
        restarts = 14
    elif "pull" in name_lower or "image" in name_lower:
        status = "ImagePullBackOff"
        ready = False
        restarts = 0
    elif "pend" in name_lower:
        status = "Pending"
        ready = False
        restarts = 0
    else:
        status = "Running"
        ready = True
        restarts = 0

    return {
        "pod_name": pod_name,
        "namespace": namespace,
        "status": status,
        "ready": ready,
        "restarts": restarts,
        "node": "worker-node-01",
        "ip": "10.244.0.15",
    }


def get_cluster_health() -> dict[str, Any]:
    """
    Return simulated cluster health and node statistics.
    """
    return {
        "cluster_name": "production-k8s-cluster",
        "status": "Healthy",
        "nodes_total": 5,
        "nodes_ready": 5,
        "cpu_utilization_pct": 42.5,
        "memory_utilization_pct": 68.1,
    }


def search_knowledge_base(
    query: str,
    limit: int = 3,
    technology: str | None = None,
) -> dict[str, Any]:
    """
    Direct MCP knowledge base search tool.
    """
    try:
        from app.services.container import ServiceContainer
        container = ServiceContainer()
        hybrid_search = container.get_search_service()
        results = hybrid_search.search(
            query=query,
            limit=limit,
            retrieval_limit=10,
            technology=technology,
        )
        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "title": r.title,
                    "content": r.content,
                    "score": round(r.score, 4),
                    "technology": r.metadata.get("technology") if r.metadata else None,
                }
                for r in results
            ],
        }
    except Exception as exc:
        return {
            "query": query,
            "count": 0,
            "results": [],
            "error": str(exc),
        }