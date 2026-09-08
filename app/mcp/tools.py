from typing import Any


def get_pod_status(
    pod_name: str,
    namespace: str = "default",
) -> dict[str, Any]:
    """
    Return Kubernetes pod status.

    This is currently a deterministic mock implementation.
    The actual Kubernetes client will replace this logic later.
    """

    return {
        "pod_name": pod_name,
        "namespace": namespace,
        "status": "Running",
        "ready": True,
        "restarts": 0,
    }