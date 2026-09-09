from fastapi import APIRouter, Response

from app.core.metrics import metrics_registry

router = APIRouter(tags=["observability"])


@router.get("/metrics", response_class=Response)
def get_metrics() -> Response:
    """
    Expose Prometheus metrics for scraping.
    """
    content = metrics_registry.generate_prometheus_text()
    return Response(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
