from fastapi.testclient import TestClient
import pytest

from app.core.metrics import MetricsRegistry, metrics_registry
from app.main import app


def test_metrics_registry_counters():
    registry = MetricsRegistry()
    registry.increment_counter("test_queries_total", value=1.0, labels={"status": "success"})
    registry.increment_counter("test_queries_total", value=2.0, labels={"status": "success"})
    registry.increment_counter("test_queries_total", value=1.0, labels={"status": "error"})

    text = registry.generate_prometheus_text()
    assert "# TYPE test_queries_total counter" in text
    assert 'test_queries_total{status="success"} 3.0' in text
    assert 'test_queries_total{status="error"} 1.0' in text


def test_metrics_registry_gauges_and_histograms():
    registry = MetricsRegistry()
    registry.set_gauge("memory_usage_bytes", value=1024.0, labels={"host": "worker-1"})
    
    # Observe some latencies
    for val in [0.01, 0.02, 0.05, 0.10, 0.25]:
        registry.observe_histogram("query_duration_seconds", value=val, labels={"route": "rag"})

    text = registry.generate_prometheus_text()
    assert "# TYPE memory_usage_bytes gauge" in text
    assert 'memory_usage_bytes{host="worker-1"} 1024.0' in text
    assert "# TYPE query_duration_seconds histogram" in text
    assert 'query_duration_seconds_count{route="rag"} 5' in text
    assert 'query_duration_seconds{quantile="0.5",route="rag"}' in text


def test_metrics_api_endpoint():
    client = TestClient(app)
    
    # Make a health call
    res_health = client.get("/health")
    assert res_health.status_code == 200

    # Scrape metrics
    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    assert "text/plain" in res_metrics.headers["content-type"]
    
    body = res_metrics.text
    assert "rag_http_requests_total" in body
    assert 'endpoint="/health"' in body
