from collections import defaultdict
import threading
import time
from typing import Any


class MetricsRegistry:
    """
    Thread-safe Prometheus-compatible metrics registry and collector.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: dict[str, dict[tuple[tuple[str, str], ...], list[float]]] = defaultdict(lambda: defaultdict(list))
        self._help: dict[str, str] = {}
        self._types: dict[str, str] = {}

    def register(self, name: str, metric_type: str, help_text: str) -> None:
        with self._lock:
            self._types[name] = metric_type
            self._help[name] = help_text

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        with self._lock:
            if name not in self._types:
                self._types[name] = "counter"
                self._help[name] = help_text or f"Counter for {name}"
            label_key = self._format_label_key(labels)
            self._counters[name][label_key] += value

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        with self._lock:
            if name not in self._types:
                self._types[name] = "gauge"
                self._help[name] = help_text or f"Gauge for {name}"
            label_key = self._format_label_key(labels)
            self._gauges[name][label_key] = value

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        with self._lock:
            if name not in self._types:
                self._types[name] = "histogram"
                self._help[name] = help_text or f"Histogram for {name}"
            label_key = self._format_label_key(labels)
            self._histograms[name][label_key].append(value)
            # Cap stored samples to avoid unbounded memory growth
            if len(self._histograms[name][label_key]) > 1000:
                self._histograms[name][label_key] = self._histograms[name][label_key][-1000:]

    def _format_label_key(self, labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        if not labels:
            return ()
        return tuple(sorted((str(k), str(v)) for k, v in labels.items()))

    def _render_labels(self, label_key: tuple[tuple[str, str], ...]) -> str:
        if not label_key:
            return ""
        items = [f'{k}="{v}"' for k, v in label_key]
        return "{" + ",".join(items) + "}"

    def generate_prometheus_text(self) -> str:
        lines: list[str] = []
        with self._lock:
            all_metrics = set(self._types.keys())

            for name in sorted(all_metrics):
                m_type = self._types.get(name, "untyped")
                help_text = self._help.get(name, name)
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {m_type}")

                if m_type == "counter":
                    for label_key, val in sorted(self._counters[name].items()):
                        lbl = self._render_labels(label_key)
                        lines.append(f"{name}{lbl} {val}")

                elif m_type == "gauge":
                    for label_key, val in sorted(self._gauges[name].items()):
                        lbl = self._render_labels(label_key)
                        lines.append(f"{name}{lbl} {val}")

                elif m_type == "histogram":
                    for label_key, samples in sorted(self._histograms[name].items()):
                        lbl = self._render_labels(label_key)
                        count = len(samples)
                        total = sum(samples) if samples else 0.0
                        # Standard Prometheus summary / histogram output
                        prefix = name
                        lines.append(f"{prefix}_count{lbl} {count}")
                        lines.append(f"{prefix}_sum{lbl} {total:.4f}")
                        if count > 0:
                            sorted_s = sorted(samples)
                            p50 = sorted_s[int(count * 0.50)]
                            p95 = sorted_s[min(int(count * 0.95), count - 1)]
                            p99 = sorted_s[min(int(count * 0.99), count - 1)]
                            # Add quantiles
                            base_labels = dict(label_key)
                            for q, val in [("0.5", p50), ("0.95", p95), ("0.99", p99)]:
                                q_labels = dict(base_labels)
                                q_labels["quantile"] = q
                                q_lbl = self._render_labels(tuple(sorted(q_labels.items())))
                                lines.append(f"{prefix}{q_lbl} {val:.4f}")

                lines.append("")

        return "\n".join(lines).strip() + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Global Registry Instance
metrics_registry = MetricsRegistry()

# Initialize core platform metric definitions
metrics_registry.register(
    "rag_http_requests_total",
    "counter",
    "Total count of HTTP requests handled by the platform.",
)
metrics_registry.register(
    "rag_http_request_duration_seconds",
    "histogram",
    "Latency distribution of HTTP requests.",
)
metrics_registry.register(
    "rag_retrieval_queries_total",
    "counter",
    "Total retrieval queries executed.",
)
metrics_registry.register(
    "rag_retrieval_latency_seconds",
    "histogram",
    "Latency distribution of retrieval pipeline stages.",
)
metrics_registry.register(
    "rag_llm_tokens_total",
    "counter",
    "Total LLM tokens consumed by generation.",
)
metrics_registry.register(
    "rag_cache_operations_total",
    "counter",
    "Total cache operations (hits/misses).",
)
metrics_registry.register(
    "rag_grounding_score",
    "histogram",
    "Distribution of answer grounding evaluation scores.",
)
