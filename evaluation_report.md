# RAG Pipeline Evaluation Scorecard

**Total Evaluated Queries:** 7

## 1. Summary Metrics

| Dimension | Metric | Score |
|---|---|---|
| **Retrieval** | MRR | `0.9286` |
| **Retrieval** | HitRate@1 | `0.8571` |
| **Retrieval** | HitRate@5 | `1.0000` |
| **Retrieval** | Precision@5 | `0.2000` |
| **Retrieval** | Recall@5 | `1.0000` |
| **Retrieval** | nDCG@5 | `0.9473` |
| **Context** | Token Compression Ratio | `0.7970` |
| **Context** | Keyword Retention | `0.5666` |
| **Generation** | Faithfulness Score | `0.4323` |
| **Generation** | Grounding Pass Rate | `0.5714` |
| **Generation** | Answer Token F1 | `0.3829` |
| **Citations** | Citation Validity Rate | `1.0000` |

## 2. Detailed Per-Query Results

| Query ID | Category | MRR | Recall@5 | Comp Ratio | Faithfulness | Answer F1 | Valid Citations |
|---|---|---|---|---|---|---|---|
| `k8s_01` | troubleshooting | 1.00 | 1.00 | 0.56 | 0.42 | 0.30 | 1.00 |
| `k8s_02` | troubleshooting | 1.00 | 1.00 | 0.50 | 0.26 | 0.57 | 1.00 |
| `fastapi_01` | factual | 0.50 | 1.00 | 1.00 | 0.23 | 0.22 | 1.00 |
| `fastapi_02` | conceptual | 1.00 | 1.00 | 1.00 | 0.76 | 0.62 | 1.00 |
| `docker_01` | best_practices | 1.00 | 1.00 | 1.00 | 0.48 | 0.20 | 1.00 |
| `redis_01` | factual | 1.00 | 1.00 | 0.80 | 0.64 | 0.40 | 1.00 |
| `redis_02` | architectural | 1.00 | 1.00 | 0.72 | 0.23 | 0.38 | 1.00 |