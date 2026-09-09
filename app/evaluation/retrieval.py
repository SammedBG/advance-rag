from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class RetrievalMetrics:
    query_id: str
    query: str
    mrr: float = 0.0
    hit_rate_at_1: float = 0.0
    hit_rate_at_3: float = 0.0
    hit_rate_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    total_retrieved: int = 0
    total_relevant: int = 0


class RetrievalEvaluator:
    """
    Evaluates retrieval performance using standard IR metrics:
    HitRate@K, Precision@K, Recall@K, MRR, and nDCG@K.
    """

    def is_relevant(
        self,
        item: Any,
        relevant_titles: list[str],
        relevant_keywords: list[str],
    ) -> bool:
        title = ""
        content = ""

        if hasattr(item, "chunk") and item.chunk is not None:
            chunk = item.chunk
            title = getattr(chunk, "title", "") or ""
            content = getattr(chunk, "content", "") or ""
        elif hasattr(item, "title"):
            title = getattr(item, "title", "") or ""
            content = getattr(item, "content", "") or ""
        elif isinstance(item, dict):
            title = item.get("title", "") or ""
            content = item.get("content", "") or ""

        # Title match
        if any(t.lower() in title.lower() for t in relevant_titles if t):
            return True

        # Keyword match (at least 2 keyword occurrences or 1 if few keywords)
        content_lower = content.lower()
        matched_kw = sum(1 for kw in relevant_keywords if kw.lower() in content_lower)
        min_matches = 1 if len(relevant_keywords) <= 2 else 2
        return matched_kw >= min_matches

    def evaluate_query(
        self,
        query_id: str,
        query: str,
        retrieved_items: list[Any],
        relevant_titles: list[str],
        relevant_keywords: list[str],
        total_ground_truth_relevant: int = 1,
    ) -> RetrievalMetrics:
        if not retrieved_items:
            return RetrievalMetrics(query_id=query_id, query=query)

        # Boolean relevance array for retrieved items
        relevance = [
            1 if self.is_relevant(item, relevant_titles, relevant_keywords) else 0
            for item in retrieved_items
        ]

        total_relevant = sum(relevance)
        total_gt = max(total_ground_truth_relevant, total_relevant, 1)

        # MRR
        mrr = 0.0
        for rank, rel in enumerate(relevance, start=1):
            if rel == 1:
                mrr = 1.0 / rank
                break

        def hit_rate_at_k(k: int) -> float:
            sub = relevance[:k]
            return 1.0 if any(r == 1 for r in sub) else 0.0

        def precision_at_k(k: int) -> float:
            sub = relevance[:k]
            if not sub:
                return 0.0
            return sum(sub) / len(sub)

        def recall_at_k(k: int) -> float:
            sub = relevance[:k]
            return sum(sub) / total_gt

        def ndcg_at_k(k: int) -> float:
            sub = relevance[:k]
            dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(sub, start=1))
            # Ideal DCG with all 1s at top
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, total_gt) + 1))
            if idcg == 0.0:
                return 0.0
            return dcg / idcg

        return RetrievalMetrics(
            query_id=query_id,
            query=query,
            mrr=round(mrr, 4),
            hit_rate_at_1=round(hit_rate_at_k(1), 4),
            hit_rate_at_3=round(hit_rate_at_k(3), 4),
            hit_rate_at_5=round(hit_rate_at_k(5), 4),
            precision_at_1=round(precision_at_k(1), 4),
            precision_at_3=round(precision_at_k(3), 4),
            precision_at_5=round(precision_at_k(5), 4),
            recall_at_1=round(recall_at_k(1), 4),
            recall_at_3=round(recall_at_k(3), 4),
            recall_at_5=round(recall_at_k(5), 4),
            ndcg_at_5=round(ndcg_at_k(5), 4),
            total_retrieved=len(retrieved_items),
            total_relevant=total_relevant,
        )
