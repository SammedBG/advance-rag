from typing import Any, Callable
import logging

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from app.models.chunk import DocumentChunk
from app.models.search import SearchFilter

logger = logging.getLogger(__name__)


class MetadataFilterBuilder:
    """
    Builds backend-specific filters from a SearchFilter definition.
    Supports Qdrant payload filters and in-memory predicates for BM25.
    """

    @staticmethod
    def build_qdrant_filter(filters: SearchFilter | dict | None) -> Filter | None:
        if not filters:
            return None

        if isinstance(filters, dict):
            filters = SearchFilter(**filters)

        conditions: list[FieldCondition] = []

        if filters.document_ids:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=filters.document_ids),
                )
            )

        if filters.technology:
            conditions.append(
                FieldCondition(
                    key="metadata.technology",
                    match=MatchValue(value=filters.technology),
                )
            )

        if filters.technology_version:
            conditions.append(
                FieldCondition(
                    key="metadata.technology_version",
                    match=MatchValue(value=filters.technology_version),
                )
            )

        if filters.source_types:
            conditions.append(
                FieldCondition(
                    key="metadata.source_type",
                    match=MatchAny(any=filters.source_types),
                )
            )

        if filters.access_scope:
            conditions.append(
                FieldCondition(
                    key="metadata.access_scope",
                    match=MatchValue(value=filters.access_scope),
                )
            )

        if filters.custom:
            for key, val in filters.custom.items():
                if isinstance(val, list):
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchAny(any=val),
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=val),
                        )
                    )

        if not conditions:
            return None

        return Filter(must=conditions)

    @staticmethod
    def build_chunk_predicate(
        filters: SearchFilter | dict | None,
    ) -> Callable[[DocumentChunk], bool]:
        if not filters:
            return lambda _: True

        if isinstance(filters, dict):
            filters = SearchFilter(**filters)

        def predicate(chunk: DocumentChunk) -> bool:
            if filters.document_ids and chunk.document_id not in filters.document_ids:
                return False

            chunk_meta = chunk.metadata or {}

            if filters.technology and chunk_meta.get("technology") != filters.technology:
                return False

            if (
                filters.technology_version
                and chunk_meta.get("technology_version") != filters.technology_version
            ):
                return False

            if (
                filters.source_types
                and chunk_meta.get("source_type") not in filters.source_types
            ):
                return False

            if (
                filters.access_scope
                and chunk_meta.get("access_scope") != filters.access_scope
            ):
                return False

            if filters.custom:
                for key, val in filters.custom.items():
                    actual = chunk_meta.get(key)
                    if isinstance(val, list):
                        if actual not in val:
                            return False
                    elif actual != val:
                        return False

            return True

        return predicate
