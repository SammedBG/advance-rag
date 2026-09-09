from datetime import datetime, timezone
import hashlib
import json
import logging
import time
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisCacheService:
    """
    High-performance caching service with Redis backend and automatic
    in-memory TTL fallback for offline / test environments.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.redis_url = redis_url or settings.redis_url
        self.enabled = enabled
        self._redis = None
        self._memory_cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry_timestamp)
        self._hits = 0
        self._misses = 0

        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        try:
            import redis

            client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
            client.ping()
            self._redis = client
            logger.info("Connected to Redis cache at %s", self.redis_url)
        except Exception as exc:
            logger.warning(
                "Could not connect to Redis at '%s' (%s). Using in-memory TTL cache.",
                self.redis_url,
                exc,
            )
            self._redis = None

    @staticmethod
    def _make_key(prefix: str, content: str, extra: Any = None) -> str:
        data_str = f"{content.strip().lower()}:{json.dumps(extra, sort_keys=True) if extra else ''}"
        hashed = hashlib.sha256(data_str.encode("utf-8")).hexdigest()
        return f"{prefix}:{hashed}"

    def get_query_cache(
        self,
        query: str,
        filters: Any = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        key = self._make_key("query", query, filters)

        # 1. Try Redis
        if self._redis is not None:
            try:
                val = self._redis.get(key)
                if val is not None:
                    self._hits += 1
                    logger.debug("Redis cache HIT for key '%s'", key)
                    return json.loads(val)
            except Exception as exc:
                logger.warning("Redis get error for '%s': %s", key, exc)

        # 2. Try In-Memory fallback
        if key in self._memory_cache:
            val, expiry = self._memory_cache[key]
            if time.time() < expiry:
                self._hits += 1
                logger.debug("In-memory cache HIT for key '%s'", key)
                return json.loads(val) if isinstance(val, str) else val
            else:
                del self._memory_cache[key]

        self._misses += 1
        return None

    def set_query_cache(
        self,
        query: str,
        data: dict[str, Any],
        filters: Any = None,
        ttl_seconds: int = 300,
    ) -> None:
        if not self.enabled:
            return

        key = self._make_key("query", query, filters)
        serialized = json.dumps(data)

        # 1. Set in Redis
        if self._redis is not None:
            try:
                self._redis.setex(key, ttl_seconds, serialized)
                logger.debug("Stored query cache in Redis for key '%s'", key)
            except Exception as exc:
                logger.warning("Redis set error for '%s': %s", key, exc)

        # 2. Set in Memory fallback
        self._memory_cache[key] = (serialized, time.time() + ttl_seconds)

    def get_embedding_cache(self, text: str) -> list[float] | None:
        if not self.enabled:
            return None

        key = self._make_key("emb", text)

        if self._redis is not None:
            try:
                val = self._redis.get(key)
                if val is not None:
                    self._hits += 1
                    return json.loads(val)
            except Exception:
                pass

        if key in self._memory_cache:
            val, expiry = self._memory_cache[key]
            if time.time() < expiry:
                self._hits += 1
                return json.loads(val) if isinstance(val, str) else val
            else:
                del self._memory_cache[key]

        self._misses += 1
        return None

    def set_embedding_cache(
        self,
        text: str,
        vector: list[float],
        ttl_seconds: int = 3600,
    ) -> None:
        if not self.enabled:
            return

        key = self._make_key("emb", text)
        serialized = json.dumps(vector)

        if self._redis is not None:
            try:
                self._redis.setex(key, ttl_seconds, serialized)
            except Exception:
                pass

        self._memory_cache[key] = (serialized, time.time() + ttl_seconds)

    def clear(self) -> None:
        if self._redis is not None:
            try:
                self._redis.flushdb()
            except Exception:
                pass
        self._memory_cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        hit_ratio = round(self._hits / total, 3) if total > 0 else 0.0
        return {
            "backend": "redis" if self._redis is not None else "memory",
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": hit_ratio,
            "in_memory_items": len(self._memory_cache),
        }
