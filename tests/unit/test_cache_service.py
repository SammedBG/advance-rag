import time
from app.cache.service import RedisCacheService


def test_cache_service_in_memory_operations():
    cache = RedisCacheService(redis_url="redis://localhost:9999/0", enabled=True)

    # 1. Query Cache Set & Get
    query = "What causes pod crash?"
    data = {"answer": "Out of memory", "score": 0.95}

    cache.set_query_cache(query=query, data=data, ttl_seconds=60)
    cached = cache.get_query_cache(query=query)

    assert cached is not None
    assert cached["answer"] == "Out of memory"
    assert cached["score"] == 0.95

    # 2. Embedding Cache Set & Get
    vec = [0.1, 0.2, 0.3, 0.4]
    cache.set_embedding_cache("kubernetes pod", vec, ttl_seconds=60)
    cached_vec = cache.get_embedding_cache("kubernetes pod")

    assert cached_vec is not None
    assert len(cached_vec) == 4
    assert cached_vec[0] == 0.1

    # 3. Cache Miss
    miss = cache.get_query_cache("Uncached non-existent query")
    assert miss is None

    # 4. Cache Stats
    stats = cache.stats
    assert stats["hits"] >= 2
    assert stats["misses"] >= 1
    assert stats["hit_ratio"] > 0.0

    # 5. Clear Cache
    cache.clear()
    after_clear = cache.get_query_cache(query=query)
    assert after_clear is None


def test_cache_expiry():
    cache = RedisCacheService(redis_url="redis://localhost:9999/0", enabled=True)

    query = "Short lived query"
    cache.set_query_cache(query=query, data={"val": 123}, ttl_seconds=1)

    # Immediate get -> hit
    assert cache.get_query_cache(query=query) is not None

    # Wait for expiration
    time.sleep(1.1)
    assert cache.get_query_cache(query=query) is None
