# Redis Caching and Memory Management

## Eviction Policies

When Redis memory usage exceeds `maxmemory`, an eviction policy dictates how keys are dropped.

### Eviction Strategies

Common eviction policies include:
- `allkeys-lru`: Evicts the least recently used keys out of all keys.
- `volatile-lru`: Evicts least recently used keys among those with an expiration set.
- `allkeys-lfu`: Evicts the least frequently used keys out of all keys.
- `noeviction`: Returns errors on write operations when memory limit is reached.

## Cache Invalidation Patterns

### Cache-Aside (Lazy Loading)

The application first checks Redis cache:
1. If cache hit, return cached data immediately.
2. If cache miss, query primary database, store result in Redis with a TTL, and return.

### Write-Through and Write-Behind

- Write-Through updates cache and database synchronously.
- Write-Behind writes to cache first and asynchronously persists to database.
