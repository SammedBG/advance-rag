import time
from fastapi import HTTPException
import pytest
from unittest.mock import MagicMock

from app.core.config import settings
from app.core.rate_limiter import SlidingWindowRateLimiter, rate_limit


def test_sliding_window_limiter_basic():
    limiter = SlidingWindowRateLimiter(requests_per_minute=3)
    client_id = "test_client_1"

    # First 3 requests should succeed
    limited, retry_after = limiter.is_rate_limited(client_id)
    assert not limited
    assert retry_after == 0

    limited, retry_after = limiter.is_rate_limited(client_id)
    assert not limited

    limited, retry_after = limiter.is_rate_limited(client_id)
    assert not limited

    # 4th request exceeds limit of 3
    limited, retry_after = limiter.is_rate_limited(client_id)
    assert limited
    assert retry_after > 0

    # Reset limiter for client
    limiter.reset(client_id)
    limited, retry_after = limiter.is_rate_limited(client_id)
    assert not limited


def test_rate_limiter_multiple_clients():
    limiter = SlidingWindowRateLimiter(requests_per_minute=2)
    client_a = "client_a"
    client_b = "client_b"

    # Client A consumes 2 tokens
    assert not limiter.is_rate_limited(client_a)[0]
    assert not limiter.is_rate_limited(client_a)[0]
    assert limiter.is_rate_limited(client_a)[0]

    # Client B should still be unaffected
    assert not limiter.is_rate_limited(client_b)[0]
    assert not limiter.is_rate_limited(client_b)[0]
    assert limiter.is_rate_limited(client_b)[0]


@pytest.mark.anyio
async def test_rate_limit_fastapi_dependency():
    dep = rate_limit(requests_per_minute=2)

    # Mock Request
    request = MagicMock()
    request.client.host = "192.168.1.100"

    # Request 1 & 2 pass
    await dep(request=request, x_api_key=None)
    await dep(request=request, x_api_key=None)

    # Request 3 should raise HTTPException 429
    with pytest.raises(HTTPException) as exc_info:
        await dep(request=request, x_api_key=None)
    
    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers
    assert int(exc_info.value.headers["Retry-After"]) > 0
