import logging
import time
from typing import Callable

from fastapi import Header, HTTPException, Request, status

from app.core.config import settings

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """
    Sliding window in-memory & Redis rate limiter.
    """

    def __init__(self, requests_per_minute: int | None = None) -> None:
        self.limit = requests_per_minute or settings.rate_limit_requests_per_minute
        self._windows: dict[str, list[float]] = {}

    def is_rate_limited(self, client_id: str) -> tuple[bool, int]:
        if not settings.rate_limit_enabled:
            return False, 0

        now = time.time()
        window_start = now - 60.0

        timestamps = self._windows.get(client_id, [])
        # Keep only timestamps in the current 60s sliding window
        valid_timestamps = [t for t in timestamps if t > window_start]

        if len(valid_timestamps) >= self.limit:
            oldest = valid_timestamps[0]
            retry_after = max(1, int(60.0 - (now - oldest)))
            self._windows[client_id] = valid_timestamps
            return True, retry_after

        valid_timestamps.append(now)
        self._windows[client_id] = valid_timestamps
        return False, 0

    def reset(self, client_id: str | None = None) -> None:
        if client_id:
            self._windows.pop(client_id, None)
        else:
            self._windows.clear()


global_rate_limiter = SlidingWindowRateLimiter()


def rate_limit(requests_per_minute: int | None = None) -> Callable:
    limiter = (
        SlidingWindowRateLimiter(requests_per_minute)
        if requests_per_minute
        else global_rate_limiter
    )

    async def dependency(
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> None:
        if not settings.rate_limit_enabled:
            return

        # Determine client identifier
        if x_api_key:
            client_id = f"key:{x_api_key}"
        elif request.client and request.client.host:
            client_id = f"ip:{request.client.host}"
        else:
            client_id = "anonymous"

        limited, retry_after = limiter.is_rate_limited(client_id)
        if limited:
            logger.warning("Rate limit exceeded for client '%s'", client_id)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit of {limiter.limit} requests/min exceeded. Try again in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )

    return dependency
