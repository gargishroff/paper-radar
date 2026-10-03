"""Throttling, retries and budget accounting shared by every Cohere call."""

import logging
import time
from collections import deque
from collections.abc import Callable

from cohere.errors import ServiceUnavailableError, TooManyRequestsError

from radar.usage import UsageTracker

log = logging.getLogger(__name__)

# Retries are ours, so every request is paced by the limiter and counted.
NO_SDK_RETRIES = {"max_retries": 0}


class RateLimiter:
    """Keeps calls, and optionally inputs, over any 60-second window under per-minute limits."""

    def __init__(self, calls_per_minute: int, inputs_per_minute: int | None = None):
        self._calls = calls_per_minute
        self._inputs = inputs_per_minute
        self._recent: deque[tuple[float, int]] = deque()  # (timestamp, n_inputs)

    def wait(self, n_inputs: int = 1) -> None:
        while True:
            now = time.monotonic()
            while self._recent and now - self._recent[0][0] >= 60:
                self._recent.popleft()
            in_window = sum(n for _, n in self._recent)
            if len(self._recent) < self._calls and (
                self._inputs is None or in_window + n_inputs <= self._inputs
            ):
                self._recent.append((now, n_inputs))
                return
            time.sleep(60 - (now - self._recent[0][0]) + 0.1)


def guarded_call[T](
    fn: Callable[[], T],
    *,
    tracker: UsageTracker,
    limiter: RateLimiter,
    endpoint: str,
    model: str,
    essential: bool,
    billed_units: Callable[[T], int],
    n_inputs: int = 1,
    max_retries: int = 5,
) -> T:
    """Check the budget, then call `fn` under the rate limit, retrying on 429/503."""
    tracker.check(essential=essential)
    for attempt in range(max_retries):
        limiter.wait(n_inputs)
        try:
            result = fn()
        except (TooManyRequestsError, ServiceUnavailableError) as exc:
            # Rejected calls may still count against the monthly cap, so record them too.
            tracker.record(f"{endpoint}_rejected", model, calls=1)
            wait = 60 * (attempt + 1)
            log.warning(
                "%s %s; waiting %ss for the rate window", endpoint, type(exc).__name__, wait
            )
            time.sleep(wait)
            continue
        tracker.record(endpoint, model, calls=1, units=billed_units(result))
        return result
    raise RuntimeError(f"{endpoint} failed after {max_retries} attempts")
