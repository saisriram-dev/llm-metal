from tenacity import (
    retry, stop_after_attempt,
    retry_if_exception_type, wait_exponential, wait_random, before_sleep_log
)

import logging

logger = logging.getLogger(__name__)

@retry(
        # Stop retrying after 3 total attempts (1 initial + 2 retries)
        stop=stop_after_attempt(3),

        # Wait time between retries:
        # - wait_exponential: starts at 2s, doubles each retry (2s, 4s, 8s...), capped at 20s
        # - wait_random(0, 2): adds a random jitter of 0-2s on top, to avoid retry storms
        #   when multiple callers fail at the same time
        wait=wait_exponential(multiplier=1, min=2, max=20) + wait_random(0, 2),

        # Only retry if the exception raised is a TimeoutError or ConnectionError.
        # Any other exception type will propagate immediately without retrying.
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),

        # Before each sleep/retry, log a warning message using the given logger,
        # so you can see retry attempts happening in your logs
        before_sleep=before_sleep_log(logger, logging.WARNING),

        # If all retry attempts are exhausted, re-raise the original exception
        # instead of raising tenacity's own RetryError wrapper
        reraise=True,
)
def some_function():
    # Example function that may raise an exception
    pass
