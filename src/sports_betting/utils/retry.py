"""Retry utilities with exponential backoff.

Usage:
    from sports_betting.utils.retry import retry_with_backoff

    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def fetch_data(url):
        return requests.get(url)
"""

import time
import logging
from functools import wraps
from typing import Tuple, Type, Callable, TypeVar, Any

T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] = None,
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_factor: Multiplier for delay between retries (default: 2.0)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        exceptions: Tuple of exception types to catch and retry on
        on_retry: Optional callback function(exception, attempt) called on each retry

    Returns:
        Decorated function that will retry on specified exceptions

    Example:
        @retry_with_backoff(max_retries=3, exceptions=(ConnectionError,))
        def flaky_api_call():
            response = requests.get("https://api.example.com")
            response.raise_for_status()
            return response.json()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            logger = logging.getLogger(func.__module__)
            last_exception = None
            delay = initial_delay

            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Final attempt failed
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise

                    # Log retry
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    # Call optional retry callback
                    if on_retry:
                        on_retry(e, attempt)

                    # Wait before retry
                    time.sleep(delay)

                    # Calculate next delay with exponential backoff
                    delay = min(delay * backoff_factor, max_delay)

            # Should never reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


class RetryableError(Exception):
    """Base exception for errors that should trigger a retry."""
    pass


class NonRetryableError(Exception):
    """Base exception for errors that should NOT trigger a retry."""
    pass
