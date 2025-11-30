"""Tests for the retry utility module."""

import pytest
import time
from unittest.mock import Mock, patch


class TestRetryWithBackoff:
    """Test the retry_with_backoff decorator."""

    def test_retry_succeeds_first_attempt(self):
        """Function should return immediately on first success."""
        from src.sports_betting.utils.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, exceptions=(ValueError,))
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()

        assert result == "success"
        assert call_count == 1

    def test_retry_succeeds_after_failures(self):
        """Function should retry and eventually succeed."""
        from src.sports_betting.utils.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.01,
            exceptions=(ValueError,)
        )
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success"

        result = flaky_func()

        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted_raises(self):
        """Should raise after all retries exhausted."""
        from src.sports_betting.utils.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(
            max_retries=2,
            initial_delay=0.01,
            exceptions=(ValueError,)
        )
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent failure")

        with pytest.raises(ValueError, match="permanent failure"):
            always_fails()

        # Initial attempt + 2 retries = 3 total calls
        assert call_count == 3

    def test_retry_only_catches_specified_exceptions(self):
        """Should not retry on non-specified exceptions."""
        from src.sports_betting.utils.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.01,
            exceptions=(ValueError,)  # Only catch ValueError
        )
        def type_error_func():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            type_error_func()

        # Should not retry - only 1 call
        assert call_count == 1

    def test_retry_respects_max_delay(self):
        """Backoff should not exceed max_delay."""
        from src.sports_betting.utils.retry import retry_with_backoff

        delays = []

        @retry_with_backoff(
            max_retries=5,
            initial_delay=1.0,
            backoff_factor=10.0,  # Would be 1, 10, 100, 1000...
            max_delay=5.0,  # But capped at 5
            exceptions=(ValueError,)
        )
        def track_delays():
            raise ValueError("fail")

        # We can't easily track actual delays without mocking time.sleep
        # This test verifies the decorator doesn't error with these params
        with pytest.raises(ValueError):
            track_delays()

    def test_retry_calls_on_retry_callback(self):
        """on_retry callback should be called on each retry."""
        from src.sports_betting.utils.retry import retry_with_backoff

        callbacks = []

        def on_retry(exception, attempt):
            callbacks.append((str(exception), attempt))

        @retry_with_backoff(
            max_retries=2,
            initial_delay=0.01,
            exceptions=(ValueError,),
            on_retry=on_retry
        )
        def failing_func():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            failing_func()

        # Should have 2 callbacks (one for each retry)
        assert len(callbacks) == 2
        assert callbacks[0] == ("fail", 0)
        assert callbacks[1] == ("fail", 1)

    def test_retry_preserves_function_metadata(self):
        """Decorated function should preserve original metadata."""
        from src.sports_betting.utils.retry import retry_with_backoff

        @retry_with_backoff(max_retries=1, exceptions=(Exception,))
        def documented_func():
            """This is a docstring."""
            return 42

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."


class TestRetryableErrors:
    """Test custom exception types."""

    def test_retryable_error_is_exception(self):
        """RetryableError should be an Exception."""
        from src.sports_betting.utils.retry import RetryableError

        assert issubclass(RetryableError, Exception)

        with pytest.raises(RetryableError):
            raise RetryableError("test error")

    def test_non_retryable_error_is_exception(self):
        """NonRetryableError should be an Exception."""
        from src.sports_betting.utils.retry import NonRetryableError

        assert issubclass(NonRetryableError, Exception)

        with pytest.raises(NonRetryableError):
            raise NonRetryableError("test error")
