"""Unit tests for retry logic with exponential backoff."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, Mock, patch
from aiohttp.client_exceptions import ClientConnectorError, ClientResponseError

from rationale_benchmark.llm.http.retry import RetryHandler
from rationale_benchmark.llm.exceptions import NetworkError, TimeoutError, RetryExhaustedError


class TestRetryHandler:
  """Test cases for RetryHandler class."""

  @pytest.fixture
  def retry_handler(self):
    """Create RetryHandler instance for testing."""
    return RetryHandler(max_retries=3, base_delay=0.1, max_delay=1.0)

  @pytest.fixture
  def mock_async_func(self):
    """Create mock async function for testing."""
    return AsyncMock()

  def test_retry_handler_initialization(self):
    """Test RetryHandler initialization with default parameters."""
    handler = RetryHandler()
    assert handler.max_retries == 3
    assert handler.base_delay == 1.0
    assert handler.max_delay == 60.0
    assert handler.backoff_factor == 2.0
    assert handler.jitter == True

  def test_retry_handler_initialization_with_custom_params(self):
    """Test RetryHandler initialization with custom parameters."""
    handler = RetryHandler(
      max_retries=5,
      base_delay=0.5,
      max_delay=30.0,
      backoff_factor=1.5,
      jitter=False
    )
    assert handler.max_retries == 5
    assert handler.base_delay == 0.5
    assert handler.max_delay == 30.0
    assert handler.backoff_factor == 1.5
    assert handler.jitter == False

  @pytest.mark.asyncio
  async def test_execute_success_on_first_attempt(self, retry_handler, mock_async_func):
    """Test successful execution on first attempt."""
    expected_result = "success"
    mock_async_func.return_value = expected_result
    
    result = await retry_handler.execute(mock_async_func, "arg1", kwarg1="value1")
    
    assert result == expected_result
    mock_async_func.assert_called_once_with("arg1", kwarg1="value1")

  @pytest.mark.asyncio
  async def test_execute_success_after_retries(self, retry_handler, mock_async_func):
    """Test successful execution after some retries."""
    expected_result = "success"
    
    # Fail twice, then succeed
    mock_async_func.side_effect = [
      ClientConnectorError(connection_key=Mock(), os_error=OSError("Connection failed")),
      ClientConnectorError(connection_key=Mock(), os_error=OSError("Connection failed")),
      expected_result
    ]
    
    start_time = time.time()
    result = await retry_handler.execute(mock_async_func)
    end_time = time.time()
    
    assert result == expected_result
    assert mock_async_func.call_count == 3
    
    # Should have some delay due to retries (but not too much due to small base_delay)
    assert end_time - start_time >= 0.1  # At least base_delay

  @pytest.mark.asyncio
  async def test_execute_max_retries_exceeded(self, retry_handler, mock_async_func):
    """Test behavior when max retries are exceeded."""
    # Always fail
    mock_async_func.side_effect = ClientConnectorError(
      connection_key=Mock(), 
      os_error=OSError("Connection failed")
    )
    
    with pytest.raises(RetryExhaustedError, match="Max retries \\(3\\) exceeded"):
      await retry_handler.execute(mock_async_func)
    
    # Should attempt max_retries + 1 times (initial + retries)
    assert mock_async_func.call_count == 4

  @pytest.mark.asyncio
  async def test_exponential_backoff_timing(self):
    """Test exponential backoff timing calculation."""
    handler = RetryHandler(max_retries=3, base_delay=0.1, backoff_factor=2.0, jitter=False)
    mock_func = AsyncMock()
    mock_func.side_effect = [
      ClientConnectorError(connection_key=Mock(), os_error=OSError("Connection failed")),
      ClientConnectorError(connection_key=Mock(), os_error=OSError("Connection failed")),
      ClientConnectorError(connection_key=Mock(), os_error=OSError("Connection failed")),
      ClientConnectorError(connection_key=Mock(), os_error=OSError("Connection failed"))
    ]
    
    start_time = time.time()
    
    with pytest.raises(RetryExhaustedError):
      await handler.execute(mock_func)
    
    end_time = time.time()
    
    # Expected delays: 0.1, 0.2, 0.4 (exponential backoff)
    # Total expected delay: ~0.7 seconds
    assert end_time - start_time >= 0.6  # Allow some tolerance

  @pytest.mark.asyncio
  async def test_max_delay_cap(self):
    """Test that delay is capped at max_delay."""
    handler = RetryHandler(
      max_retries=5, 
      base_delay=1.0, 
      max_delay=2.0, 
      backoff_factor=10.0,  # Large factor to test capping
      jitter=False
    )
    
    # Test delay calculation
    delay1 = handler._calculate_delay(1)  # Should be 1.0 * 10^1 = 10.0, capped to 2.0
    delay2 = handler._calculate_delay(2)  # Should be 1.0 * 10^2 = 100.0, capped to 2.0
    
    assert delay1 == 2.0
    assert delay2 == 2.0

  @pytest.mark.asyncio
  async def test_jitter_adds_randomness(self):
    """Test that jitter adds randomness to delay calculation."""
    handler = RetryHandler(base_delay=1.0, jitter=True)
    
    delays = []
    for _ in range(10):
      delay = handler._calculate_delay(1)
      delays.append(delay)
    
    # With jitter, delays should vary
    assert len(set(delays)) > 1  # Should have different values
    
    # All delays should be within reasonable range (0.5 to 1.5 of base delay)
    for delay in delays:
      assert 0.5 <= delay <= 3.0  # 2.0 (base * factor) * 1.5 (jitter range)

  @pytest.mark.asyncio
  async def test_jitter_disabled(self):
    """Test delay calculation with jitter disabled."""
    handler = RetryHandler(base_delay=1.0, backoff_factor=2.0, jitter=False)
    
    delay1 = handler._calculate_delay(1)
    delay2 = handler._calculate_delay(1)
    delay3 = handler._calculate_delay(2)
    
    # Without jitter, same attempt should give same delay
    assert delay1 == delay2 == 2.0  # base_delay * backoff_factor^1
    assert delay3 == 4.0  # base_delay * backoff_factor^2

  @pytest.mark.asyncio
  async def test_retryable_exceptions(self, retry_handler, mock_async_func):
    """Test that only retryable exceptions trigger retries."""
    # Test retryable exceptions
    retryable_exceptions = [
      ClientConnectorError(connection_key=Mock(), os_error=OSError("Connection failed")),
      asyncio.TimeoutError("Request timeout"),
      OSError("Network error")
    ]
    
    for exception in retryable_exceptions:
      mock_async_func.reset_mock()
      mock_async_func.side_effect = [exception, "success"]
      
      result = await retry_handler.execute(mock_async_func)
      
      assert result == "success"
      assert mock_async_func.call_count == 2  # Initial attempt + 1 retry

  @pytest.mark.asyncio
  async def test_non_retryable_exceptions(self, retry_handler, mock_async_func):
    """Test that non-retryable exceptions are not retried."""
    # Test non-retryable exceptions
    non_retryable_exceptions = [
      ValueError("Invalid parameter"),
      KeyError("Missing key"),
      TypeError("Wrong type")
    ]
    
    for exception in non_retryable_exceptions:
      mock_async_func.reset_mock()
      mock_async_func.side_effect = exception
      
      with pytest.raises(type(exception)):
        await retry_handler.execute(mock_async_func)
      
      # Should not retry non-retryable exceptions
      assert mock_async_func.call_count == 1

  @pytest.mark.asyncio
  async def test_client_response_error_handling(self, retry_handler, mock_async_func):
    """Test handling of ClientResponseError based on status code."""
    # 5xx errors should be retryable
    server_error = ClientResponseError(
      request_info=Mock(),
      history=(),
      status=500,
      message="Internal Server Error"
    )
    
    mock_async_func.side_effect = [server_error, "success"]
    result = await retry_handler.execute(mock_async_func)
    
    assert result == "success"
    assert mock_async_func.call_count == 2
    
    # 4xx errors should not be retryable
    mock_async_func.reset_mock()
    client_error = ClientResponseError(
      request_info=Mock(),
      history=(),
      status=400,
      message="Bad Request"
    )
    
    mock_async_func.side_effect = client_error
    
    with pytest.raises(ClientResponseError):
      await retry_handler.execute(mock_async_func)
    
    assert mock_async_func.call_count == 1

  @pytest.mark.asyncio
  async def test_retry_with_different_error_types(self, retry_handler, mock_async_func):
    """Test retry behavior with different error types in sequence."""
    mock_async_func.side_effect = [
      ClientConnectorError(connection_key=Mock(), os_error=OSError("Connection failed")),
      asyncio.TimeoutError("Request timeout"),
      OSError("Network error"),
      "success"
    ]
    
    result = await retry_handler.execute(mock_async_func)
    
    assert result == "success"
    assert mock_async_func.call_count == 4

  @pytest.mark.asyncio
  async def test_zero_max_retries(self, mock_async_func):
    """Test behavior with zero max retries."""
    handler = RetryHandler(max_retries=0)
    mock_async_func.side_effect = ClientConnectorError(
      connection_key=Mock(), 
      os_error=OSError("Connection failed")
    )
    
    with pytest.raises(RetryExhaustedError, match="Max retries \\(0\\) exceeded"):
      await handler.execute(mock_async_func)
    
    # Should only attempt once (no retries)
    assert mock_async_func.call_count == 1

  @pytest.mark.asyncio
  async def test_is_retryable_method(self, retry_handler):
    """Test the is_retryable method with various exception types."""
    # Retryable exceptions
    assert retry_handler.is_retryable(
      ClientConnectorError(connection_key=Mock(), os_error=OSError("Connection failed"))
    )
    assert retry_handler.is_retryable(asyncio.TimeoutError("Timeout"))
    assert retry_handler.is_retryable(OSError("Network error"))
    
    # Server errors (5xx) should be retryable
    server_error = ClientResponseError(
      request_info=Mock(),
      history=(),
      status=500,
      message="Internal Server Error"
    )
    assert retry_handler.is_retryable(server_error)
    
    # Client errors (4xx) should not be retryable
    client_error = ClientResponseError(
      request_info=Mock(),
      history=(),
      status=400,
      message="Bad Request"
    )
    assert not retry_handler.is_retryable(client_error)
    
    # Non-retryable exceptions
    assert not retry_handler.is_retryable(ValueError("Invalid value"))
    assert not retry_handler.is_retryable(KeyError("Missing key"))
    assert not retry_handler.is_retryable(TypeError("Wrong type"))

  @pytest.mark.asyncio
  async def test_execute_with_args_and_kwargs(self, retry_handler, mock_async_func):
    """Test execute method properly passes args and kwargs."""
    mock_async_func.return_value = "success"
    
    result = await retry_handler.execute(
      mock_async_func,
      "arg1", "arg2",
      kwarg1="value1",
      kwarg2="value2"
    )
    
    assert result == "success"
    mock_async_func.assert_called_once_with(
      "arg1", "arg2",
      kwarg1="value1",
      kwarg2="value2"
    )

  @pytest.mark.asyncio
  async def test_concurrent_executions(self, retry_handler):
    """Test multiple concurrent executions of retry handler."""
    async def mock_func(delay, fail_count):
      if hasattr(mock_func, 'call_counts'):
        mock_func.call_counts[delay] = mock_func.call_counts.get(delay, 0) + 1
      else:
        mock_func.call_counts = {delay: 1}
      
      if mock_func.call_counts[delay] <= fail_count:
        raise ClientConnectorError(
          connection_key=Mock(), 
          os_error=OSError("Connection failed")
        )
      
      await asyncio.sleep(delay)
      return f"success_{delay}"
    
    # Run multiple concurrent executions
    tasks = [
      retry_handler.execute(mock_func, 0.01, 1),  # Fail once, then succeed
      retry_handler.execute(mock_func, 0.02, 2),  # Fail twice, then succeed
      retry_handler.execute(mock_func, 0.03, 0),  # Succeed immediately
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert results == ["success_0.01", "success_0.02", "success_0.03"]