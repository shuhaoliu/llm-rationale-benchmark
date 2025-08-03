"""Unit tests for ProviderRequestQueue class."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from rationale_benchmark.llm.concurrent.queue import ProviderRequestQueue
from rationale_benchmark.llm.models import ModelRequest, ModelResponse
from rationale_benchmark.llm.exceptions import ProviderError, ResponseValidationError


@pytest.fixture
def mock_provider():
  """Create a mock LLM provider."""
  provider = AsyncMock()
  provider.name = "test_provider"
  provider.generate_response = AsyncMock()
  return provider


@pytest.fixture
def mock_response_validator():
  """Create a mock response validator."""
  validator = Mock()
  validator.validate_response = Mock()
  return validator


@pytest.fixture
def sample_request():
  """Create a sample ModelRequest."""
  return ModelRequest(
    prompt="Test prompt",
    model="test-model",
    temperature=0.7,
    max_tokens=100
  )


@pytest.fixture
def sample_response():
  """Create a sample ModelResponse."""
  return ModelResponse(
    text="Test response",
    model="test-model",
    provider="test_provider",
    timestamp=datetime.now(),
    latency_ms=500,
    token_count=10
  )


class TestProviderRequestQueue:
  """Test cases for ProviderRequestQueue class."""

  @pytest.mark.asyncio
  async def test_init_creates_queue_and_starts_worker(self, mock_provider, mock_response_validator):
    """Test that initialization creates queue and starts worker task."""
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    assert queue.provider == mock_provider
    assert queue.response_validator == mock_response_validator
    assert isinstance(queue._request_queue, asyncio.Queue)
    assert queue._worker_task is not None
    assert not queue._worker_task.done()
    
    # Cleanup
    await queue.shutdown()

  @pytest.mark.asyncio
  async def test_submit_request_adds_to_queue(self, mock_provider, mock_response_validator, sample_request):
    """Test that submit_request adds request to queue and returns future."""
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    future = await queue.submit_request(sample_request)
    
    assert isinstance(future, asyncio.Future)
    assert queue._request_queue.qsize() == 1
    
    # Cleanup
    await queue.shutdown()

  @pytest.mark.asyncio
  async def test_fifo_processing_order(self, mock_provider, mock_response_validator, sample_response):
    """Test that requests are processed in FIFO order."""
    mock_provider.generate_response.return_value = sample_response
    mock_response_validator.validate_response.return_value = None
    
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    # Submit multiple requests
    requests = []
    futures = []
    for i in range(3):
      request = ModelRequest(
        prompt=f"Test prompt {i}",
        model="test-model",
        temperature=0.7,
        max_tokens=100
      )
      requests.append(request)
      future = await queue.submit_request(request)
      futures.append(future)
    
    # Wait for all requests to complete
    responses = await asyncio.gather(*futures)
    
    # Verify provider was called in order
    assert mock_provider.generate_response.call_count == 3
    calls = mock_provider.generate_response.call_args_list
    
    for i, call in enumerate(calls):
      assert call[0][0].prompt == f"Test prompt {i}"
    
    # Cleanup
    await queue.shutdown()

  @pytest.mark.asyncio
  async def test_response_validation_before_return(self, mock_provider, mock_response_validator, sample_request, sample_response):
    """Test that responses are validated before being returned."""
    mock_provider.generate_response.return_value = sample_response
    mock_response_validator.validate_response.return_value = None
    
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    future = await queue.submit_request(sample_request)
    response = await future
    
    # Verify validation was called
    mock_response_validator.validate_response.assert_called_once_with(sample_response)
    assert response == sample_response
    
    # Cleanup
    await queue.shutdown()

  @pytest.mark.asyncio
  async def test_validation_error_propagated_to_future(self, mock_provider, mock_response_validator, sample_request, sample_response):
    """Test that validation errors are propagated to the request future."""
    mock_provider.generate_response.return_value = sample_response
    validation_error = ResponseValidationError("Invalid response structure")
    mock_response_validator.validate_response.side_effect = validation_error
    
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    future = await queue.submit_request(sample_request)
    
    with pytest.raises(ResponseValidationError) as exc_info:
      await future
    
    assert exc_info.value == validation_error
    
    # Cleanup
    await queue.shutdown()

  @pytest.mark.asyncio
  async def test_provider_error_propagated_to_future(self, mock_provider, mock_response_validator, sample_request):
    """Test that provider errors are propagated to the request future."""
    provider_error = ProviderError("test_provider", "API request failed")
    mock_provider.generate_response.side_effect = provider_error
    
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    future = await queue.submit_request(sample_request)
    
    with pytest.raises(ProviderError) as exc_info:
      await future
    
    assert exc_info.value == provider_error
    
    # Cleanup
    await queue.shutdown()

  @pytest.mark.asyncio
  async def test_concurrent_submissions_processed_sequentially(self, mock_provider, mock_response_validator, sample_response):
    """Test that concurrent submissions are still processed sequentially."""
    # Add delay to provider response to test sequential processing
    async def delayed_response(request):
      await asyncio.sleep(0.1)
      return sample_response
    
    mock_provider.generate_response.side_effect = delayed_response
    mock_response_validator.validate_response.return_value = None
    
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    # Submit requests concurrently
    requests = [
      ModelRequest(prompt=f"Test {i}", model="test-model", temperature=0.7, max_tokens=100)
      for i in range(3)
    ]
    
    futures = await asyncio.gather(*[
      queue.submit_request(request) for request in requests
    ])
    
    # Start timing
    start_time = asyncio.get_event_loop().time()
    
    # Wait for all responses
    await asyncio.gather(*futures)
    
    # End timing
    end_time = asyncio.get_event_loop().time()
    
    # Should take at least 0.3 seconds (3 * 0.1) due to sequential processing
    assert end_time - start_time >= 0.25  # Allow some tolerance
    
    # Cleanup
    await queue.shutdown()

  @pytest.mark.asyncio
  async def test_shutdown_stops_worker_and_cancels_pending(self, mock_provider, mock_response_validator):
    """Test that shutdown stops worker task and cancels pending requests."""
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    # Submit a request but don't wait for it
    request = ModelRequest(prompt="Test", model="test-model", temperature=0.7, max_tokens=100)
    future = await queue.submit_request(request)
    
    # Shutdown immediately
    await queue.shutdown()
    
    # Worker task should be cancelled
    assert queue._worker_task.cancelled() or queue._worker_task.done()
    
    # Future should be cancelled
    assert future.cancelled()

  @pytest.mark.asyncio
  async def test_queue_status_reporting(self, mock_provider, mock_response_validator):
    """Test that queue provides status information."""
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    # Initially empty
    status = queue.get_status()
    assert status["provider"] == "test_provider"
    assert status["queue_size"] == 0
    assert status["is_processing"] == False
    assert status["total_processed"] == 0
    
    # Add requests
    request1 = ModelRequest(prompt="Test 1", model="test-model", temperature=0.7, max_tokens=100)
    request2 = ModelRequest(prompt="Test 2", model="test-model", temperature=0.7, max_tokens=100)
    
    await queue.submit_request(request1)
    await queue.submit_request(request2)
    
    status = queue.get_status()
    assert status["queue_size"] == 2
    
    # Cleanup
    await queue.shutdown()

  @pytest.mark.asyncio
  async def test_worker_handles_unexpected_errors(self, mock_provider, mock_response_validator, sample_request):
    """Test that worker handles unexpected errors gracefully."""
    # Make provider raise an unexpected error
    unexpected_error = RuntimeError("Unexpected error")
    mock_provider.generate_response.side_effect = unexpected_error
    
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    future = await queue.submit_request(sample_request)
    
    with pytest.raises(RuntimeError) as exc_info:
      await future
    
    assert exc_info.value == unexpected_error
    
    # Worker should still be running after error
    assert not queue._worker_task.done()
    
    # Cleanup
    await queue.shutdown()

  @pytest.mark.asyncio
  async def test_multiple_requests_with_mixed_success_failure(self, mock_provider, mock_response_validator, sample_response):
    """Test handling of multiple requests with mixed success and failure."""
    # First request succeeds, second fails, third succeeds
    responses = [
      sample_response,
      ProviderError("test_provider", "API error"),
      sample_response
    ]
    mock_provider.generate_response.side_effect = responses
    mock_response_validator.validate_response.return_value = None
    
    queue = ProviderRequestQueue(mock_provider, mock_response_validator)
    
    # Submit three requests
    futures = []
    for i in range(3):
      request = ModelRequest(prompt=f"Test {i}", model="test-model", temperature=0.7, max_tokens=100)
      future = await queue.submit_request(request)
      futures.append(future)
    
    # First request should succeed
    response1 = await futures[0]
    assert response1 == sample_response
    
    # Second request should fail
    with pytest.raises(ProviderError):
      await futures[1]
    
    # Third request should succeed
    response3 = await futures[2]
    assert response3 == sample_response
    
    # Cleanup
    await queue.shutdown()

  @pytest.mark.asyncio
  async def test_request_timeout_handling(self, mock_provider, mock_response_validator, sample_request):
    """Test that request timeouts are handled properly."""
    # Make provider hang indefinitely
    async def hanging_response(request):
      await asyncio.sleep(10)  # Long delay
      return sample_response
    
    mock_provider.generate_response.side_effect = hanging_response
    
    queue = ProviderRequestQueue(mock_provider, mock_response_validator, request_timeout=0.1)
    
    future = await queue.submit_request(sample_request)
    
    with pytest.raises(asyncio.TimeoutError):
      await future
    
    # Cleanup
    await queue.shutdown()