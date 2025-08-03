"""Unit tests for ConcurrentLLMManager class."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from rationale_benchmark.llm.concurrent.manager import ConcurrentLLMManager
from rationale_benchmark.llm.concurrent.queue import ProviderRequestQueue
from rationale_benchmark.llm.concurrent.validator import ResponseValidator
from rationale_benchmark.llm.models import ModelRequest, ModelResponse
from rationale_benchmark.llm.exceptions import ProviderError, ResponseValidationError


@pytest.fixture
def mock_providers():
  """Create mock LLM providers."""
  providers = {}
  for name in ["openai", "anthropic", "gemini"]:
    provider = AsyncMock()
    provider.name = name
    provider.generate_response = AsyncMock()
    providers[name] = provider
  return providers


@pytest.fixture
def mock_response_validator():
  """Create a mock response validator."""
  validator = Mock()
  validator.validate_response = Mock()
  return validator


@pytest.fixture
def sample_requests():
  """Create sample ModelRequests for different providers."""
  return [
    ModelRequest(
      prompt="Test prompt 1",
      model="gpt-4",
      temperature=0.7,
      max_tokens=100
    ),
    ModelRequest(
      prompt="Test prompt 2", 
      model="claude-3-opus",
      temperature=0.8,
      max_tokens=150
    ),
    ModelRequest(
      prompt="Test prompt 3",
      model="gemini-pro",
      temperature=0.6,
      max_tokens=200
    )
  ]


@pytest.fixture
def sample_responses():
  """Create sample ModelResponses."""
  return [
    ModelResponse(
      text="OpenAI response",
      model="gpt-4",
      provider="openai",
      timestamp=datetime.now(),
      latency_ms=1000,
      token_count=20
    ),
    ModelResponse(
      text="Anthropic response",
      model="claude-3-opus", 
      provider="anthropic",
      timestamp=datetime.now(),
      latency_ms=1200,
      token_count=25
    ),
    ModelResponse(
      text="Gemini response",
      model="gemini-pro",
      provider="gemini", 
      timestamp=datetime.now(),
      latency_ms=800,
      token_count=18
    )
  ]


@pytest.fixture
def model_to_provider_mapping():
  """Create model to provider mapping."""
  return {
    "gpt-4": "openai",
    "gpt-3.5-turbo": "openai",
    "claude-3-opus": "anthropic",
    "claude-3-sonnet": "anthropic",
    "gemini-pro": "gemini",
    "gemini-pro-vision": "gemini"
  }


class TestConcurrentLLMManager:
  """Test cases for ConcurrentLLMManager class."""

  @pytest.mark.asyncio
  async def test_init_creates_provider_queues(self, mock_providers, mock_response_validator, model_to_provider_mapping):
    """Test that initialization creates provider queues for all providers."""
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    assert len(manager._provider_queues) == len(mock_providers)
    for provider_name in mock_providers:
      assert provider_name in manager._provider_queues
      assert isinstance(manager._provider_queues[provider_name], ProviderRequestQueue)
    
    # Cleanup
    await manager.shutdown()

  @pytest.mark.asyncio
  async def test_submit_request_routes_to_correct_provider(self, mock_providers, mock_response_validator, model_to_provider_mapping, sample_responses):
    """Test that requests are routed to the correct provider queue."""
    # Setup mock responses
    mock_providers["openai"].generate_response.return_value = sample_responses[0]
    mock_response_validator.validate_response.return_value = None
    
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.7,
      max_tokens=100
    )
    
    future = await manager.submit_request(request)
    response = await future
    
    assert response == sample_responses[0]
    mock_providers["openai"].generate_response.assert_called_once_with(request)
    
    # Cleanup
    await manager.shutdown()

  @pytest.mark.asyncio
  async def test_submit_request_with_unknown_model_raises_error(self, mock_providers, mock_response_validator, model_to_provider_mapping):
    """Test that requests for unknown models raise an error."""
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    request = ModelRequest(
      prompt="Test prompt",
      model="unknown-model",
      temperature=0.7,
      max_tokens=100
    )
    
    with pytest.raises(ValueError) as exc_info:
      await manager.submit_request(request)
    
    assert "Unknown model" in str(exc_info.value)
    assert "unknown-model" in str(exc_info.value)
    
    # Cleanup
    await manager.shutdown()

  @pytest.mark.asyncio
  async def test_submit_multiple_requests_concurrent_execution(self, mock_providers, mock_response_validator, model_to_provider_mapping, sample_requests, sample_responses):
    """Test that multiple requests to different providers execute concurrently."""
    # Setup mock responses with delays to test concurrency
    async def delayed_response(provider_idx, request):
      await asyncio.sleep(0.1)  # Small delay
      return sample_responses[provider_idx]
    
    async def openai_response(req):
      return await delayed_response(0, req)
    async def anthropic_response(req):
      return await delayed_response(1, req)
    async def gemini_response(req):
      return await delayed_response(2, req)
    
    mock_providers["openai"].generate_response.side_effect = openai_response
    mock_providers["anthropic"].generate_response.side_effect = anthropic_response
    mock_providers["gemini"].generate_response.side_effect = gemini_response
    mock_response_validator.validate_response.return_value = None
    
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    # Submit requests concurrently
    start_time = asyncio.get_event_loop().time()
    
    futures = []
    for request in sample_requests:
      future = await manager.submit_request(request)
      futures.append(future)
    
    # Wait for all responses
    responses = await asyncio.gather(*futures)
    
    end_time = asyncio.get_event_loop().time()
    
    # Should complete in roughly 0.1 seconds (concurrent) rather than 0.3 seconds (sequential)
    assert end_time - start_time < 0.25
    assert len(responses) == 3
    
    # Cleanup
    await manager.shutdown()

  @pytest.mark.asyncio
  async def test_submit_multiple_requests_same_provider_sequential(self, mock_providers, mock_response_validator, model_to_provider_mapping, sample_responses):
    """Test that multiple requests to the same provider execute sequentially."""
    # Setup mock response with delay
    async def delayed_response(request):
      await asyncio.sleep(0.1)
      return sample_responses[0]
    
    mock_providers["openai"].generate_response.side_effect = delayed_response
    mock_response_validator.validate_response.return_value = None
    
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    # Submit multiple requests to same provider
    requests = [
      ModelRequest(prompt=f"Test {i}", model="gpt-4", temperature=0.7, max_tokens=100)
      for i in range(3)
    ]
    
    start_time = asyncio.get_event_loop().time()
    
    futures = []
    for request in requests:
      future = await manager.submit_request(request)
      futures.append(future)
    
    # Wait for all responses
    responses = await asyncio.gather(*futures)
    
    end_time = asyncio.get_event_loop().time()
    
    # Should take at least 0.3 seconds (3 * 0.1) due to sequential processing
    assert end_time - start_time >= 0.25
    assert len(responses) == 3
    
    # Cleanup
    await manager.shutdown()

  @pytest.mark.asyncio
  async def test_request_order_preservation(self, mock_providers, mock_response_validator, model_to_provider_mapping):
    """Test that request order is preserved within each provider queue."""
    responses_in_order = []
    
    async def ordered_response(request):
      await asyncio.sleep(0.05)  # Small delay
      response = ModelResponse(
        text=f"Response for: {request.prompt}",
        model=request.model,
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=50,
        token_count=10
      )
      responses_in_order.append(response)
      return response
    
    mock_providers["openai"].generate_response.side_effect = ordered_response
    mock_response_validator.validate_response.return_value = None
    
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    # Submit requests in specific order
    requests = [
      ModelRequest(prompt=f"Request {i}", model="gpt-4", temperature=0.7, max_tokens=100)
      for i in range(5)
    ]
    
    futures = []
    for request in requests:
      future = await manager.submit_request(request)
      futures.append(future)
    
    # Wait for all responses
    responses = await asyncio.gather(*futures)
    
    # Verify order is preserved
    for i, response in enumerate(responses):
      assert f"Request {i}" in response.text
    
    # Cleanup
    await manager.shutdown()

  @pytest.mark.asyncio
  async def test_error_handling_isolated_per_provider(self, mock_providers, mock_response_validator, model_to_provider_mapping, sample_responses):
    """Test that errors in one provider don't affect others."""
    # Make one provider fail
    mock_providers["openai"].generate_response.side_effect = ProviderError("openai", "API error")
    mock_providers["anthropic"].generate_response.return_value = sample_responses[1]
    mock_providers["gemini"].generate_response.return_value = sample_responses[2]
    mock_response_validator.validate_response.return_value = None
    
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    # Submit requests to all providers
    requests = [
      ModelRequest(prompt="Test 1", model="gpt-4", temperature=0.7, max_tokens=100),
      ModelRequest(prompt="Test 2", model="claude-3-opus", temperature=0.7, max_tokens=100),
      ModelRequest(prompt="Test 3", model="gemini-pro", temperature=0.7, max_tokens=100)
    ]
    
    futures = []
    for request in requests:
      future = await manager.submit_request(request)
      futures.append(future)
    
    # First request should fail
    with pytest.raises(ProviderError):
      await futures[0]
    
    # Other requests should succeed
    response2 = await futures[1]
    response3 = await futures[2]
    
    assert response2 == sample_responses[1]
    assert response3 == sample_responses[2]
    
    # Cleanup
    await manager.shutdown()

  @pytest.mark.asyncio
  async def test_get_provider_status(self, mock_providers, mock_response_validator, model_to_provider_mapping):
    """Test that provider status information is available."""
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    status = manager.get_provider_status()
    
    assert isinstance(status, dict)
    assert len(status) == len(mock_providers)
    
    for provider_name in mock_providers:
      assert provider_name in status
      provider_status = status[provider_name]
      assert "provider" in provider_status
      assert "queue_size" in provider_status
      assert "is_processing" in provider_status
      assert "total_processed" in provider_status
    
    # Cleanup
    await manager.shutdown()

  @pytest.mark.asyncio
  async def test_get_overall_status(self, mock_providers, mock_response_validator, model_to_provider_mapping):
    """Test that overall manager status is available."""
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    status = manager.get_overall_status()
    
    assert isinstance(status, dict)
    assert "total_providers" in status
    assert "active_providers" in status
    assert "total_queue_size" in status
    assert "total_processed" in status
    assert "providers" in status
    
    assert status["total_providers"] == len(mock_providers)
    
    # Cleanup
    await manager.shutdown()

  @pytest.mark.asyncio
  async def test_shutdown_stops_all_queues(self, mock_providers, mock_response_validator, model_to_provider_mapping):
    """Test that shutdown stops all provider queues."""
    # Make provider hang so request doesn't complete before shutdown
    async def hanging_response(request):
      await asyncio.sleep(10)
      return ModelResponse(
        text="Response",
        model=request.model,
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=1000,
        token_count=10
      )
    
    mock_providers["openai"].generate_response.side_effect = hanging_response
    
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    # Submit some requests
    request = ModelRequest(prompt="Test", model="gpt-4", temperature=0.7, max_tokens=100)
    future = await manager.submit_request(request)
    
    # Give a moment for request to enter queue
    await asyncio.sleep(0.01)
    
    # Shutdown
    await manager.shutdown()
    
    # Future should be cancelled
    assert future.cancelled()
    
    # All queues should be shut down
    for queue in manager._provider_queues.values():
      assert queue._worker_task.done()

  @pytest.mark.asyncio
  async def test_submit_request_after_shutdown_raises_error(self, mock_providers, mock_response_validator, model_to_provider_mapping):
    """Test that submitting requests after shutdown raises an error."""
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    await manager.shutdown()
    
    request = ModelRequest(prompt="Test", model="gpt-4", temperature=0.7, max_tokens=100)
    
    with pytest.raises(RuntimeError) as exc_info:
      await manager.submit_request(request)
    
    assert "shut down" in str(exc_info.value).lower()

  @pytest.mark.asyncio
  async def test_concurrent_requests_different_models_same_provider(self, mock_providers, mock_response_validator, model_to_provider_mapping, sample_responses):
    """Test concurrent requests for different models from the same provider."""
    async def delayed_response(request):
      await asyncio.sleep(0.1)
      return sample_responses[0]
    
    mock_providers["openai"].generate_response.side_effect = delayed_response
    mock_response_validator.validate_response.return_value = None
    
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    # Submit requests for different OpenAI models
    requests = [
      ModelRequest(prompt="Test 1", model="gpt-4", temperature=0.7, max_tokens=100),
      ModelRequest(prompt="Test 2", model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
    ]
    
    start_time = asyncio.get_event_loop().time()
    
    futures = []
    for request in requests:
      future = await manager.submit_request(request)
      futures.append(future)
    
    responses = await asyncio.gather(*futures)
    
    end_time = asyncio.get_event_loop().time()
    
    # Should take at least 0.2 seconds (sequential processing within same provider)
    assert end_time - start_time >= 0.15
    assert len(responses) == 2
    
    # Cleanup
    await manager.shutdown()

  @pytest.mark.asyncio
  async def test_provider_queue_isolation(self, mock_providers, mock_response_validator, model_to_provider_mapping):
    """Test that provider queues are isolated from each other."""
    # Make OpenAI provider hang
    async def hanging_response(request):
      await asyncio.sleep(10)  # Very long delay
      return ModelResponse(
        text="Delayed response",
        model=request.model,
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=10000,
        token_count=10
      )
    
    # Make Anthropic provider respond quickly
    async def quick_response(request):
      return ModelResponse(
        text="Quick response",
        model=request.model,
        provider="anthropic",
        timestamp=datetime.now(),
        latency_ms=50,
        token_count=10
      )
    
    mock_providers["openai"].generate_response.side_effect = hanging_response
    mock_providers["anthropic"].generate_response.side_effect = quick_response
    mock_response_validator.validate_response.return_value = None
    
    manager = ConcurrentLLMManager(
      providers=mock_providers,
      response_validator=mock_response_validator,
      model_to_provider=model_to_provider_mapping
    )
    
    # Submit requests to both providers
    openai_request = ModelRequest(prompt="Slow", model="gpt-4", temperature=0.7, max_tokens=100)
    anthropic_request = ModelRequest(prompt="Fast", model="claude-3-opus", temperature=0.7, max_tokens=100)
    
    openai_future = await manager.submit_request(openai_request)
    anthropic_future = await manager.submit_request(anthropic_request)
    
    # Anthropic should respond quickly despite OpenAI hanging
    start_time = asyncio.get_event_loop().time()
    anthropic_response = await anthropic_future
    end_time = asyncio.get_event_loop().time()
    
    assert end_time - start_time < 1.0  # Should be very fast
    assert anthropic_response.text == "Quick response"
    
    # Cleanup (this will cancel the hanging OpenAI request)
    await manager.shutdown()
    # The OpenAI future should be cancelled or done after shutdown
    assert openai_future.cancelled() or openai_future.done()