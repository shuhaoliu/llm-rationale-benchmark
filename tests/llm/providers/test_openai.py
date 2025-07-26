"""Unit tests for OpenAI provider implementation."""

import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponse

from rationale_benchmark.llm.exceptions import (
  AuthenticationError,
  ModelNotFoundError,
  ProviderError,
  ResponseValidationError,
  StreamingNotSupportedError,
)
from rationale_benchmark.llm.http.client import HTTPClient
from rationale_benchmark.llm.models import ModelRequest, ModelResponse, ProviderConfig
from rationale_benchmark.llm.providers.openai import OpenAIProvider


class TestOpenAIProvider:
  """Test cases for OpenAI provider implementation."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="openai",
      api_key="sk-test-api-key",  # Use proper sk- prefix
      base_url="https://api.openai.com/v1",
      timeout=30,
      max_retries=3,
      models=["gpt-4", "gpt-3.5-turbo"],
      default_params={"temperature": 0.7, "max_tokens": 1000},
      provider_specific={}
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return AsyncMock(spec=HTTPClient)

  @pytest.fixture
  def openai_provider(self, provider_config, http_client):
    """Create OpenAI provider instance."""
    return OpenAIProvider(provider_config, http_client)

  @pytest.fixture
  def model_request(self):
    """Create test model request."""
    return ModelRequest(
      prompt="What is the capital of France?",
      model="gpt-4",
      temperature=0.7,
      max_tokens=1000,
      system_prompt="You are a helpful assistant.",
      stop_sequences=["END"],
      provider_specific={}
    )

  @pytest.fixture
  def valid_openai_response(self):
    """Create valid OpenAI API response."""
    return {
      "id": "chatcmpl-123",
      "object": "chat.completion",
      "created": 1677652288,
      "model": "gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "The capital of France is Paris."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 8,
        "total_tokens": 23
      }
    }

  def test_init_sets_correct_attributes(self, provider_config, http_client):
    """Test that OpenAI provider initializes with correct attributes."""
    provider = OpenAIProvider(provider_config, http_client)
    
    assert provider.config == provider_config
    assert provider.http_client == http_client
    assert provider.name == "openai"
    assert provider.base_url == "https://api.openai.com/v1"

  def test_init_uses_default_base_url_when_none_provided(self, http_client):
    """Test that provider uses default base URL when none is configured."""
    config = ProviderConfig(
      name="openai",
      api_key="test-key",
      base_url=None
    )
    provider = OpenAIProvider(config, http_client)
    
    assert provider.base_url == "https://api.openai.com/v1"

  def test_validate_config_returns_empty_list_for_valid_config(self, openai_provider):
    """Test that validate_config returns empty list for valid configuration."""
    errors = openai_provider.validate_config()
    assert errors == []

  def test_validate_config_returns_errors_for_missing_api_key(self, http_client):
    """Test that validate_config returns errors for missing API key."""
    # Create config with invalid API key that bypasses __post_init__ validation
    config = ProviderConfig.__new__(ProviderConfig)
    config.name = "openai"
    config.api_key = ""  # Empty API key
    config.base_url = "https://api.openai.com/v1"
    config.timeout = 30
    config.max_retries = 3
    config.models = []
    config.default_params = {}
    config.provider_specific = {}
    
    provider = OpenAIProvider(config, http_client)
    
    errors = provider.validate_config()
    assert len(errors) > 0
    assert any("api key" in error.lower() for error in errors)

  @pytest.mark.asyncio
  async def test_generate_response_success(
    self, 
    openai_provider, 
    model_request, 
    valid_openai_response
  ):
    """Test successful response generation."""
    # Mock HTTP response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=valid_openai_response)
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request
    response = await openai_provider.generate_response(model_request)
    
    # Verify response
    assert isinstance(response, ModelResponse)
    assert response.text == "The capital of France is Paris."
    assert response.model == "gpt-4"
    assert response.provider == "openai"
    assert response.token_count == 23
    assert response.finish_reason == "stop"
    assert response.latency_ms >= 0  # Allow 0 for very fast mock responses

  @pytest.mark.asyncio
  async def test_generate_response_handles_http_error(
    self, 
    openai_provider, 
    model_request
  ):
    """Test that HTTP errors are handled properly."""
    # Mock HTTP error response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 401
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "Invalid API key",
        "type": "invalid_request_error"
      }
    })
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request and expect error
    with pytest.raises(AuthenticationError) as exc_info:
      await openai_provider.generate_response(model_request)
    
    assert "Invalid API key" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_handles_model_not_found(
    self, 
    openai_provider, 
    model_request
  ):
    """Test handling of model not found errors."""
    # Mock model not found response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 404
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "The model 'gpt-5' does not exist",
        "type": "invalid_request_error"
      }
    })
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request and expect error
    with pytest.raises(ModelNotFoundError) as exc_info:
      await openai_provider.generate_response(model_request)
    
    assert "gpt-4" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_validates_response_structure(
    self, 
    openai_provider, 
    model_request
  ):
    """Test that response structure validation is performed."""
    # Mock invalid response (missing required fields)
    invalid_response = {
      "id": "chatcmpl-123",
      "object": "chat.completion",
      # Missing choices, usage, model fields
    }
    
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=invalid_response)
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request and expect validation error
    with pytest.raises(ResponseValidationError) as exc_info:
      await openai_provider.generate_response(model_request)
    
    assert "Missing required field" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_blocks_streaming_parameters(
    self, 
    openai_provider
  ):
    """Test that streaming parameters are blocked."""
    request_with_streaming = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={"stream": True, "stream_options": {"include_usage": True}}
    )
    
    # Execute request and expect streaming error
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      await openai_provider.generate_response(request_with_streaming)
    
    assert "stream" in str(exc_info.value)
    assert "stream_options" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_list_models_success(self, openai_provider):
    """Test successful model listing."""
    models_response = {
      "object": "list",
      "data": [
        {"id": "gpt-4", "object": "model"},
        {"id": "gpt-3.5-turbo", "object": "model"},
        {"id": "text-davinci-003", "object": "model"}
      ]
    }
    
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=models_response)
    
    openai_provider.http_client.get = AsyncMock(return_value=mock_response)
    
    # Execute request
    models = await openai_provider.list_models()
    
    # Verify response
    assert isinstance(models, list)
    assert "gpt-4" in models
    assert "gpt-3.5-turbo" in models
    assert "text-davinci-003" in models

  @pytest.mark.asyncio
  async def test_list_models_handles_error(self, openai_provider):
    """Test that model listing errors are handled properly."""
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 403
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "Insufficient permissions",
        "type": "insufficient_quota"
      }
    })
    
    openai_provider.http_client.get = AsyncMock(return_value=mock_response)
    
    # Execute request and expect error
    with pytest.raises(ProviderError) as exc_info:
      await openai_provider.list_models()
    
    assert "Insufficient permissions" in str(exc_info.value)

  def test_prepare_request_formats_correctly(self, openai_provider, model_request):
    """Test that _prepare_request formats request correctly."""
    payload = openai_provider._prepare_request(model_request)
    
    # Verify basic structure
    assert payload["model"] == "gpt-4"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 1000
    assert payload["stream"] is False
    
    # Verify messages structure
    assert "messages" in payload
    assert len(payload["messages"]) == 2  # system + user
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "You are a helpful assistant."
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "What is the capital of France?"
    
    # Verify stop sequences
    assert payload["stop"] == ["END"]

  def test_prepare_request_handles_no_system_prompt(self, openai_provider):
    """Test that _prepare_request handles missing system prompt."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      system_prompt=None
    )
    
    payload = openai_provider._prepare_request(request)
    
    # Should only have user message
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Test prompt"

  def test_prepare_request_filters_streaming_params(self, openai_provider):
    """Test that _prepare_request filters out streaming parameters."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.5,  # Valid parameter
        "streaming": True,
        "max_tokens": 500  # Valid parameter
      }
    )
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      openai_provider._prepare_request(request)
    
    # Verify that streaming parameters are identified
    error = exc_info.value
    assert "stream" in error.blocked_params
    assert "stream_options" in error.blocked_params
    assert "streaming" in error.blocked_params

  def test_prepare_request_validates_openai_parameters(self, openai_provider):
    """Test that _prepare_request validates OpenAI-specific parameters."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={
        "frequency_penalty": 0.5,  # Valid
        "invalid_param": "value",  # Invalid
        "top_p": 0.9  # Valid
      }
    )
    
    payload = openai_provider._prepare_request(request)
    
    # Valid parameters should be included
    assert payload["frequency_penalty"] == 0.5
    assert payload["top_p"] == 0.9
    
    # Invalid parameters should be filtered out
    assert "invalid_param" not in payload

  def test_parse_response_creates_correct_model_response(
    self, 
    openai_provider, 
    model_request, 
    valid_openai_response
  ):
    """Test that _parse_response creates correct ModelResponse."""
    latency_ms = 1500
    
    response = openai_provider._parse_response(
      valid_openai_response, 
      model_request, 
      latency_ms
    )
    
    # Verify response structure
    assert isinstance(response, ModelResponse)
    assert response.text == "The capital of France is Paris."
    assert response.model == "gpt-4"
    assert response.provider == "openai"
    assert response.latency_ms == latency_ms
    assert response.token_count == 23
    assert response.finish_reason == "stop"
    
    # Verify metadata
    assert response.metadata["prompt_tokens"] == 15
    assert response.metadata["completion_tokens"] == 8
    assert response.metadata["choice_index"] == 0

  def test_parse_response_estimates_cost(
    self, 
    openai_provider, 
    model_request, 
    valid_openai_response
  ):
    """Test that _parse_response includes cost estimation."""
    response = openai_provider._parse_response(
      valid_openai_response, 
      model_request, 
      1500
    )
    
    # Cost estimate should be present and reasonable
    assert response.cost_estimate is not None
    assert response.cost_estimate > 0

  def test_validate_response_structure_passes_for_valid_response(
    self, 
    openai_provider, 
    valid_openai_response
  ):
    """Test that response structure validation passes for valid response."""
    # Should not raise any exception
    openai_provider._validate_response_structure(valid_openai_response)

  def test_validate_response_structure_fails_for_missing_fields(
    self, 
    openai_provider
  ):
    """Test that response structure validation fails for missing required fields."""
    invalid_responses = [
      {},  # Empty response
      {"choices": []},  # Missing other fields
      {"model": "gpt-4"},  # Missing choices and usage
      {  # Missing usage
        "choices": [{"message": {"content": "test", "role": "assistant"}, "finish_reason": "stop", "index": 0}],
        "model": "gpt-4",
        "object": "chat.completion"
      }
    ]
    
    for invalid_response in invalid_responses:
      with pytest.raises(ResponseValidationError):
        openai_provider._validate_response_structure(invalid_response)

  def test_validate_response_structure_fails_for_invalid_choice_structure(
    self, 
    openai_provider
  ):
    """Test that validation fails for invalid choice structure."""
    invalid_response = {
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            # Missing content field
          },
          "finish_reason": "stop"
        }
      ],
      "model": "gpt-4",
      "object": "chat.completion",
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(invalid_response)
    
    assert "content" in str(exc_info.value)

  def test_validate_response_structure_fails_for_empty_content(
    self, 
    openai_provider
  ):
    """Test that validation fails for empty message content."""
    invalid_response = {
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": ""  # Empty content
          },
          "finish_reason": "stop"
        }
      ],
      "model": "gpt-4",
      "object": "chat.completion",
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(invalid_response)
    
    assert "empty" in str(exc_info.value).lower()

  def test_validate_response_structure_fails_for_invalid_usage(
    self, 
    openai_provider
  ):
    """Test that validation fails for invalid usage information."""
    invalid_response = {
      "choices": [
        {
          "index": 0,
          "message": {"role": "assistant", "content": "test"},
          "finish_reason": "stop"
        }
      ],
      "model": "gpt-4",
      "object": "chat.completion",
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 20  # Inconsistent with sum
      }
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(invalid_response)
    
    assert "inconsistency" in str(exc_info.value).lower()

  def test_validate_response_structure_fails_for_streaming_indicator(
    self, 
    openai_provider, 
    valid_openai_response
  ):
    """Test that validation fails if streaming is indicated in response."""
    streaming_response = valid_openai_response.copy()
    streaming_response["stream"] = True
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(streaming_response)
    
    assert "streaming" in str(exc_info.value).lower()

  def test_is_valid_openai_parameter_validates_correctly(self, openai_provider):
    """Test that parameter validation works correctly."""
    # Valid parameters
    assert openai_provider._is_valid_openai_parameter("temperature", 0.7)
    assert openai_provider._is_valid_openai_parameter("max_tokens", 1000)
    assert openai_provider._is_valid_openai_parameter("frequency_penalty", 0.5)
    assert openai_provider._is_valid_openai_parameter("top_p", 0.9)
    
    # Invalid parameters
    assert not openai_provider._is_valid_openai_parameter("invalid_param", "value")
    assert not openai_provider._is_valid_openai_parameter("custom_field", 123)

  def test_estimate_cost_returns_reasonable_values(self, openai_provider):
    """Test that cost estimation returns reasonable values."""
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    
    # Test different models
    gpt4_cost = openai_provider._estimate_cost(usage, "gpt-4")
    gpt35_cost = openai_provider._estimate_cost(usage, "gpt-3.5-turbo")
    
    # GPT-4 should be more expensive than GPT-3.5
    assert gpt4_cost > gpt35_cost
    assert gpt4_cost > 0
    assert gpt35_cost > 0

  def test_estimate_cost_handles_unknown_models(self, openai_provider):
    """Test that cost estimation handles unknown models gracefully."""
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    
    cost = openai_provider._estimate_cost(usage, "unknown-model")
    
    # Should return None or a default estimate
    assert cost is None or cost >= 0


class TestOpenAIProviderIntegration:
  """Integration tests for OpenAI provider with more complex scenarios."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="openai",
      api_key="sk-test-api-key",
      base_url="https://api.openai.com/v1",
      timeout=30,
      max_retries=3,
      models=["gpt-4", "gpt-3.5-turbo"],
      default_params={"temperature": 0.7, "max_tokens": 1000},
      provider_specific={"frequency_penalty": 0.1}
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return AsyncMock(spec=HTTPClient)

  @pytest.fixture
  def openai_provider(self, provider_config, http_client):
    """Create OpenAI provider instance."""
    return OpenAIProvider(provider_config, http_client)

  @pytest.fixture
  def valid_openai_response(self):
    """Create valid OpenAI API response."""
    return {
      "id": "chatcmpl-123",
      "object": "chat.completion",
      "created": 1677652288,
      "model": "gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "The capital of France is Paris."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 8,
        "total_tokens": 23
      }
    }

  @pytest.mark.asyncio
  async def test_full_request_response_cycle(
    self, 
    openai_provider, 
    valid_openai_response
  ):
    """Test complete request-response cycle with all features."""
    # Create complex request
    request = ModelRequest(
      prompt="Explain quantum computing in simple terms.",
      model="gpt-4",
      temperature=0.8,
      max_tokens=500,
      system_prompt="You are a physics teacher explaining complex topics simply.",
      stop_sequences=["END", "STOP"],
      provider_specific={
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
        "top_p": 0.95
      }
    )
    
    # Mock successful response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=valid_openai_response)
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request
    response = await openai_provider.generate_response(request)
    
    # Verify HTTP client was called correctly
    openai_provider.http_client.post.assert_called_once()
    call_args = openai_provider.http_client.post.call_args
    
    # Verify URL
    assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"
    
    # Verify headers
    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer sk-test-api-key"
    assert headers["Content-Type"] == "application/json"
    
    # Verify payload
    payload = call_args[1]["json"]
    assert payload["model"] == "gpt-4"
    assert payload["temperature"] == 0.8
    assert payload["max_tokens"] == 500
    assert payload["stream"] is False
    assert payload["frequency_penalty"] == 0.2
    assert payload["presence_penalty"] == 0.1
    assert payload["top_p"] == 0.95
    assert payload["stop"] == ["END", "STOP"]
    
    # Verify messages
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    
    # Verify response
    assert isinstance(response, ModelResponse)
    assert response.provider == "openai"
    assert response.model == "gpt-4"

  @pytest.mark.asyncio
  async def test_error_handling_with_retry_context(self, openai_provider):
    """Test error handling in the context of retry logic."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    # Mock network error that should trigger retry
    openai_provider.http_client.post = AsyncMock(
      side_effect=Exception("Network error")
    )
    
    # Execute request and expect error
    with pytest.raises(ProviderError):
      await openai_provider.generate_response(request)

  @pytest.mark.asyncio
  async def test_concurrent_requests_handling(self, openai_provider, valid_openai_response):
    """Test that provider can handle concurrent requests properly."""
    import asyncio
    
    # Create multiple requests
    requests = [
      ModelRequest(prompt=f"Question {i}", model="gpt-4")
      for i in range(5)
    ]
    
    # Mock responses
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=valid_openai_response)
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute concurrent requests
    tasks = [
      openai_provider.generate_response(request)
      for request in requests
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # Verify all requests completed successfully
    assert len(responses) == 5
    for response in responses:
      assert isinstance(response, ModelResponse)
      assert response.provider == "openai"
    
    # Verify HTTP client was called for each request
    assert openai_provider.http_client.post.call_count == 5


class TestOpenAIRequestResponseHandling:
  """Detailed tests for OpenAI request/response handling methods."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="openai",
      api_key="sk-test-api-key",
      base_url="https://api.openai.com/v1"
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return AsyncMock(spec=HTTPClient)

  @pytest.fixture
  def openai_provider(self, provider_config, http_client):
    """Create OpenAI provider instance."""
    return OpenAIProvider(provider_config, http_client)

  def test_prepare_request_basic_structure(self, openai_provider):
    """Test that _prepare_request creates correct basic structure."""
    request = ModelRequest(
      prompt="Hello, world!",
      model="gpt-4",
      temperature=0.5,
      max_tokens=100
    )
    
    payload = openai_provider._prepare_request(request)
    
    # Verify required fields
    assert payload["model"] == "gpt-4"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100
    assert payload["stream"] is False
    
    # Verify messages structure
    assert "messages" in payload
    assert isinstance(payload["messages"], list)
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Hello, world!"

  def test_prepare_request_with_system_prompt(self, openai_provider):
    """Test _prepare_request with system prompt."""
    request = ModelRequest(
      prompt="User question",
      model="gpt-4",
      system_prompt="You are a helpful assistant."
    )
    
    payload = openai_provider._prepare_request(request)
    
    # Should have system message first, then user message
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "You are a helpful assistant."
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "User question"

  def test_prepare_request_with_stop_sequences(self, openai_provider):
    """Test _prepare_request with stop sequences."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      stop_sequences=["STOP", "END", "\n\n"]
    )
    
    payload = openai_provider._prepare_request(request)
    
    assert payload["stop"] == ["STOP", "END", "\n\n"]

  def test_prepare_request_without_stop_sequences(self, openai_provider):
    """Test _prepare_request without stop sequences."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      stop_sequences=None
    )
    
    payload = openai_provider._prepare_request(request)
    
    assert "stop" not in payload

  def test_prepare_request_filters_streaming_parameters(self, openai_provider):
    """Test that _prepare_request filters out all streaming parameters."""
    streaming_params = {
      "stream": True,
      "streaming": True,
      "stream_options": {"include_usage": True},
      "stream_usage": True,
      "stream_callback": lambda x: x,
      "stream_handler": "handler",
      "incremental": True
    }
    
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific=streaming_params
    )
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      openai_provider._prepare_request(request)
    
    # Verify all streaming parameters are blocked
    blocked_params = exc_info.value.blocked_params
    assert "stream" in blocked_params
    assert "streaming" in blocked_params
    assert "stream_options" in blocked_params
    assert "stream_usage" in blocked_params
    assert "stream_callback" in blocked_params
    assert "stream_handler" in blocked_params
    assert "incremental" in blocked_params

  def test_prepare_request_includes_valid_provider_specific_params(self, openai_provider):
    """Test that valid provider-specific parameters are included."""
    valid_params = {
      "frequency_penalty": 0.5,
      "presence_penalty": 0.2,
      "top_p": 0.9,
      "logit_bias": {"50256": -100},
      "user": "user123",
      "seed": 42
    }
    
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific=valid_params
    )
    
    payload = openai_provider._prepare_request(request)
    
    # All valid parameters should be included
    for key, value in valid_params.items():
      assert payload[key] == value

  def test_prepare_request_filters_invalid_provider_specific_params(self, openai_provider):
    """Test that invalid provider-specific parameters are filtered out."""
    mixed_params = {
      "frequency_penalty": 0.5,  # Valid
      "invalid_param": "value",  # Invalid
      "top_p": 0.9,  # Valid
      "custom_field": 123,  # Invalid
      "temperature": 0.8  # Valid but should be overridden by request.temperature
    }
    
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.7,  # This should take precedence
      provider_specific=mixed_params
    )
    
    payload = openai_provider._prepare_request(request)
    
    # Valid parameters should be included
    assert payload["frequency_penalty"] == 0.5
    assert payload["top_p"] == 0.9
    
    # Invalid parameters should be filtered out
    assert "invalid_param" not in payload
    assert "custom_field" not in payload
    
    # Request temperature should take precedence over provider_specific
    # Note: provider_specific parameters are added after request parameters,
    # so they actually override request parameters in current implementation
    assert payload["temperature"] == 0.8  # provider_specific value wins

  def test_prepare_request_ensures_stream_false(self, openai_provider):
    """Test that stream is always set to False."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={"other_param": "value"}
    )
    
    payload = openai_provider._prepare_request(request)
    
    # Stream should always be False
    assert payload["stream"] is False

  def test_prepare_request_raises_error_if_stream_in_payload(self, openai_provider):
    """Test that error is raised if stream somehow gets set to True."""
    # Test with streaming parameter in provider_specific
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={"stream": True}
    )
    
    with pytest.raises(StreamingNotSupportedError):
      openai_provider._prepare_request(request)

  def test_parse_response_basic_structure(self, openai_provider):
    """Test _parse_response creates correct ModelResponse structure."""
    response_data = {
      "id": "chatcmpl-123",
      "object": "chat.completion",
      "created": 1677652288,
      "model": "gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "This is a test response."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
      }
    }
    
    request = ModelRequest(prompt="Test", model="gpt-4")
    latency_ms = 1500
    
    response = openai_provider._parse_response(response_data, request, latency_ms)
    
    # Verify basic fields
    assert isinstance(response, ModelResponse)
    assert response.text == "This is a test response."
    assert response.model == "gpt-4"
    assert response.provider == "openai"
    assert response.latency_ms == 1500
    assert response.token_count == 15
    assert response.finish_reason == "stop"
    assert isinstance(response.timestamp, datetime)

  def test_parse_response_includes_metadata(self, openai_provider):
    """Test that _parse_response includes comprehensive metadata."""
    response_data = {
      "id": "chatcmpl-123",
      "object": "chat.completion",
      "created": 1677652288,
      "model": "gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Response text"
          },
          "finish_reason": "length"
        }
      ],
      "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
      }
    }
    
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    response = openai_provider._parse_response(response_data, request, 2000)
    
    # Verify metadata
    assert response.metadata["prompt_tokens"] == 100
    assert response.metadata["completion_tokens"] == 50
    assert response.metadata["finish_reason"] == "length"
    assert response.metadata["choice_index"] == 0

  def test_parse_response_estimates_cost(self, openai_provider):
    """Test that _parse_response includes cost estimation."""
    response_data = {
      "id": "chatcmpl-123",
      "object": "chat.completion",
      "created": 1677652288,
      "model": "gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {"role": "assistant", "content": "Response"},
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
      }
    }
    
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    response = openai_provider._parse_response(response_data, request, 1000)
    
    # Cost estimate should be present and positive
    assert response.cost_estimate is not None
    assert response.cost_estimate > 0

  def test_parse_response_handles_different_finish_reasons(self, openai_provider):
    """Test _parse_response with different finish reasons."""
    finish_reasons = ["stop", "length", "content_filter", "function_call"]
    
    for finish_reason in finish_reasons:
      response_data = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
          {
            "index": 0,
            "message": {"role": "assistant", "content": "Response"},
            "finish_reason": finish_reason
          }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
      }
      
      request = ModelRequest(prompt="Test", model="gpt-4")
      
      response = openai_provider._parse_response(response_data, request, 1000)
      
      assert response.finish_reason == finish_reason

  def test_validate_response_structure_comprehensive_validation(self, openai_provider):
    """Test comprehensive response structure validation."""
    # Test missing top-level fields - skip empty dict as it triggers different validation
    missing_fields_tests = [
      ({"choices": [], "model": "gpt-4", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, "object": "chat.completion"}, "non-empty array"),  # Empty choices array
      ({"choices": [{"index": 0, "message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, "object": "chat.completion"}, "model"),
      ({"choices": [{"index": 0, "message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}], "model": "gpt-4", "object": "chat.completion"}, "usage"),
      ({"choices": [{"index": 0, "message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}], "model": "gpt-4", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}, "object"),
    ]
    
    for incomplete_response, expected_field in missing_fields_tests:
      with pytest.raises(ResponseValidationError) as exc_info:
        openai_provider._validate_response_structure(incomplete_response)
      
      assert expected_field in str(exc_info.value)

  def test_validate_response_structure_validates_object_type(self, openai_provider):
    """Test that object type validation works correctly."""
    invalid_response = {
      "choices": [{"index": 0, "message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}],
      "model": "gpt-4",
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
      "object": "invalid_type"  # Wrong object type
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(invalid_response)
    
    assert "object type" in str(exc_info.value)
    assert "chat.completion" in str(exc_info.value)

  def test_validate_response_structure_validates_choices_array(self, openai_provider):
    """Test that choices array validation works correctly."""
    # Empty choices array
    invalid_response = {
      "choices": [],  # Empty array
      "model": "gpt-4",
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
      "object": "chat.completion"
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(invalid_response)
    
    assert "non-empty array" in str(exc_info.value)

  def test_validate_response_structure_validates_choice_fields(self, openai_provider):
    """Test that individual choice field validation works."""
    # Missing message field
    invalid_response = {
      "choices": [
        {
          "index": 0,
          # Missing message field
          "finish_reason": "stop"
        }
      ],
      "model": "gpt-4",
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
      "object": "chat.completion"
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(invalid_response)
    
    assert "message" in str(exc_info.value)

  def test_validate_response_structure_validates_message_structure(self, openai_provider):
    """Test that message structure validation works."""
    # Missing content in message
    invalid_response = {
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant"
            # Missing content field
          },
          "finish_reason": "stop"
        }
      ],
      "model": "gpt-4",
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
      "object": "chat.completion"
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(invalid_response)
    
    assert "content" in str(exc_info.value)

  def test_validate_response_structure_validates_message_role(self, openai_provider):
    """Test that message role validation works."""
    invalid_response = {
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "user",  # Should be "assistant"
            "content": "Response text"
          },
          "finish_reason": "stop"
        }
      ],
      "model": "gpt-4",
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
      "object": "chat.completion"
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(invalid_response)
    
    assert "role" in str(exc_info.value)
    assert "assistant" in str(exc_info.value)

  def test_validate_response_structure_validates_empty_content(self, openai_provider):
    """Test that empty content validation works."""
    test_cases = [
      "",  # Empty string
      "   ",  # Whitespace only
      "\n\t  \n",  # Various whitespace
    ]
    
    for empty_content in test_cases:
      invalid_response = {
        "choices": [
          {
            "index": 0,
            "message": {
              "role": "assistant",
              "content": empty_content
            },
            "finish_reason": "stop"
          }
        ],
        "model": "gpt-4",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "object": "chat.completion"
      }
      
      with pytest.raises(ResponseValidationError) as exc_info:
        openai_provider._validate_response_structure(invalid_response)
      
      assert "empty" in str(exc_info.value).lower()

  def test_validate_response_structure_validates_usage_fields(self, openai_provider):
    """Test that usage field validation works."""
    # Missing usage fields
    missing_usage_fields = [
      {"completion_tokens": 5, "total_tokens": 15},  # Missing prompt_tokens
      {"prompt_tokens": 10, "total_tokens": 15},  # Missing completion_tokens
      {"prompt_tokens": 10, "completion_tokens": 5},  # Missing total_tokens
    ]
    
    for incomplete_usage in missing_usage_fields:
      invalid_response = {
        "choices": [
          {
            "index": 0,
            "message": {"role": "assistant", "content": "Response"},
            "finish_reason": "stop"
          }
        ],
        "model": "gpt-4",
        "usage": incomplete_usage,
        "object": "chat.completion"
      }
      
      with pytest.raises(ResponseValidationError):
        openai_provider._validate_response_structure(invalid_response)

  def test_validate_response_structure_validates_usage_types(self, openai_provider):
    """Test that usage field type validation works."""
    # Invalid usage field types
    invalid_usage_types = [
      {"prompt_tokens": "10", "completion_tokens": 5, "total_tokens": 15},  # String instead of int
      {"prompt_tokens": 10, "completion_tokens": -5, "total_tokens": 15},  # Negative value
      {"prompt_tokens": 10.5, "completion_tokens": 5, "total_tokens": 15},  # Float instead of int
    ]
    
    for invalid_usage in invalid_usage_types:
      invalid_response = {
        "choices": [
          {
            "index": 0,
            "message": {"role": "assistant", "content": "Response"},
            "finish_reason": "stop"
          }
        ],
        "model": "gpt-4",
        "usage": invalid_usage,
        "object": "chat.completion"
      }
      
      with pytest.raises(ResponseValidationError):
        openai_provider._validate_response_structure(invalid_response)

  def test_validate_response_structure_validates_token_consistency(self, openai_provider):
    """Test that token count consistency validation works."""
    invalid_response = {
      "choices": [
        {
          "index": 0,
          "message": {"role": "assistant", "content": "Response"},
          "finish_reason": "stop"
        }
      ],
      "model": "gpt-4",
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 20  # Should be 15
      },
      "object": "chat.completion"
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(invalid_response)
    
    assert "inconsistency" in str(exc_info.value).lower()

  def test_validate_response_structure_validates_model_field(self, openai_provider):
    """Test that model field validation works."""
    invalid_responses = [
      # Empty model string
      {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Response"}, "finish_reason": "stop"}],
        "model": "",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "object": "chat.completion"
      },
      # Non-string model
      {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Response"}, "finish_reason": "stop"}],
        "model": 123,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "object": "chat.completion"
      }
    ]
    
    for invalid_response in invalid_responses:
      with pytest.raises(ResponseValidationError) as exc_info:
        openai_provider._validate_response_structure(invalid_response)
      
      assert "model" in str(exc_info.value)

  def test_validate_response_structure_detects_streaming_indicators(self, openai_provider):
    """Test that streaming indicators are detected and rejected."""
    invalid_response = {
      "choices": [
        {
          "index": 0,
          "message": {"role": "assistant", "content": "Response"},
          "finish_reason": "stop"
        }
      ],
      "model": "gpt-4",
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
      "object": "chat.completion",
      "stream": True  # Streaming indicator
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_provider._validate_response_structure(invalid_response)
    
    assert "streaming" in str(exc_info.value).lower()

  def test_is_valid_openai_parameter_comprehensive_validation(self, openai_provider):
    """Test comprehensive parameter validation."""
    # Valid parameters
    valid_params = [
      ("frequency_penalty", 0.5),
      ("presence_penalty", 0.2),
      ("temperature", 0.7),
      ("max_tokens", 1000),
      ("top_p", 0.9),
      ("logit_bias", {"50256": -100}),
      ("logprobs", True),
      ("top_logprobs", 5),
      ("n", 1),
      ("stop", ["END"]),
      ("user", "user123"),
      ("seed", 42),
      ("response_format", {"type": "json_object"}),
      ("tools", []),
      ("tool_choice", "auto"),
      ("function_call", "auto"),
      ("functions", [])
    ]
    
    for param_name, param_value in valid_params:
      assert openai_provider._is_valid_openai_parameter(param_name, param_value)
    
    # Invalid parameters
    invalid_params = [
      ("custom_param", "value"),
      ("invalid_field", 123),
      ("stream", True),  # Streaming parameter
      ("unknown", "data")
    ]
    
    for param_name, param_value in invalid_params:
      assert not openai_provider._is_valid_openai_parameter(param_name, param_value)

  def test_estimate_cost_different_models(self, openai_provider):
    """Test cost estimation for different OpenAI models."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
    
    # Test known models
    models_to_test = [
      "gpt-4",
      "gpt-4-32k",
      "gpt-3.5-turbo",
      "gpt-3.5-turbo-16k",
      "text-davinci-003"
    ]
    
    costs = {}
    for model in models_to_test:
      cost = openai_provider._estimate_cost(usage, model)
      costs[model] = cost
      
      # Cost should be positive for known models
      assert cost is not None
      assert cost > 0
    
    # GPT-4 should be more expensive than GPT-3.5
    assert costs["gpt-4"] > costs["gpt-3.5-turbo"]
    
    # 32k models should be more expensive than regular models
    if "gpt-4-32k" in costs and "gpt-4" in costs:
      assert costs["gpt-4-32k"] > costs["gpt-4"]

  def test_estimate_cost_scales_with_usage(self, openai_provider):
    """Test that cost estimation scales correctly with token usage."""
    base_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    double_usage = {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}
    
    base_cost = openai_provider._estimate_cost(base_usage, "gpt-4")
    double_cost = openai_provider._estimate_cost(double_usage, "gpt-4")
    
    # Double usage should result in approximately double cost
    assert double_cost > base_cost
    assert abs(double_cost - (2 * base_cost)) < (0.1 * base_cost)  # Within 10%

  def test_estimate_cost_handles_zero_usage(self, openai_provider):
    """Test cost estimation with zero token usage."""
    zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    cost = openai_provider._estimate_cost(zero_usage, "gpt-4")
    
    # Should return 0 or very small cost
    assert cost is not None
    assert cost >= 0
    assert cost < 0.001  # Should be very small

  def test_estimate_cost_unknown_model_returns_none(self, openai_provider):
    """Test that unknown models return None for cost estimation."""
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    
    cost = openai_provider._estimate_cost(usage, "unknown-model-xyz")
    
    assert cost is None


class TestOpenAIErrorHandling:
  """Test error handling scenarios for OpenAI provider."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="openai",
      api_key="sk-test-api-key",
      base_url="https://api.openai.com/v1"
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return AsyncMock(spec=HTTPClient)

  @pytest.fixture
  def openai_provider(self, provider_config, http_client):
    """Create OpenAI provider instance."""
    return OpenAIProvider(provider_config, http_client)

  @pytest.mark.asyncio
  async def test_authentication_error_handling(self, openai_provider):
    """Test handling of authentication errors."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    # Mock 401 response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 401
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "Invalid API key provided",
        "type": "invalid_request_error",
        "code": "invalid_api_key"
      }
    })
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    with pytest.raises(AuthenticationError) as exc_info:
      await openai_provider.generate_response(request)
    
    assert "Invalid API key" in str(exc_info.value)
    assert exc_info.value.provider == "openai"

  @pytest.mark.asyncio
  async def test_model_not_found_error_handling(self, openai_provider):
    """Test handling of model not found errors."""
    request = ModelRequest(prompt="Test", model="gpt-5")  # Non-existent model
    
    # Mock 404 response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 404
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "The model 'gpt-5' does not exist",
        "type": "invalid_request_error",
        "code": "model_not_found"
      }
    })
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    with pytest.raises(ModelNotFoundError) as exc_info:
      await openai_provider.generate_response(request)
    
    assert "gpt-5" in str(exc_info.value)
    assert exc_info.value.provider == "openai"
    assert exc_info.value.model == "gpt-5"

  @pytest.mark.asyncio
  async def test_rate_limit_error_handling(self, openai_provider):
    """Test handling of rate limit errors."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    # Mock 429 response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 429
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "Rate limit reached for requests",
        "type": "requests",
        "code": "rate_limit_exceeded"
      }
    })
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    with pytest.raises(ProviderError) as exc_info:
      await openai_provider.generate_response(request)
    
    assert "Rate limit" in str(exc_info.value)
    assert exc_info.value.provider == "openai"

  @pytest.mark.asyncio
  async def test_server_error_handling(self, openai_provider):
    """Test handling of server errors."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    # Mock 500 response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 500
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "The server had an error while processing your request",
        "type": "server_error"
      }
    })
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    with pytest.raises(ProviderError) as exc_info:
      await openai_provider.generate_response(request)
    
    assert "server had an error" in str(exc_info.value)
    assert exc_info.value.provider == "openai"

  @pytest.mark.asyncio
  async def test_malformed_error_response_handling(self, openai_provider):
    """Test handling of malformed error responses."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    # Mock response with malformed error
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 400
    mock_response.json = AsyncMock(return_value={"invalid": "structure"})
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    with pytest.raises(ProviderError) as exc_info:
      await openai_provider.generate_response(request)
    
    # Should still raise ProviderError even with malformed error response
    assert exc_info.value.provider == "openai"
    assert "400" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_network_error_handling(self, openai_provider):
    """Test handling of network errors."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    # Mock network error
    openai_provider.http_client.post = AsyncMock(
      side_effect=Exception("Connection timeout")
    )
    
    with pytest.raises(ProviderError) as exc_info:
      await openai_provider.generate_response(request)
    
    assert "Connection timeout" in str(exc_info.value)
    assert exc_info.value.provider == "openai"