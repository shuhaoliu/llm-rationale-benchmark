"""Unit tests for Anthropic provider implementation."""

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
from rationale_benchmark.llm.providers.anthropic import AnthropicProvider


class TestAnthropicProvider:
  """Test cases for Anthropic provider implementation."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="anthropic",
      api_key="sk-ant-api03-test-key",  # Use proper Anthropic key format
      base_url="https://api.anthropic.com",
      timeout=30,
      max_retries=3,
      models=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
      default_params={"temperature": 0.7, "max_tokens": 1000},
      provider_specific={}
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return AsyncMock(spec=HTTPClient)

  @pytest.fixture
  def anthropic_provider(self, provider_config, http_client):
    """Create Anthropic provider instance."""
    return AnthropicProvider(provider_config, http_client)

  @pytest.fixture
  def model_request(self):
    """Create test model request."""
    return ModelRequest(
      prompt="What is the capital of France?",
      model="claude-3-opus-20240229",
      temperature=0.7,
      max_tokens=1000,
      system_prompt="You are a helpful assistant.",
      stop_sequences=["END"],
      provider_specific={}
    )

  @pytest.fixture
  def valid_anthropic_response(self):
    """Create valid Anthropic API response."""
    return {
      "id": "msg_01EhbhqnW8Jk9C7j2Y3xQ4Z5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "The capital of France is Paris."
        }
      ],
      "model": "claude-3-opus-20240229",
      "stop_reason": "end_turn",
      "stop_sequence": None,
      "usage": {
        "input_tokens": 15,
        "output_tokens": 8
      }
    }

  def test_init_sets_correct_attributes(self, provider_config, http_client):
    """Test that Anthropic provider initializes with correct attributes."""
    provider = AnthropicProvider(provider_config, http_client)
    
    assert provider.config == provider_config
    assert provider.http_client == http_client
    assert provider.name == "anthropic"
    assert provider.base_url == "https://api.anthropic.com"

  def test_init_uses_default_base_url_when_none_provided(self, http_client):
    """Test that provider uses default base URL when none is configured."""
    config = ProviderConfig(
      name="anthropic",
      api_key="sk-ant-api03-test-key",
      base_url=None
    )
    provider = AnthropicProvider(config, http_client)
    
    assert provider.base_url == "https://api.anthropic.com"

  def test_validate_config_returns_empty_list_for_valid_config(self, anthropic_provider):
    """Test that validate_config returns empty list for valid configuration."""
    errors = anthropic_provider.validate_config()
    assert errors == []

  def test_validate_config_returns_errors_for_missing_api_key(self, http_client):
    """Test that validate_config returns errors for missing API key."""
    # Create config with invalid API key that bypasses __post_init__ validation
    config = ProviderConfig.__new__(ProviderConfig)
    config.name = "anthropic"
    config.api_key = ""  # Empty API key
    config.base_url = "https://api.anthropic.com"
    config.timeout = 30
    config.max_retries = 3
    config.models = []
    config.default_params = {}
    config.provider_specific = {}
    
    provider = AnthropicProvider(config, http_client)
    
    errors = provider.validate_config()
    assert len(errors) > 0
    assert any("api key" in error.lower() for error in errors)

  def test_validate_config_returns_errors_for_invalid_api_key_format(self, http_client):
    """Test that validate_config returns errors for invalid API key format."""
    config = ProviderConfig.__new__(ProviderConfig)
    config.name = "anthropic"
    config.api_key = "invalid-key-format"  # Invalid format
    config.base_url = "https://api.anthropic.com"
    config.timeout = 30
    config.max_retries = 3
    config.models = []
    config.default_params = {}
    config.provider_specific = {}
    
    provider = AnthropicProvider(config, http_client)
    
    errors = provider.validate_config()
    assert len(errors) > 0
    assert any("api key format" in error.lower() for error in errors)

  @pytest.mark.asyncio
  async def test_generate_response_success(
    self, 
    anthropic_provider, 
    model_request, 
    valid_anthropic_response
  ):
    """Test successful response generation."""
    # Mock HTTP response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=valid_anthropic_response)
    
    anthropic_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request
    response = await anthropic_provider.generate_response(model_request)
    
    # Verify response
    assert isinstance(response, ModelResponse)
    assert response.text == "The capital of France is Paris."
    assert response.model == "claude-3-opus-20240229"
    assert response.provider == "anthropic"
    assert response.token_count == 23  # input_tokens + output_tokens
    assert response.finish_reason == "end_turn"
    assert response.latency_ms >= 0  # Allow 0 for very fast mock responses

  @pytest.mark.asyncio
  async def test_generate_response_handles_http_error(
    self, 
    anthropic_provider, 
    model_request
  ):
    """Test that HTTP errors are handled properly."""
    # Mock HTTP error response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 401
    mock_response.json = AsyncMock(return_value={
      "type": "error",
      "error": {
        "type": "authentication_error",
        "message": "Invalid API key"
      }
    })
    
    anthropic_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request and expect error
    with pytest.raises(AuthenticationError) as exc_info:
      await anthropic_provider.generate_response(model_request)
    
    assert "Invalid API key" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_handles_model_not_found(
    self, 
    anthropic_provider, 
    model_request
  ):
    """Test handling of model not found errors."""
    # Mock model not found response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 400
    mock_response.json = AsyncMock(return_value={
      "type": "error",
      "error": {
        "type": "invalid_request_error",
        "message": "The model 'claude-invalid' does not exist"
      }
    })
    
    anthropic_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request and expect error
    with pytest.raises(ModelNotFoundError) as exc_info:
      await anthropic_provider.generate_response(model_request)
    
    assert "claude-3-opus-20240229" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_validates_response_structure(
    self, 
    anthropic_provider, 
    model_request
  ):
    """Test that response structure validation is performed."""
    # Mock invalid response (missing required fields)
    invalid_response = {
      "id": "msg_123",
      "type": "message",
      # Missing content, role, model, usage fields
    }
    
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=invalid_response)
    
    anthropic_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request and expect validation error
    with pytest.raises(ResponseValidationError) as exc_info:
      await anthropic_provider.generate_response(model_request)
    
    assert "Missing required field" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_blocks_streaming_parameters(
    self, 
    anthropic_provider
  ):
    """Test that streaming parameters are blocked."""
    request_with_streaming = ModelRequest(
      prompt="Test prompt",
      model="claude-3-opus-20240229",
      provider_specific={"stream": True, "stream_options": {"include_usage": True}}
    )
    
    # Execute request and expect streaming error
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      await anthropic_provider.generate_response(request_with_streaming)
    
    assert "stream" in str(exc_info.value)
    assert "stream_options" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_list_models_success(self, anthropic_provider):
    """Test successful model listing."""
    # Anthropic doesn't have a models endpoint, so this returns configured models
    models = await anthropic_provider.list_models()
    
    # Verify response
    assert isinstance(models, list)
    assert "claude-3-opus-20240229" in models
    assert "claude-3-sonnet-20240229" in models
    assert "claude-3-haiku-20240307" in models

  def test_prepare_request_formats_correctly(self, anthropic_provider, model_request):
    """Test that _prepare_request formats request correctly."""
    payload = anthropic_provider._prepare_request(model_request)
    
    # Verify basic structure
    assert payload["model"] == "claude-3-opus-20240229"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 1000
    
    # Verify messages structure
    assert "messages" in payload
    assert len(payload["messages"]) == 1  # Only user message
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "What is the capital of France?"
    
    # Verify system prompt is separate
    assert payload["system"] == "You are a helpful assistant."
    
    # Verify stop sequences
    assert payload["stop_sequences"] == ["END"]

  def test_prepare_request_handles_no_system_prompt(self, anthropic_provider):
    """Test that _prepare_request handles missing system prompt."""
    request = ModelRequest(
      prompt="Test prompt",
      model="claude-3-opus-20240229",
      system_prompt=None
    )
    
    payload = anthropic_provider._prepare_request(request)
    
    # Should only have user message and no system field
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Test prompt"
    assert "system" not in payload

  def test_prepare_request_filters_streaming_params(self, anthropic_provider):
    """Test that _prepare_request filters out streaming parameters."""
    request = ModelRequest(
      prompt="Test prompt",
      model="claude-3-opus-20240229",
      provider_specific={
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.5,  # Valid parameter
        "streaming": True,
        "max_tokens": 500  # Valid parameter
      }
    )
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      anthropic_provider._prepare_request(request)
    
    # Verify that streaming parameters are identified
    error = exc_info.value
    assert "stream" in error.blocked_params
    assert "stream_options" in error.blocked_params
    assert "streaming" in error.blocked_params

  def test_prepare_request_validates_anthropic_parameters(self, anthropic_provider):
    """Test that _prepare_request validates Anthropic-specific parameters."""
    request = ModelRequest(
      prompt="Test prompt",
      model="claude-3-opus-20240229",
      provider_specific={
        "top_p": 0.9,  # Valid
        "invalid_param": "value",  # Invalid
        "top_k": 40  # Valid
      }
    )
    
    payload = anthropic_provider._prepare_request(request)
    
    # Valid parameters should be included
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 40
    
    # Invalid parameters should be filtered out
    assert "invalid_param" not in payload

  def test_parse_response_creates_correct_model_response(
    self, 
    anthropic_provider, 
    model_request, 
    valid_anthropic_response
  ):
    """Test that _parse_response creates correct ModelResponse."""
    latency_ms = 1500
    
    response = anthropic_provider._parse_response(
      valid_anthropic_response, 
      model_request, 
      latency_ms
    )
    
    # Verify response structure
    assert isinstance(response, ModelResponse)
    assert response.text == "The capital of France is Paris."
    assert response.model == "claude-3-opus-20240229"
    assert response.provider == "anthropic"
    assert response.latency_ms == latency_ms
    assert response.token_count == 23  # input_tokens + output_tokens
    assert response.finish_reason == "end_turn"
    
    # Verify metadata
    assert response.metadata["input_tokens"] == 15
    assert response.metadata["output_tokens"] == 8
    assert response.metadata["stop_reason"] == "end_turn"

  def test_parse_response_estimates_cost(
    self, 
    anthropic_provider, 
    model_request, 
    valid_anthropic_response
  ):
    """Test that _parse_response includes cost estimation."""
    response = anthropic_provider._parse_response(
      valid_anthropic_response, 
      model_request, 
      1500
    )
    
    # Cost estimate should be present and reasonable
    assert response.cost_estimate is not None
    assert response.cost_estimate > 0

  def test_validate_response_structure_passes_for_valid_response(
    self, 
    anthropic_provider, 
    valid_anthropic_response
  ):
    """Test that response structure validation passes for valid response."""
    # Should not raise any exception
    anthropic_provider._validate_response_structure(valid_anthropic_response)

  def test_validate_response_structure_fails_for_missing_fields(
    self, 
    anthropic_provider
  ):
    """Test that response structure validation fails for missing required fields."""
    invalid_responses = [
      {},  # Empty response
      {"content": []},  # Missing other fields
      {"model": "claude-3-opus-20240229"},  # Missing content and usage
      {  # Missing usage
        "content": [{"type": "text", "text": "test"}],
        "model": "claude-3-opus-20240229",
        "role": "assistant",
        "stop_reason": "end_turn"
      }
    ]
    
    for invalid_response in invalid_responses:
      with pytest.raises(ResponseValidationError):
        anthropic_provider._validate_response_structure(invalid_response)

  def test_validate_response_structure_fails_for_invalid_content_structure(
    self, 
    anthropic_provider
  ):
    """Test that validation fails for invalid content structure."""
    invalid_response = {
      "content": [
        {
          "type": "text",
          # Missing text field
        }
      ],
      "model": "claude-3-opus-20240229",
      "role": "assistant",
      "stop_reason": "end_turn",
      "usage": {"input_tokens": 10, "output_tokens": 5}
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      anthropic_provider._validate_response_structure(invalid_response)
    
    assert "text" in str(exc_info.value)

  def test_validate_response_structure_fails_for_empty_content(
    self, 
    anthropic_provider
  ):
    """Test that validation fails for empty message content."""
    invalid_response = {
      "content": [
        {
          "type": "text",
          "text": ""  # Empty content
        }
      ],
      "model": "claude-3-opus-20240229",
      "role": "assistant",
      "stop_reason": "end_turn",
      "usage": {"input_tokens": 10, "output_tokens": 5}
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      anthropic_provider._validate_response_structure(invalid_response)
    
    assert "empty" in str(exc_info.value).lower()

  def test_validate_response_structure_fails_for_invalid_usage(
    self, 
    anthropic_provider
  ):
    """Test that validation fails for invalid usage information."""
    invalid_response = {
      "content": [{"type": "text", "text": "test"}],
      "model": "claude-3-opus-20240229",
      "role": "assistant",
      "stop_reason": "end_turn",
      "usage": {
        "input_tokens": -1,  # Invalid negative value
        "output_tokens": 5
      }
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      anthropic_provider._validate_response_structure(invalid_response)
    
    assert "negative" in str(exc_info.value).lower()

  def test_is_valid_anthropic_parameter_validates_correctly(self, anthropic_provider):
    """Test that parameter validation works correctly."""
    # Valid parameters
    assert anthropic_provider._is_valid_anthropic_parameter("temperature", 0.7)
    assert anthropic_provider._is_valid_anthropic_parameter("max_tokens", 1000)
    assert anthropic_provider._is_valid_anthropic_parameter("top_p", 0.9)
    assert anthropic_provider._is_valid_anthropic_parameter("top_k", 40)
    
    # Invalid parameters
    assert not anthropic_provider._is_valid_anthropic_parameter("invalid_param", "value")
    assert not anthropic_provider._is_valid_anthropic_parameter("custom_field", 123)

  def test_estimate_cost_returns_reasonable_values(self, anthropic_provider):
    """Test that cost estimation returns reasonable values."""
    usage = {"input_tokens": 100, "output_tokens": 50}
    
    # Test different models
    opus_cost = anthropic_provider._estimate_cost(usage, "claude-3-opus-20240229")
    sonnet_cost = anthropic_provider._estimate_cost(usage, "claude-3-sonnet-20240229")
    haiku_cost = anthropic_provider._estimate_cost(usage, "claude-3-haiku-20240307")
    
    # Opus should be most expensive, Haiku least expensive
    assert opus_cost > sonnet_cost > haiku_cost
    assert opus_cost > 0
    assert sonnet_cost > 0
    assert haiku_cost > 0

  def test_estimate_cost_handles_unknown_models(self, anthropic_provider):
    """Test that cost estimation handles unknown models gracefully."""
    usage = {"input_tokens": 100, "output_tokens": 50}
    
    cost = anthropic_provider._estimate_cost(usage, "unknown-model")
    
    # Should return None or a default estimate
    assert cost is None or cost >= 0


class TestAnthropicProviderIntegration:
  """Integration tests for Anthropic provider with more complex scenarios."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="anthropic",
      api_key="sk-ant-api03-test-key",
      base_url="https://api.anthropic.com",
      timeout=30,
      max_retries=3,
      models=["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
      default_params={"temperature": 0.7, "max_tokens": 1000},
      provider_specific={"top_p": 0.9}
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return AsyncMock(spec=HTTPClient)

  @pytest.fixture
  def anthropic_provider(self, provider_config, http_client):
    """Create Anthropic provider instance."""
    return AnthropicProvider(provider_config, http_client)

  @pytest.fixture
  def valid_anthropic_response(self):
    """Create valid Anthropic API response."""
    return {
      "id": "msg_01EhbhqnW8Jk9C7j2Y3xQ4Z5",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "The capital of France is Paris."
        }
      ],
      "model": "claude-3-opus-20240229",
      "stop_reason": "end_turn",
      "stop_sequence": None,
      "usage": {
        "input_tokens": 15,
        "output_tokens": 8
      }
    }

  @pytest.mark.asyncio
  async def test_full_request_response_cycle(
    self, 
    anthropic_provider, 
    valid_anthropic_response
  ):
    """Test complete request-response cycle with all features."""
    # Create complex request
    request = ModelRequest(
      prompt="Explain quantum computing in simple terms.",
      model="claude-3-opus-20240229",
      temperature=0.8,
      max_tokens=500,
      system_prompt="You are a physics teacher explaining complex topics simply.",
      stop_sequences=["END", "STOP"],
      provider_specific={
        "top_p": 0.95,
        "top_k": 40
      }
    )
    
    # Mock successful response
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=valid_anthropic_response)
    
    anthropic_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute request
    response = await anthropic_provider.generate_response(request)
    
    # Verify HTTP client was called correctly
    anthropic_provider.http_client.post.assert_called_once()
    call_args = anthropic_provider.http_client.post.call_args
    
    # Verify URL
    assert call_args[0][0] == "https://api.anthropic.com/v1/messages"
    
    # Verify headers
    headers = call_args[1]["headers"]
    assert headers["x-api-key"] == "sk-ant-api03-test-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["anthropic-version"] == "2023-06-01"
    
    # Verify payload
    payload = call_args[1]["json"]
    assert payload["model"] == "claude-3-opus-20240229"
    assert payload["temperature"] == 0.8
    assert payload["max_tokens"] == 500
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 40
    assert payload["stop_sequences"] == ["END", "STOP"]
    assert payload["system"] == "You are a physics teacher explaining complex topics simply."
    
    # Verify messages
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Explain quantum computing in simple terms."
    
    # Verify response
    assert isinstance(response, ModelResponse)
    assert response.provider == "anthropic"
    assert response.model == "claude-3-opus-20240229"

  @pytest.mark.asyncio
  async def test_error_handling_with_retry_context(self, anthropic_provider):
    """Test error handling in the context of retry logic."""
    request = ModelRequest(prompt="Test", model="claude-3-opus-20240229")
    
    # Mock network error that should trigger retry
    anthropic_provider.http_client.post = AsyncMock(
      side_effect=Exception("Network error")
    )
    
    # Execute request and expect error
    with pytest.raises(ProviderError):
      await anthropic_provider.generate_response(request)

  @pytest.mark.asyncio
  async def test_concurrent_requests_handling(self, anthropic_provider, valid_anthropic_response):
    """Test that provider can handle concurrent requests properly."""
    import asyncio
    
    # Create multiple requests
    requests = [
      ModelRequest(prompt=f"Question {i}", model="claude-3-opus-20240229")
      for i in range(5)
    ]
    
    # Mock responses
    mock_response = AsyncMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=valid_anthropic_response)
    
    anthropic_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    # Execute concurrent requests
    tasks = [
      anthropic_provider.generate_response(request)
      for request in requests
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # Verify all requests completed successfully
    assert len(responses) == 5
    for response in responses:
      assert isinstance(response, ModelResponse)
      assert response.provider == "anthropic"
    
    # Verify HTTP client was called for each request
    assert anthropic_provider.http_client.post.call_count == 5


class TestAnthropicRequestResponseHandling:
  """Detailed tests for Anthropic request/response handling methods."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="anthropic",
      api_key="sk-ant-api03-test-key",
      base_url="https://api.anthropic.com"
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return AsyncMock(spec=HTTPClient)

  @pytest.fixture
  def anthropic_provider(self, provider_config, http_client):
    """Create Anthropic provider instance."""
    return AnthropicProvider(provider_config, http_client)

  def test_prepare_request_basic_structure(self, anthropic_provider):
    """Test that _prepare_request creates correct basic structure."""
    request = ModelRequest(
      prompt="Hello, world!",
      model="claude-3-opus-20240229",
      temperature=0.5,
      max_tokens=100
    )
    
    payload = anthropic_provider._prepare_request(request)
    
    # Verify required fields
    assert payload["model"] == "claude-3-opus-20240229"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100
    
    # Verify messages structure
    assert "messages" in payload
    assert isinstance(payload["messages"], list)
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Hello, world!"

  def test_prepare_request_with_system_prompt(self, anthropic_provider):
    """Test _prepare_request with system prompt."""
    request = ModelRequest(
      prompt="User question",
      model="claude-3-opus-20240229",
      system_prompt="You are a helpful assistant."
    )
    
    payload = anthropic_provider._prepare_request(request)
    
    # Should have system field separate from messages
    assert payload["system"] == "You are a helpful assistant."
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "User question"

  def test_prepare_request_with_stop_sequences(self, anthropic_provider):
    """Test _prepare_request with stop sequences."""
    request = ModelRequest(
      prompt="Test prompt",
      model="claude-3-opus-20240229",
      stop_sequences=["STOP", "END", "\n\n"]
    )
    
    payload = anthropic_provider._prepare_request(request)
    
    assert payload["stop_sequences"] == ["STOP", "END", "\n\n"]

  def test_prepare_request_without_stop_sequences(self, anthropic_provider):
    """Test _prepare_request without stop sequences."""
    request = ModelRequest(
      prompt="Test prompt",
      model="claude-3-opus-20240229",
      stop_sequences=None
    )
    
    payload = anthropic_provider._prepare_request(request)
    
    assert "stop_sequences" not in payload

  def test_prepare_request_filters_streaming_parameters(self, anthropic_provider):
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
      model="claude-3-opus-20240229",
      provider_specific=streaming_params
    )
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      anthropic_provider._prepare_request(request)
    
    # Verify all streaming parameters are blocked
    blocked_params = exc_info.value.blocked_params
    assert "stream" in blocked_params
    assert "streaming" in blocked_params
    assert "stream_options" in blocked_params
    assert "stream_usage" in blocked_params
    assert "stream_callback" in blocked_params
    assert "stream_handler" in blocked_params
    assert "incremental" in blocked_params

  def test_prepare_request_includes_valid_provider_specific_params(self, anthropic_provider):
    """Test that valid provider-specific parameters are included."""
    valid_params = {
      "top_p": 0.9,
      "top_k": 40,
      "metadata": {"user_id": "123"}
    }
    
    request = ModelRequest(
      prompt="Test prompt",
      model="claude-3-opus-20240229",
      provider_specific=valid_params
    )
    
    payload = anthropic_provider._prepare_request(request)
    
    # All valid parameters should be included
    for key, value in valid_params.items():
      assert payload[key] == value

  def test_prepare_request_filters_invalid_provider_specific_params(self, anthropic_provider):
    """Test that invalid provider-specific parameters are filtered out."""
    mixed_params = {
      "top_p": 0.9,  # Valid
      "invalid_param": "value",  # Invalid
      "top_k": 40,  # Valid
      "custom_field": 123,  # Invalid
      "temperature": 0.8  # Valid but should be overridden by request.temperature
    }
    
    request = ModelRequest(
      prompt="Test prompt",
      model="claude-3-opus-20240229",
      temperature=0.7,  # This should take precedence
      provider_specific=mixed_params
    )
    
    payload = anthropic_provider._prepare_request(request)
    
    # Valid parameters should be included
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 40
    
    # Invalid parameters should be filtered out
    assert "invalid_param" not in payload
    assert "custom_field" not in payload
    
    # Request temperature should take precedence
    assert payload["temperature"] == 0.7

  def test_parse_response_basic_structure(self, anthropic_provider):
    """Test that _parse_response creates correct basic structure."""
    response_data = {
      "id": "msg_123",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Hello, world!"
        }
      ],
      "model": "claude-3-opus-20240229",
      "stop_reason": "end_turn",
      "usage": {
        "input_tokens": 10,
        "output_tokens": 5
      }
    }
    
    request = ModelRequest(prompt="Test", model="claude-3-opus-20240229")
    latency_ms = 1000
    
    response = anthropic_provider._parse_response(response_data, request, latency_ms)
    
    # Verify basic response structure
    assert isinstance(response, ModelResponse)
    assert response.text == "Hello, world!"
    assert response.model == "claude-3-opus-20240229"
    assert response.provider == "anthropic"
    assert response.latency_ms == latency_ms
    assert response.token_count == 15  # input_tokens + output_tokens
    assert response.finish_reason == "end_turn"

  def test_parse_response_with_metadata(self, anthropic_provider):
    """Test that _parse_response includes metadata correctly."""
    response_data = {
      "id": "msg_123",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Test response"
        }
      ],
      "model": "claude-3-opus-20240229",
      "stop_reason": "stop_sequence",
      "stop_sequence": "END",
      "usage": {
        "input_tokens": 20,
        "output_tokens": 10
      }
    }
    
    request = ModelRequest(prompt="Test", model="claude-3-opus-20240229")
    
    response = anthropic_provider._parse_response(response_data, request, 1500)
    
    # Verify metadata
    assert response.metadata["input_tokens"] == 20
    assert response.metadata["output_tokens"] == 10
    assert response.metadata["stop_reason"] == "stop_sequence"
    assert response.metadata["stop_sequence"] == "END"
    assert response.metadata["message_id"] == "msg_123"

  def test_validate_response_structure_comprehensive_validation(self, anthropic_provider):
    """Test comprehensive response structure validation."""
    # Test various invalid response structures
    invalid_responses = [
      # Missing content
      {
        "role": "assistant",
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
      },
      # Empty content array
      {
        "content": [],
        "role": "assistant",
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
      },
      # Invalid content block type
      {
        "content": [{"type": "image", "text": "test"}],
        "role": "assistant",
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
      },
      # Missing usage
      {
        "content": [{"type": "text", "text": "test"}],
        "role": "assistant",
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn"
      }
    ]
    
    for invalid_response in invalid_responses:
      with pytest.raises(ResponseValidationError):
        anthropic_provider._validate_response_structure(invalid_response)