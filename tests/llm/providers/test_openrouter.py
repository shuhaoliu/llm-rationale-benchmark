"""Unit tests for OpenRouter provider implementation."""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from rationale_benchmark.llm.exceptions import (
  AuthenticationError,
  ModelNotFoundError,
  ProviderError,
  ResponseValidationError,
  StreamingNotSupportedError,
)
from rationale_benchmark.llm.http.client import HTTPClient
from rationale_benchmark.llm.models import ModelRequest, ModelResponse, ProviderConfig
from rationale_benchmark.llm.providers.openrouter import OpenRouterProvider


@pytest.fixture
def provider_config():
  """Create a test provider configuration."""
  return ProviderConfig(
    name="openrouter",
    api_key="test-api-key",
    base_url="https://openrouter.ai/api/v1",
    timeout=30,
    max_retries=3,
    models=["openai/gpt-4", "anthropic/claude-3-opus"],
    default_params={"temperature": 0.7, "max_tokens": 1000},
    provider_specific={}
  )


@pytest.fixture
def provider_config_with_custom_auth():
  """Create a test provider configuration with custom auth headers."""
  return ProviderConfig(
    name="openrouter",
    api_key="test-api-key",
    base_url="https://openrouter.ai/api/v1",
    timeout=30,
    max_retries=3,
    models=["openai/gpt-4"],
    default_params={},
    provider_specific={
      "auth_headers": {
        "X-Custom-Auth": "custom-value",
        "X-API-Version": "v1"
      }
    }
  )


@pytest.fixture
def http_client():
  """Create a mock HTTP client."""
  return AsyncMock(spec=HTTPClient)


@pytest.fixture
def openrouter_provider(provider_config, http_client):
  """Create an OpenRouter provider instance."""
  return OpenRouterProvider(provider_config, http_client)


@pytest.fixture
def openrouter_provider_custom_auth(provider_config_with_custom_auth, http_client):
  """Create an OpenRouter provider instance with custom auth."""
  return OpenRouterProvider(provider_config_with_custom_auth, http_client)


@pytest.fixture
def model_request():
  """Create a test model request."""
  return ModelRequest(
    prompt="What is the capital of France?",
    model="openai/gpt-4",
    temperature=0.7,
    max_tokens=1000,
    system_prompt="You are a helpful assistant.",
    stop_sequences=["END"],
    provider_specific={"frequency_penalty": 0.1}
  )


@pytest.fixture
def valid_openrouter_response():
  """Create a valid OpenRouter API response."""
  return {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "openai/gpt-4",
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
    },
    "provider": "openai"
  }


@pytest.fixture
def models_list_response():
  """Create a valid models list response."""
  return {
    "object": "list",
    "data": [
      {
        "id": "openai/gpt-4",
        "object": "model",
        "created": 1677610602,
        "owned_by": "openai"
      },
      {
        "id": "anthropic/claude-3-opus",
        "object": "model", 
        "created": 1677610602,
        "owned_by": "anthropic"
      }
    ]
  }


class TestOpenRouterProviderInitialization:
  """Test OpenRouter provider initialization."""

  def test_init_with_default_config(self, provider_config, http_client):
    """Test provider initialization with default configuration."""
    provider = OpenRouterProvider(provider_config, http_client)
    
    assert provider.config == provider_config
    assert provider.http_client == http_client
    assert provider.name == "openrouter"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert "Authorization" in provider.auth_headers
    assert provider.auth_headers["Authorization"] == "Bearer test-api-key"

  def test_init_with_custom_base_url(self, http_client):
    """Test provider initialization with custom base URL."""
    config = ProviderConfig(
      name="openrouter",
      api_key="test-key",
      base_url="https://custom.openrouter.com/v1",
      timeout=30,
      max_retries=3
    )
    
    provider = OpenRouterProvider(config, http_client)
    assert provider.base_url == "https://custom.openrouter.com/v1"

  def test_init_with_custom_auth_headers(self, provider_config_with_custom_auth, http_client):
    """Test provider initialization with custom authentication headers."""
    provider = OpenRouterProvider(provider_config_with_custom_auth, http_client)
    
    assert "Authorization" in provider.auth_headers
    assert "X-Custom-Auth" in provider.auth_headers
    assert "X-API-Version" in provider.auth_headers
    assert provider.auth_headers["Authorization"] == "Bearer test-api-key"
    assert provider.auth_headers["X-Custom-Auth"] == "custom-value"
    assert provider.auth_headers["X-API-Version"] == "v1"

  def test_init_without_api_key(self, http_client):
    """Test provider initialization without API key."""
    # Create config with minimal validation bypass
    config = ProviderConfig.__new__(ProviderConfig)
    config.name = "openrouter"
    config.api_key = ""
    config.timeout = 30
    config.max_retries = 3
    config.base_url = None
    config.models = []
    config.default_params = {}
    config.provider_specific = {}
    
    provider = OpenRouterProvider(config, http_client)
    assert "Authorization" not in provider.auth_headers

  def test_default_headers_setup(self, openrouter_provider):
    """Test that default headers are properly set up."""
    assert "Content-Type" in openrouter_provider._default_headers
    assert "HTTP-Referer" in openrouter_provider._default_headers
    assert "X-Title" in openrouter_provider._default_headers
    assert openrouter_provider._default_headers["Content-Type"] == "application/json"


class TestOpenRouterProviderGeneration:
  """Test OpenRouter provider response generation."""

  @pytest.mark.asyncio
  async def test_generate_response_success(
    self, 
    openrouter_provider, 
    model_request, 
    valid_openrouter_response
  ):
    """Test successful response generation."""
    # Mock HTTP response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=valid_openrouter_response)
    
    openrouter_provider.http_client.post.return_value = mock_response
    
    # Generate response
    response = await openrouter_provider.generate_response(model_request)
    
    # Verify response
    assert isinstance(response, ModelResponse)
    assert response.text == "The capital of France is Paris."
    assert response.model == "openai/gpt-4"
    assert response.provider == "openrouter"
    assert response.token_count == 23
    assert response.finish_reason == "stop"
    assert response.latency_ms >= 0
    assert response.cost_estimate is None  # OpenRouter doesn't provide cost estimates
    
    # Verify metadata
    assert "prompt_tokens" in response.metadata
    assert "completion_tokens" in response.metadata
    assert "request_id" in response.metadata
    assert "upstream_provider" in response.metadata
    assert response.metadata["upstream_provider"] == "openai"

  @pytest.mark.asyncio
  async def test_generate_response_with_streaming_blocked(
    self, 
    openrouter_provider, 
    model_request
  ):
    """Test that streaming parameters are blocked."""
    # Add streaming parameter to request
    model_request.provider_specific["stream"] = True
    
    # Should raise StreamingNotSupportedError
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      await openrouter_provider.generate_response(model_request)
    
    assert "stream" in str(exc_info.value)
    assert "stream" in exc_info.value.blocked_params

  @pytest.mark.asyncio
  async def test_generate_response_authentication_error(
    self, 
    openrouter_provider, 
    model_request
  ):
    """Test authentication error handling."""
    # Mock 401 response
    mock_response = AsyncMock()
    mock_response.status = 401
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "Invalid API key",
        "type": "authentication_error"
      }
    })
    
    openrouter_provider.http_client.post.return_value = mock_response
    
    # Should raise AuthenticationError
    with pytest.raises(AuthenticationError) as exc_info:
      await openrouter_provider.generate_response(model_request)
    
    assert "openrouter" in str(exc_info.value)
    assert "Invalid API key" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_model_not_found(
    self, 
    openrouter_provider, 
    model_request
  ):
    """Test model not found error handling."""
    # Mock 404 response
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "Model 'invalid/model' not found",
        "type": "model_not_found"
      }
    })
    
    openrouter_provider.http_client.post.return_value = mock_response
    
    # Should raise ModelNotFoundError
    with pytest.raises(ModelNotFoundError) as exc_info:
      await openrouter_provider.generate_response(model_request)
    
    assert "openrouter" in str(exc_info.value)
    assert model_request.model in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_rate_limit_error(
    self, 
    openrouter_provider, 
    model_request
  ):
    """Test rate limit error handling."""
    # Mock 429 response
    mock_response = AsyncMock()
    mock_response.status = 429
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "Rate limit exceeded",
        "type": "rate_limit_exceeded"
      }
    })
    
    openrouter_provider.http_client.post.return_value = mock_response
    
    # Should raise ProviderError
    with pytest.raises(ProviderError) as exc_info:
      await openrouter_provider.generate_response(model_request)
    
    assert "openrouter" in str(exc_info.value)
    assert "Rate limit exceeded" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_network_error(
    self, 
    openrouter_provider, 
    model_request
  ):
    """Test network error handling."""
    # Mock network error
    openrouter_provider.http_client.post.side_effect = Exception("Connection failed")
    
    # Should raise ProviderError
    with pytest.raises(ProviderError) as exc_info:
      await openrouter_provider.generate_response(model_request)
    
    assert "openrouter" in str(exc_info.value)
    assert "Request failed" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_generate_response_validates_headers(
    self, 
    openrouter_provider, 
    model_request, 
    valid_openrouter_response
  ):
    """Test that proper headers are sent with requests."""
    # Mock HTTP response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=valid_openrouter_response)
    
    openrouter_provider.http_client.post.return_value = mock_response
    
    # Generate response
    await openrouter_provider.generate_response(model_request)
    
    # Verify headers were sent
    call_args = openrouter_provider.http_client.post.call_args
    headers = call_args.kwargs["headers"]
    
    assert "Authorization" in headers
    assert "Content-Type" in headers
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers
    assert headers["Authorization"] == "Bearer test-api-key"
    assert headers["Content-Type"] == "application/json"

  @pytest.mark.asyncio
  async def test_generate_response_with_custom_auth_headers(
    self, 
    openrouter_provider_custom_auth, 
    model_request, 
    valid_openrouter_response
  ):
    """Test response generation with custom authentication headers."""
    # Mock HTTP response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=valid_openrouter_response)
    
    openrouter_provider_custom_auth.http_client.post.return_value = mock_response
    
    # Generate response
    await openrouter_provider_custom_auth.generate_response(model_request)
    
    # Verify custom headers were sent
    call_args = openrouter_provider_custom_auth.http_client.post.call_args
    headers = call_args.kwargs["headers"]
    
    assert "Authorization" in headers
    assert "X-Custom-Auth" in headers
    assert "X-API-Version" in headers
    assert headers["X-Custom-Auth"] == "custom-value"
    assert headers["X-API-Version"] == "v1"


class TestOpenRouterProviderModels:
  """Test OpenRouter provider model listing."""

  @pytest.mark.asyncio
  async def test_list_models_success(
    self, 
    openrouter_provider, 
    models_list_response
  ):
    """Test successful model listing."""
    # Mock HTTP response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=models_list_response)
    
    openrouter_provider.http_client.get.return_value = mock_response
    
    # List models
    models = await openrouter_provider.list_models()
    
    # Verify models
    assert isinstance(models, list)
    assert len(models) == 2
    assert "anthropic/claude-3-opus" in models
    assert "openai/gpt-4" in models
    assert models == sorted(models)  # Should be sorted

  @pytest.mark.asyncio
  async def test_list_models_authentication_error(self, openrouter_provider):
    """Test model listing with authentication error."""
    # Mock 401 response
    mock_response = AsyncMock()
    mock_response.status = 401
    mock_response.json = AsyncMock(return_value={
      "error": {"message": "Invalid API key"}
    })
    
    openrouter_provider.http_client.get.return_value = mock_response
    
    # Should raise AuthenticationError
    with pytest.raises(AuthenticationError):
      await openrouter_provider.list_models()

  @pytest.mark.asyncio
  async def test_list_models_invalid_response_format(self, openrouter_provider):
    """Test model listing with invalid response format."""
    # Mock invalid response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"invalid": "format"})
    
    openrouter_provider.http_client.get.return_value = mock_response
    
    # Should raise ProviderError
    with pytest.raises(ProviderError) as exc_info:
      await openrouter_provider.list_models()
    
    assert "Invalid models list response format" in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_list_models_network_error(self, openrouter_provider):
    """Test model listing with network error."""
    # Mock network error
    openrouter_provider.http_client.get.side_effect = Exception("Connection failed")
    
    # Should raise ProviderError
    with pytest.raises(ProviderError) as exc_info:
      await openrouter_provider.list_models()
    
    assert "Failed to list models" in str(exc_info.value)


class TestOpenRouterProviderValidation:
  """Test OpenRouter provider configuration validation."""

  def test_validate_config_valid(self, openrouter_provider):
    """Test validation of valid configuration."""
    errors = openrouter_provider.validate_config()
    assert errors == []

  def test_validate_config_missing_api_key(self, http_client):
    """Test validation with missing API key."""
    # Create config with minimal validation bypass
    config = ProviderConfig.__new__(ProviderConfig)
    config.name = "openrouter"
    config.api_key = ""
    config.timeout = 30
    config.max_retries = 3
    config.base_url = None
    config.models = []
    config.default_params = {}
    config.provider_specific = {}
    
    provider = OpenRouterProvider(config, http_client)
    errors = provider.validate_config()
    
    assert len(errors) == 1
    assert "API key is required" in errors[0]

  def test_validate_config_invalid_base_url(self, http_client):
    """Test validation with invalid base URL."""
    config = ProviderConfig(
      name="openrouter",
      api_key="test-key",
      base_url="http://insecure.com",
      timeout=30,
      max_retries=3
    )
    
    provider = OpenRouterProvider(config, http_client)
    errors = provider.validate_config()
    
    assert len(errors) == 1
    assert "should use HTTPS" in errors[0]

  def test_validate_config_invalid_timeout(self, http_client):
    """Test validation with invalid timeout."""
    # Create config with minimal validation bypass
    config = ProviderConfig.__new__(ProviderConfig)
    config.name = "openrouter"
    config.api_key = "test-key"
    config.timeout = 0
    config.max_retries = 3
    config.base_url = None
    config.models = []
    config.default_params = {}
    config.provider_specific = {}
    
    provider = OpenRouterProvider(config, http_client)
    errors = provider.validate_config()
    
    assert len(errors) == 1
    assert "Timeout must be positive" in errors[0]

  def test_validate_config_invalid_max_retries(self, http_client):
    """Test validation with invalid max retries."""
    # Create config with minimal validation bypass
    config = ProviderConfig.__new__(ProviderConfig)
    config.name = "openrouter"
    config.api_key = "test-key"
    config.timeout = 30
    config.max_retries = -1
    config.base_url = None
    config.models = []
    config.default_params = {}
    config.provider_specific = {}
    
    provider = OpenRouterProvider(config, http_client)
    errors = provider.validate_config()
    
    assert len(errors) == 1
    assert "Max retries cannot be negative" in errors[0]

  def test_validate_config_invalid_custom_headers(self, http_client):
    """Test validation with invalid custom auth headers."""
    config = ProviderConfig(
      name="openrouter",
      api_key="test-key",
      timeout=30,
      max_retries=3,
      provider_specific={"auth_headers": "invalid"}
    )
    
    provider = OpenRouterProvider(config, http_client)
    errors = provider.validate_config()
    
    assert len(errors) == 1
    assert "Custom auth_headers must be a dictionary" in errors[0]

  def test_validate_config_streaming_parameters(self, http_client):
    """Test validation with streaming parameters in configuration."""
    config = ProviderConfig(
      name="openrouter",
      api_key="test-key",
      timeout=30,
      max_retries=3,
      provider_specific={
        "stream": True,
        "stream_options": {"include_usage": True}
      }
    )
    
    provider = OpenRouterProvider(config, http_client)
    errors = provider.validate_config()
    
    assert len(errors) == 1
    assert "Streaming parameters not supported" in errors[0]
    assert "stream" in errors[0]
    assert "stream_options" in errors[0]

  def test_validate_config_multiple_errors(self, http_client):
    """Test validation with multiple configuration errors."""
    # Create config with minimal validation bypass
    config = ProviderConfig.__new__(ProviderConfig)
    config.name = "openrouter"
    config.api_key = ""
    config.base_url = "http://insecure.com"
    config.timeout = 0
    config.max_retries = -1
    config.models = []
    config.default_params = {}
    config.provider_specific = {}
    
    provider = OpenRouterProvider(config, http_client)
    errors = provider.validate_config()
    
    assert len(errors) == 4
    assert any("API key is required" in error for error in errors)
    assert any("should use HTTPS" in error for error in errors)
    assert any("Timeout must be positive" in error for error in errors)
    assert any("Max retries cannot be negative" in error for error in errors)


class TestOpenRouterRequestHandling:
  """Test OpenRouter provider request preparation and validation."""

  def test_prepare_request_basic(self, openrouter_provider, model_request):
    """Test basic request preparation."""
    payload = openrouter_provider._prepare_request(model_request)
    
    # Verify basic structure
    assert payload["model"] == "openai/gpt-4"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 1000
    assert payload["stream"] is False  # Critical: streaming disabled
    
    # Verify messages structure
    assert "messages" in payload
    assert len(payload["messages"]) == 2  # system + user
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "You are a helpful assistant."
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "What is the capital of France?"
    
    # Verify stop sequences
    assert payload["stop"] == ["END"]
    
    # Verify provider-specific parameters
    assert payload["frequency_penalty"] == 0.1

  def test_prepare_request_without_system_prompt(self, openrouter_provider):
    """Test request preparation without system prompt."""
    request = ModelRequest(
      prompt="Hello world",
      model="openai/gpt-4",
      temperature=0.5,
      max_tokens=500
    )
    
    payload = openrouter_provider._prepare_request(request)
    
    # Should only have user message
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Hello world"

  def test_prepare_request_without_stop_sequences(self, openrouter_provider):
    """Test request preparation without stop sequences."""
    request = ModelRequest(
      prompt="Hello world",
      model="openai/gpt-4",
      temperature=0.5,
      max_tokens=500
    )
    
    payload = openrouter_provider._prepare_request(request)
    
    # Should not have stop parameter
    assert "stop" not in payload

  def test_prepare_request_filters_streaming_params(self, openrouter_provider):
    """Test that streaming parameters are filtered out."""
    request = ModelRequest(
      prompt="Hello world",
      model="openai/gpt-4",
      temperature=0.5,
      max_tokens=500,
      provider_specific={
        "stream": True,
        "streaming": True,
        "stream_options": {"include_usage": True},
        "valid_param": "value"
      }
    )
    
    # Should raise StreamingNotSupportedError
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      openrouter_provider._prepare_request(request)
    
    # Verify blocked parameters
    blocked_params = exc_info.value.blocked_params
    assert "stream" in blocked_params
    assert "streaming" in blocked_params
    assert "stream_options" in blocked_params
    assert "valid_param" not in blocked_params

  def test_prepare_request_validates_openrouter_parameters(self, openrouter_provider):
    """Test that only valid OpenRouter parameters are included."""
    request = ModelRequest(
      prompt="Hello world",
      model="openai/gpt-4",
      temperature=0.5,
      max_tokens=500,
      provider_specific={
        "frequency_penalty": 0.1,  # Valid OpenAI param
        "provider": "openai",      # Valid OpenRouter param
        "invalid_param": "value",  # Invalid param
        "temperature": 0.8         # Valid but should not override
      }
    )
    
    payload = openrouter_provider._prepare_request(request)
    
    # Valid parameters should be included
    assert payload["frequency_penalty"] == 0.1
    assert payload["provider"] == "openai"
    
    # Invalid parameter should be filtered out
    assert "invalid_param" not in payload
    
    # Base temperature should not be overridden
    assert payload["temperature"] == 0.5

  def test_prepare_request_explicit_stream_false(self, openrouter_provider):
    """Test that stream is explicitly set to False."""
    request = ModelRequest(
      prompt="Hello world",
      model="openai/gpt-4"
    )
    
    payload = openrouter_provider._prepare_request(request)
    
    # Stream should be explicitly False
    assert "stream" in payload
    assert payload["stream"] is False

  def test_prepare_request_raises_on_stream_true(self, openrouter_provider):
    """Test that setting stream=True raises an error."""
    request = ModelRequest(
      prompt="Hello world",
      model="openai/gpt-4",
      provider_specific={"stream": True}
    )
    
    # Should raise StreamingNotSupportedError
    with pytest.raises(StreamingNotSupportedError):
      openrouter_provider._prepare_request(request)

  def test_is_valid_openrouter_parameter(self, openrouter_provider):
    """Test parameter validation for OpenRouter API."""
    # Valid OpenAI parameters
    assert openrouter_provider._is_valid_openrouter_parameter("temperature", 0.7)
    assert openrouter_provider._is_valid_openrouter_parameter("max_tokens", 1000)
    assert openrouter_provider._is_valid_openrouter_parameter("frequency_penalty", 0.1)
    assert openrouter_provider._is_valid_openrouter_parameter("top_p", 0.9)
    
    # Valid OpenRouter-specific parameters
    assert openrouter_provider._is_valid_openrouter_parameter("provider", "openai")
    assert openrouter_provider._is_valid_openrouter_parameter("route", "fallback")
    assert openrouter_provider._is_valid_openrouter_parameter("models", ["gpt-4"])
    
    # Invalid parameters
    assert not openrouter_provider._is_valid_openrouter_parameter("invalid_param", "value")
    assert not openrouter_provider._is_valid_openrouter_parameter("custom_field", 123)

  def test_validate_openrouter_request_valid(self, openrouter_provider):
    """Test validation of valid OpenRouter request."""
    payload = {
      "model": "openai/gpt-4",
      "messages": [{"role": "user", "content": "Hello"}],
      "temperature": 0.7,
      "stream": False
    }
    
    # Should not raise any exception
    openrouter_provider._validate_openrouter_request(payload)

  def test_validate_openrouter_request_missing_model(self, openrouter_provider):
    """Test validation with missing model."""
    payload = {
      "messages": [{"role": "user", "content": "Hello"}],
      "temperature": 0.7
    }
    
    # Should raise ProviderError
    with pytest.raises(ProviderError) as exc_info:
      openrouter_provider._validate_openrouter_request(payload)
    
    assert "Model must be specified" in str(exc_info.value)

  def test_validate_openrouter_request_empty_messages(self, openrouter_provider):
    """Test validation with empty messages."""
    payload = {
      "model": "openai/gpt-4",
      "messages": [],
      "temperature": 0.7
    }
    
    # Should raise ProviderError
    with pytest.raises(ProviderError) as exc_info:
      openrouter_provider._validate_openrouter_request(payload)
    
    assert "Messages must be a non-empty list" in str(exc_info.value)

  def test_validate_openrouter_request_streaming_enabled(self, openrouter_provider):
    """Test validation with streaming enabled."""
    payload = {
      "model": "openai/gpt-4",
      "messages": [{"role": "user", "content": "Hello"}],
      "stream": True
    }
    
    # Should raise StreamingNotSupportedError
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      openrouter_provider._validate_openrouter_request(payload)
    
    assert "does not support streaming" in str(exc_info.value)


class TestOpenRouterResponseHandling:
  """Test OpenRouter provider response parsing and validation."""

  def test_parse_response_basic(
    self, 
    openrouter_provider, 
    model_request, 
    valid_openrouter_response
  ):
    """Test basic response parsing."""
    response = openrouter_provider._parse_response(
      valid_openrouter_response, 
      model_request, 
      1500
    )
    
    # Verify basic response structure
    assert isinstance(response, ModelResponse)
    assert response.text == "The capital of France is Paris."
    assert response.model == "openai/gpt-4"
    assert response.provider == "openrouter"
    assert response.latency_ms == 1500
    assert response.token_count == 23
    assert response.finish_reason == "stop"
    assert response.cost_estimate is None  # OpenRouter doesn't provide cost estimates
    
    # Verify metadata
    assert response.metadata["prompt_tokens"] == 15
    assert response.metadata["completion_tokens"] == 8
    assert response.metadata["finish_reason"] == "stop"
    assert response.metadata["choice_index"] == 0
    assert response.metadata["request_id"] == "chatcmpl-test123"
    assert response.metadata["created"] == 1677652288
    assert response.metadata["upstream_provider"] == "openai"

  def test_parse_response_without_optional_fields(self, openrouter_provider, model_request):
    """Test response parsing without optional fields."""
    response_data = {
      "object": "chat.completion",
      "model": "openai/gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Test response"
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
    
    response = openrouter_provider._parse_response(response_data, model_request, 1000)
    
    # Should still work without optional fields
    assert response.text == "Test response"
    assert response.model == "openai/gpt-4"
    assert response.token_count == 15
    
    # Optional metadata should not be present
    assert "request_id" not in response.metadata
    assert "created" not in response.metadata
    assert "upstream_provider" not in response.metadata

  def test_validate_response_structure_valid(
    self, 
    openrouter_provider, 
    valid_openrouter_response
  ):
    """Test validation of valid response structure."""
    # Should not raise any exception
    openrouter_provider._validate_response_structure(valid_openrouter_response)

  def test_validate_response_structure_empty_response(self, openrouter_provider):
    """Test validation of empty response."""
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure({})
    
    assert "Empty response" in str(exc_info.value)
    assert exc_info.value.provider == "openrouter"

  def test_validate_response_structure_missing_required_fields(self, openrouter_provider):
    """Test validation with missing required fields."""
    response_data = {
      "model": "openai/gpt-4",
      "choices": []
    }
    # Missing: usage, object
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure(response_data)
    
    assert "Missing required field" in str(exc_info.value)
    assert exc_info.value.provider == "openrouter"

  def test_validate_response_structure_invalid_object_type(self, openrouter_provider):
    """Test validation with invalid object type."""
    response_data = {
      "object": "invalid.type",
      "model": "openai/gpt-4",
      "choices": [{"index": 0, "message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure(response_data)
    
    assert "Invalid object type" in str(exc_info.value)
    assert "chat.completion" in str(exc_info.value)

  def test_validate_response_structure_empty_choices(self, openrouter_provider):
    """Test validation with empty choices array."""
    response_data = {
      "object": "chat.completion",
      "model": "openai/gpt-4",
      "choices": [],
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure(response_data)
    
    assert "non-empty array" in str(exc_info.value)

  def test_validate_response_structure_invalid_choice_structure(self, openrouter_provider):
    """Test validation with invalid choice structure."""
    response_data = {
      "object": "chat.completion",
      "model": "openai/gpt-4",
      "choices": [{"invalid": "choice"}],
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure(response_data)
    
    assert "Missing required field" in str(exc_info.value)
    assert "choice" in str(exc_info.value)

  def test_validate_response_structure_invalid_message_role(self, openrouter_provider):
    """Test validation with invalid message role."""
    response_data = {
      "object": "chat.completion",
      "model": "openai/gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "user",  # Should be "assistant"
            "content": "Test response"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure(response_data)
    
    assert "Invalid message role" in str(exc_info.value)
    assert "assistant" in str(exc_info.value)

  def test_validate_response_structure_empty_content(self, openrouter_provider):
    """Test validation with empty message content."""
    response_data = {
      "object": "chat.completion",
      "model": "openai/gpt-4",
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
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure(response_data)
    
    assert "non-empty string" in str(exc_info.value)

  def test_validate_response_structure_whitespace_only_content(self, openrouter_provider):
    """Test validation with whitespace-only content."""
    response_data = {
      "object": "chat.completion",
      "model": "openai/gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "   \n\t  "  # Whitespace only
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure(response_data)
    
    assert "cannot be empty or whitespace only" in str(exc_info.value)

  def test_validate_response_structure_invalid_usage(self, openrouter_provider):
    """Test validation with invalid usage information."""
    response_data = {
      "object": "chat.completion",
      "model": "openai/gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Test response"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": -1,  # Invalid negative value
        "completion_tokens": 5,
        "total_tokens": 15
      }
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure(response_data)
    
    assert "non-negative integer" in str(exc_info.value)

  def test_validate_response_structure_inconsistent_token_count(self, openrouter_provider):
    """Test validation with inconsistent token counts."""
    response_data = {
      "object": "chat.completion",
      "model": "openai/gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Test response"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 20  # Should be 15
      }
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure(response_data)
    
    assert "token count inconsistency" in str(exc_info.value)

  def test_validate_response_structure_streaming_indicator(self, openrouter_provider):
    """Test validation with streaming indicator present."""
    response_data = {
      "object": "chat.completion",
      "model": "openai/gpt-4",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Test response"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
      "stream": True  # Should not be present
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openrouter_provider._validate_response_structure(response_data)
    
    assert "streaming mode" in str(exc_info.value)
    assert "not supported" in str(exc_info.value)

  def test_estimate_cost_returns_none(self, openrouter_provider):
    """Test that cost estimation returns None for OpenRouter."""
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    cost = openrouter_provider._estimate_cost(usage, "openai/gpt-4")
    
    # OpenRouter doesn't provide cost estimates
    assert cost is None


class TestOpenRouterProviderStreamingValidation:
  """Test comprehensive streaming parameter validation."""

  def test_streaming_params_blocked_in_provider_specific(self, openrouter_provider):
    """Test that all streaming parameters are blocked."""
    streaming_params = [
      "stream", "streaming", "stream_options", "stream_usage",
      "stream_callback", "stream_handler", "incremental"
    ]
    
    for param in streaming_params:
      request = ModelRequest(
        prompt="Test",
        model="openai/gpt-4",
        provider_specific={param: True}
      )
      
      with pytest.raises(StreamingNotSupportedError) as exc_info:
        openrouter_provider._prepare_request(request)
      
      assert param in exc_info.value.blocked_params

  def test_multiple_streaming_params_blocked(self, openrouter_provider):
    """Test that multiple streaming parameters are all blocked."""
    request = ModelRequest(
      prompt="Test",
      model="openai/gpt-4",
      provider_specific={
        "stream": True,
        "streaming": True,
        "stream_options": {"include_usage": True}
      }
    )
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      openrouter_provider._prepare_request(request)
    
    blocked = exc_info.value.blocked_params
    assert "stream" in blocked
    assert "streaming" in blocked
    assert "stream_options" in blocked

  def test_no_streaming_validation_helper(self, openrouter_provider):
    """Test the _validate_no_streaming_params helper method."""
    # Valid parameters should pass
    valid_params = {"temperature": 0.7, "max_tokens": 1000}
    openrouter_provider._validate_no_streaming_params(valid_params, "test")
    
    # Streaming parameters should raise error
    streaming_params = {"stream": True, "temperature": 0.7}
    with pytest.raises(StreamingNotSupportedError):
      openrouter_provider._validate_no_streaming_params(streaming_params, "test")

  def test_filter_streaming_params_helper(self, openrouter_provider):
    """Test the _filter_streaming_params helper method."""
    params = {
      "temperature": 0.7,
      "stream": True,
      "max_tokens": 1000,
      "streaming": True,
      "stream_options": {"include_usage": True}
    }
    
    filtered = openrouter_provider._filter_streaming_params(params)
    
    # Should keep non-streaming params
    assert "temperature" in filtered
    assert "max_tokens" in filtered
    
    # Should remove streaming params
    assert "stream" not in filtered
    assert "streaming" not in filtered
    assert "stream_options" not in filtered