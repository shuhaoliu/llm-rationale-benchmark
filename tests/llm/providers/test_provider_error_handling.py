"""Unit tests for provider-specific error handling with streaming detection."""

import pytest
from unittest.mock import AsyncMock, Mock

from rationale_benchmark.llm.exceptions import (
  AuthenticationError,
  ModelNotFoundError,
  NetworkError,
  ProviderError,
  RateLimitError,
  StreamingNotSupportedError,
)
from rationale_benchmark.llm.http.client import HTTPClient
from rationale_benchmark.llm.models import ModelRequest, ProviderConfig
from rationale_benchmark.llm.providers.anthropic import AnthropicProvider
from rationale_benchmark.llm.providers.gemini import GeminiProvider
from rationale_benchmark.llm.providers.openai import OpenAIProvider
from rationale_benchmark.llm.providers.openrouter import OpenRouterProvider


class TestOpenAIErrorHandling:
  """Test OpenAI provider error handling and mapping."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="openai",
      api_key="sk-test123",
      timeout=30,
      max_retries=3,
      models=["gpt-4", "gpt-3.5-turbo"]
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return Mock(spec=HTTPClient)

  @pytest.fixture
  def provider(self, provider_config, http_client):
    """Create OpenAI provider instance."""
    return OpenAIProvider(provider_config, http_client)

  def test_map_openai_error_401_authentication(self, provider):
    """Test mapping of 401 authentication errors."""
    response = Mock()
    response.status = 401
    
    error_data = {
      "error": {
        "message": "Invalid API key",
        "type": "invalid_request_error",
        "code": "invalid_api_key"
      }
    }
    
    exception = provider._map_openai_error(response, error_data)
    
    assert isinstance(exception, AuthenticationError)
    assert "OpenAI authentication failed" in str(exception)
    assert "OPENAI_API_KEY environment variable" in str(exception)
    assert "Verify your API key is valid" in str(exception)

  def test_map_openai_error_404_model_not_found(self, provider):
    """Test mapping of 404 model not found errors."""
    response = Mock()
    response.status = 404
    
    error_data = {
      "error": {
        "message": "The model 'gpt-5' does not exist",
        "type": "invalid_request_error",
        "code": "model_not_found"
      }
    }
    
    exception = provider._map_openai_error(response, error_data)
    
    assert isinstance(exception, ModelNotFoundError)
    assert "model not found" in str(exception).lower()
    assert "Check the model name spelling" in str(exception)
    assert "list_models()" in str(exception)

  def test_map_openai_error_429_rate_limit(self, provider):
    """Test mapping of 429 rate limit errors."""
    response = Mock()
    response.status = 429
    response.headers = {"retry-after": "60"}
    
    error_data = {
      "error": {
        "message": "Rate limit exceeded",
        "type": "rate_limit_error"
      }
    }
    
    exception = provider._map_openai_error(response, error_data)
    
    assert isinstance(exception, RateLimitError)
    assert exception.retry_after == 60
    assert "rate limit exceeded" in str(exception).lower()
    assert "exponential backoff" in str(exception)

  def test_map_openai_error_500_server_error(self, provider):
    """Test mapping of 500 server errors."""
    response = Mock()
    response.status = 500
    
    error_data = {
      "error": {
        "message": "Internal server error",
        "type": "server_error"
      }
    }
    
    exception = provider._map_openai_error(response, error_data)
    
    assert isinstance(exception, ProviderError)
    assert "Server error" in str(exception)
    assert "OpenAI status page" in str(exception)
    assert "exponential backoff" in str(exception)

  def test_map_openai_error_streaming_detection(self, provider):
    """Test detection of streaming-related errors."""
    response = Mock()
    response.status = 400
    
    error_data = {
      "error": {
        "message": "Streaming is not supported for this endpoint",
        "type": "invalid_request_error"
      }
    }
    
    exception = provider._map_openai_error(response, error_data)
    
    assert isinstance(exception, StreamingNotSupportedError)
    assert "streaming" in str(exception).lower()
    assert "Remove any 'stream' parameters" in str(exception)

  def test_map_openai_error_no_error_data(self, provider):
    """Test error mapping when no error data is available."""
    response = Mock()
    response.status = 400
    
    exception = provider._map_openai_error(response, None)
    
    assert isinstance(exception, ProviderError)
    assert "HTTP 400" in str(exception)
    assert "troubleshooting" in str(exception)


class TestAnthropicErrorHandling:
  """Test Anthropic provider error handling and mapping."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="anthropic",
      api_key="sk-ant-api03-test123",
      timeout=30,
      max_retries=3,
      models=["claude-3-opus-20240229"]
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return Mock(spec=HTTPClient)

  @pytest.fixture
  def provider(self, provider_config, http_client):
    """Create Anthropic provider instance."""
    return AnthropicProvider(provider_config, http_client)

  def test_map_anthropic_error_401_authentication(self, provider):
    """Test mapping of 401 authentication errors."""
    response = Mock()
    response.status = 401
    
    error_data = {
      "error": {
        "message": "Invalid API key",
        "type": "authentication_error"
      }
    }
    
    exception = provider._map_anthropic_error(response, error_data)
    
    assert isinstance(exception, AuthenticationError)
    assert "Anthropic authentication failed" in str(exception)
    assert "ANTHROPIC_API_KEY environment variable" in str(exception)
    assert "sk-ant-api" in str(exception)

  def test_map_anthropic_error_404_model_not_found(self, provider):
    """Test mapping of 404 model not found errors."""
    response = Mock()
    response.status = 404
    
    error_data = {
      "error": {
        "message": "Model claude-4 not found",
        "type": "not_found_error"
      }
    }
    
    exception = provider._map_anthropic_error(response, error_data)
    
    assert isinstance(exception, ModelNotFoundError)
    assert "model not found" in str(exception).lower()
    assert "claude-3-opus-20240229" in str(exception)

  def test_map_anthropic_error_429_rate_limit(self, provider):
    """Test mapping of 429 rate limit errors."""
    response = Mock()
    response.status = 429
    response.headers = {"retry-after": "30"}
    
    error_data = {
      "error": {
        "message": "Rate limit exceeded",
        "type": "rate_limit_error"
      }
    }
    
    exception = provider._map_anthropic_error(response, error_data)
    
    assert isinstance(exception, RateLimitError)
    assert exception.retry_after == 30
    assert "rate limit exceeded" in str(exception).lower()

  def test_map_anthropic_error_streaming_detection(self, provider):
    """Test detection of streaming-related errors."""
    response = Mock()
    response.status = 400
    
    error_data = {
      "error": {
        "message": "Streaming mode is not supported",
        "type": "invalid_request_error"
      }
    }
    
    exception = provider._map_anthropic_error(response, error_data)
    
    assert isinstance(exception, StreamingNotSupportedError)
    assert "streaming" in str(exception).lower()


class TestGeminiErrorHandling:
  """Test Gemini provider error handling and mapping."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="gemini",
      api_key="AIzaTest123456789012345678901234567890",
      timeout=30,
      max_retries=3,
      models=["gemini-1.5-pro"]
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return Mock(spec=HTTPClient)

  @pytest.fixture
  def provider(self, provider_config, http_client):
    """Create Gemini provider instance."""
    return GeminiProvider(provider_config, http_client)

  def test_map_gemini_error_401_authentication(self, provider):
    """Test mapping of 401 authentication errors."""
    response = Mock()
    response.status = 401
    
    error_data = {
      "error": {
        "message": "API key not valid",
        "code": 401,
        "status": "UNAUTHENTICATED"
      }
    }
    
    exception = provider._map_gemini_error(response, error_data)
    
    assert isinstance(exception, AuthenticationError)
    assert "Gemini authentication failed" in str(exception)
    assert "GOOGLE_API_KEY environment variable" in str(exception)
    assert "AIza" in str(exception)

  def test_map_gemini_error_403_api_not_enabled(self, provider):
    """Test mapping of 403 API not enabled errors."""
    response = Mock()
    response.status = 403
    
    error_data = {
      "error": {
        "message": "Generative Language API has not been used",
        "code": 403,
        "status": "PERMISSION_DENIED"
      }
    }
    
    exception = provider._map_gemini_error(response, error_data)
    
    assert isinstance(exception, ProviderError)
    assert "API not enabled" in str(exception)
    assert "Google Cloud Console" in str(exception)
    assert "Enable" in str(exception)

  def test_map_gemini_error_429_quota_exceeded(self, provider):
    """Test mapping of 429 quota exceeded errors."""
    response = Mock()
    response.status = 429
    
    error_data = {
      "error": {
        "message": "Quota exceeded",
        "code": 429,
        "status": "RESOURCE_EXHAUSTED"
      }
    }
    
    exception = provider._map_gemini_error(response, error_data)
    
    assert isinstance(exception, RateLimitError)
    assert "rate limit exceeded" in str(exception).lower()
    assert "quota limits" in str(exception)

  def test_map_gemini_error_content_safety(self, provider):
    """Test mapping of content safety errors."""
    response = Mock()
    response.status = 400
    
    error_data = {
      "error": {
        "message": "Content blocked due to safety filters",
        "code": 400
      }
    }
    
    exception = provider._map_gemini_error(response, error_data)
    
    assert isinstance(exception, ProviderError)
    assert "Content safety error" in str(exception)
    assert "safety filter" in str(exception)


class TestOpenRouterErrorHandling:
  """Test OpenRouter provider error handling and mapping."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="openrouter",
      api_key="sk-or-test123",
      timeout=30,
      max_retries=3,
      models=["openai/gpt-4"]
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return Mock(spec=HTTPClient)

  @pytest.fixture
  def provider(self, provider_config, http_client):
    """Create OpenRouter provider instance."""
    return OpenRouterProvider(provider_config, http_client)

  def test_map_openrouter_error_401_authentication(self, provider):
    """Test mapping of 401 authentication errors."""
    response = Mock()
    response.status = 401
    
    error_data = {
      "error": {
        "message": "Invalid API key",
        "type": "invalid_request_error"
      }
    }
    
    exception = provider._map_openrouter_error(response, error_data)
    
    assert isinstance(exception, AuthenticationError)
    assert "OpenRouter authentication failed" in str(exception)
    assert "OPENROUTER_API_KEY environment variable" in str(exception)

  def test_map_openrouter_error_402_payment_required(self, provider):
    """Test mapping of 402 payment required errors."""
    response = Mock()
    response.status = 402
    
    error_data = {
      "error": {
        "message": "Insufficient credits",
        "type": "insufficient_quota"
      }
    }
    
    exception = provider._map_openrouter_error(response, error_data)
    
    assert isinstance(exception, RateLimitError)
    assert "Payment required" in str(exception)
    assert "Add credits" in str(exception)

  def test_map_openrouter_error_model_unavailable(self, provider):
    """Test mapping of model unavailable errors."""
    response = Mock()
    response.status = 503
    
    error_data = {
      "error": {
        "message": "Model openai/gpt-4 is currently unavailable",
        "type": "model_unavailable"
      }
    }
    
    exception = provider._map_openrouter_error(response, error_data)
    
    assert isinstance(exception, NetworkError)
    assert "service unavailable" in str(exception).lower()
    assert "different model" in str(exception)

  def test_map_openrouter_error_streaming_detection(self, provider):
    """Test detection of streaming-related errors."""
    response = Mock()
    response.status = 400
    
    error_data = {
      "error": {
        "message": "Streaming parameter not allowed",
        "type": "invalid_request_error"
      }
    }
    
    exception = provider._map_openrouter_error(response, error_data)
    
    assert isinstance(exception, StreamingNotSupportedError)
    assert "streaming" in str(exception).lower()
    assert "explicitly blocked" in str(exception)


class TestStreamingDetection:
  """Test streaming parameter detection across all providers."""

  @pytest.fixture
  def provider_config(self):
    """Create test provider configuration."""
    return ProviderConfig(
      name="test",
      api_key="test-key",
      timeout=30,
      max_retries=3,
      models=["test-model"]
    )

  @pytest.fixture
  def http_client(self):
    """Create mock HTTP client."""
    return Mock(spec=HTTPClient)

  def test_detect_streaming_in_payload(self, provider_config, http_client):
    """Test detection of streaming parameters in request payload."""
    provider = OpenAIProvider(provider_config, http_client)
    
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={}
    )
    
    payload = {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Test"}],
      "stream": True  # This should be detected
    }
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      provider._detect_streaming_parameters(request, payload)
    
    assert "stream" in str(exc_info.value)
    assert "not supported" in str(exc_info.value)

  def test_detect_streaming_in_provider_specific(self, provider_config, http_client):
    """Test detection of streaming parameters in provider_specific section."""
    provider = OpenAIProvider(provider_config, http_client)
    
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={
        "stream_options": {"include_usage": True}  # This should be detected
      }
    )
    
    payload = {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Test"}],
      "stream": False
    }
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      provider._detect_streaming_parameters(request, payload)
    
    assert "stream_options" in str(exc_info.value)
    assert "provider_specific" in str(exc_info.value)

  def test_no_streaming_parameters_detected(self, provider_config, http_client):
    """Test that valid requests without streaming pass detection."""
    provider = OpenAIProvider(provider_config, http_client)
    
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={
        "temperature": 0.8,
        "max_tokens": 100
      }
    )
    
    payload = {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Test"}],
      "stream": False,
      "temperature": 0.8
    }
    
    # Should not raise any exception
    provider._detect_streaming_parameters(request, payload)

  def test_streaming_parameter_variations(self, provider_config, http_client):
    """Test detection of various streaming parameter names."""
    provider = OpenAIProvider(provider_config, http_client)
    
    streaming_variations = [
      "stream", "streaming", "stream_options", "stream_usage",
      "stream_callback", "incremental", "server_sent_events"
    ]
    
    for param_name in streaming_variations:
      request = ModelRequest(
        prompt="Test prompt",
        model="gpt-4",
        provider_specific={param_name: True}
      )
      
      payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Test"}],
        "stream": False
      }
      
      with pytest.raises(StreamingNotSupportedError) as exc_info:
        provider._detect_streaming_parameters(request, payload)
      
      assert param_name in str(exc_info.value)