"""Unit tests for provider-specific error handling with streaming detection."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from rationale_benchmark.llm.exceptions import (
  AuthenticationError,
  ModelNotFoundError,
  NetworkError,
  ProviderError,
  RateLimitError,
  ResponseValidationError,
  StreamingNotSupportedError,
)
from rationale_benchmark.llm.models import ModelRequest, ProviderConfig
from rationale_benchmark.llm.providers.openai import OpenAIProvider
from rationale_benchmark.llm.providers.anthropic import AnthropicProvider
from rationale_benchmark.llm.providers.gemini import GeminiProvider
from rationale_benchmark.llm.providers.openrouter import OpenRouterProvider


class TestProviderErrorMapping:
  """Test error mapping from provider APIs to custom exceptions."""
  
  @pytest.fixture
  def openai_provider(self):
    """Create OpenAI provider for testing."""
    config = ProviderConfig(
      name="openai",
      api_key="test-key",
      base_url="https://api.openai.com/v1",
      timeout=30,
      max_retries=3,
    )
    http_client = AsyncMock()
    return OpenAIProvider(config, http_client)
  
  @pytest.fixture
  def anthropic_provider(self):
    """Create Anthropic provider for testing."""
    config = ProviderConfig(
      name="anthropic",
      api_key="test-key",
      base_url="https://api.anthropic.com",
      timeout=30,
      max_retries=3,
    )
    http_client = AsyncMock()
    return AnthropicProvider(config, http_client)
  
  @pytest.fixture
  def sample_request(self):
    """Create sample model request for testing."""
    return ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.7,
      max_tokens=100,
    )
  
  @pytest.mark.asyncio
  async def test_openai_authentication_error_401(self, openai_provider, sample_request):
    """Test OpenAI 401 authentication error mapping."""
    # Mock HTTP response for 401 error
    mock_response = Mock()
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
      await openai_provider.generate_response(sample_request)
    
    error = exc_info.value
    assert error.provider == "openai"
    assert "Invalid API key" in str(error)
    assert "Check your OpenAI API key" in str(error)
  
  @pytest.mark.asyncio
  async def test_openai_model_not_found_404(self, openai_provider, sample_request):
    """Test OpenAI 404 model not found error mapping."""
    # Mock HTTP response for 404 error
    mock_response = Mock()
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
      await openai_provider.generate_response(sample_request)
    
    error = exc_info.value
    assert error.provider == "openai"
    assert "gpt-4" in error.model  # Should extract model from request
    assert "Check the model name spelling" in str(error)
  
  @pytest.mark.asyncio
  async def test_openai_rate_limit_429(self, openai_provider, sample_request):
    """Test OpenAI 429 rate limit error mapping."""
    # Mock HTTP response for 429 error
    mock_response = Mock()
    mock_response.status = 429
    mock_response.headers = {"retry-after": "60"}
    mock_response.json = AsyncMock(return_value={
      "error": {
        "message": "Rate limit exceeded",
        "type": "rate_limit_error",
        "code": "rate_limit_exceeded"
      }
    })
    
    openai_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    with pytest.raises(RateLimitError) as exc_info:
      await openai_provider.generate_response(sample_request)
    
    error = exc_info.value
    assert error.provider == "openai"
    assert error.retry_after == 60
    assert "Implement exponential backoff" in str(error)
  
  @pytest.mark.asyncio
  async def test_anthropic_authentication_error_401(self, anthropic_provider, sample_request):
    """Test Anthropic 401 authentication error mapping."""
    # Mock HTTP response for 401 error
    mock_response = Mock()
    mock_response.status = 401
    mock_response.json = AsyncMock(return_value={
      "error": {
        "type": "authentication_error",
        "message": "Invalid API key"
      }
    })
    
    anthropic_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    with pytest.raises(AuthenticationError) as exc_info:
      await anthropic_provider.generate_response(sample_request)
    
    error = exc_info.value
    assert error.provider == "anthropic"
    assert "Invalid API key" in str(error)
    assert "Check your Anthropic API key" in str(error)
  
  @pytest.mark.asyncio
  async def test_anthropic_rate_limit_429(self, anthropic_provider, sample_request):
    """Test Anthropic 429 rate limit error mapping."""
    # Mock HTTP response for 429 error
    mock_response = Mock()
    mock_response.status = 429
    mock_response.headers = {"retry-after": "30"}
    mock_response.json = AsyncMock(return_value={
      "error": {
        "type": "rate_limit_error",
        "message": "Rate limit exceeded"
      }
    })
    
    anthropic_provider.http_client.post = AsyncMock(return_value=mock_response)
    
    with pytest.raises(RateLimitError) as exc_info:
      await anthropic_provider.generate_response(sample_request)
    
    error = exc_info.value
    assert error.provider == "anthropic"
    assert error.retry_after == 30
    assert "Rate limit exceeded" in str(error)
  
  @pytest.mark.asyncio
  async def test_network_error_handling(self, openai_provider, sample_request):
    """Test network error handling and mapping."""
    import aiohttp
    
    # Mock network error
    openai_provider.http_client.post = AsyncMock(
      side_effect=aiohttp.ClientError("Connection timeout")
    )
    
    with pytest.raises(ProviderError) as exc_info:
      await openai_provider.generate_response(sample_request)
    
    error = exc_info.value
    assert error.provider == "openai"
    assert "Request failed" in str(error)
    assert "Connection timeout" in str(error)
  
  def test_provider_specific_error_guidance_openai(self, openai_provider):
    """Test OpenAI-specific error guidance messages."""
    # Test authentication error guidance
    auth_error = openai_provider._map_http_error(401, "Invalid API key", None)
    assert isinstance(auth_error, AuthenticationError)
    assert "Check your OpenAI API key" in str(auth_error)
    assert "Verify the key is active" in str(auth_error)
    
    # Test rate limit error guidance
    rate_error = openai_provider._map_http_error(429, "Rate limit exceeded", None)
    assert isinstance(rate_error, RateLimitError)
    assert "Implement exponential backoff" in str(rate_error)
    assert "Reduce your request frequency" in str(rate_error)
  
  def test_provider_specific_error_guidance_anthropic(self, anthropic_provider):
    """Test Anthropic-specific error guidance messages."""
    # Test authentication error guidance
    auth_error = anthropic_provider._map_http_error(401, "Invalid API key", None)
    assert isinstance(auth_error, AuthenticationError)
    assert "Check your Anthropic API key" in str(auth_error)
    assert "Verify the key format" in str(auth_error)
    
    # Test model error guidance
    model_error = anthropic_provider._map_http_error(404, "Model not found", None)
    assert isinstance(model_error, ProviderError)
    assert "Check the model name" in str(model_error)


class TestStreamingDetection:
  """Test streaming parameter detection and error creation."""
  
  @pytest.fixture
  def openai_provider(self):
    """Create OpenAI provider for testing."""
    config = ProviderConfig(
      name="openai",
      api_key="test-key",
      base_url="https://api.openai.com/v1",
      timeout=30,
      max_retries=3,
    )
    http_client = AsyncMock()
    return OpenAIProvider(config, http_client)
  
  def test_streaming_parameter_detection_in_payload(self, openai_provider):
    """Test detection of streaming parameters in request payload."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.7,
      max_tokens=100,
    )
    
    # Create payload with streaming parameter
    payload = {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Test prompt"}],
      "stream": True,  # This should be detected
      "temperature": 0.7,
    }
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      openai_provider._detect_streaming_parameters(request, payload)
    
    error = exc_info.value
    assert "stream" in error.blocked_params
    assert "Streaming not supported" in str(error)
    assert "Remove all streaming-related parameters" in str(error)
  
  def test_streaming_parameter_detection_in_provider_specific(self, openai_provider):
    """Test detection of streaming parameters in provider_specific section."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.7,
      max_tokens=100,
      provider_specific={
        "stream": True,
        "stream_options": {"include_usage": True},
      }
    )
    
    payload = {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Test prompt"}],
      "temperature": 0.7,
    }
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      openai_provider._detect_streaming_parameters(request, payload)
    
    error = exc_info.value
    assert "provider_specific.stream" in error.blocked_params
    assert "provider_specific.stream_options" in error.blocked_params
    assert len(error.blocked_params) == 2
  
  def test_streaming_parameter_detection_string_values(self, openai_provider):
    """Test detection of streaming parameters with string values."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.7,
      max_tokens=100,
      provider_specific={
        "stream": "true",  # String value should be detected
        "streaming": "True",  # Case insensitive
      }
    )
    
    payload = {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Test prompt"}],
      "temperature": 0.7,
    }
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      openai_provider._detect_streaming_parameters(request, payload)
    
    error = exc_info.value
    assert "provider_specific.stream" in error.blocked_params
    assert "provider_specific.streaming" in error.blocked_params
  
  def test_no_streaming_parameters_detected(self, openai_provider):
    """Test that no error is raised when no streaming parameters are present."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.7,
      max_tokens=100,
      provider_specific={
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
      }
    )
    
    payload = {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Test prompt"}],
      "temperature": 0.7,
      "stream": False,  # Explicitly disabled
    }
    
    # Should not raise any exception
    openai_provider._detect_streaming_parameters(request, payload)
  
  def test_streaming_not_supported_error_creation(self):
    """Test StreamingNotSupportedError creation with blocked parameters."""
    blocked_params = ["stream", "stream_options", "provider_specific.streaming"]
    
    error = StreamingNotSupportedError(
      "Streaming parameters detected",
      blocked_params=blocked_params
    )
    
    assert error.blocked_params == blocked_params
    assert len(error.blocked_params) == 3
    assert "stream" in error.blocked_params
    assert "stream_options" in error.blocked_params
    assert "provider_specific.streaming" in error.blocked_params
  
  @patch('rationale_benchmark.llm.logging.get_llm_logger')
  def test_streaming_detection_logging(self, mock_get_logger, openai_provider):
    """Test that streaming detection is properly logged."""
    mock_logger = Mock()
    mock_get_logger.return_value = mock_logger
    
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.7,
      max_tokens=100,
      provider_specific={"stream": True}
    )
    
    payload = {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Test prompt"}],
      "temperature": 0.7,
    }
    
    with pytest.raises(StreamingNotSupportedError):
      openai_provider._detect_streaming_parameters(request, payload)
    
    # Verify logging was called
    mock_logger.log_streaming_detection.assert_called_once()
    call_args = mock_logger.log_streaming_detection.call_args
    assert "provider_specific.stream" in call_args[1]["blocked_params"]
    assert call_args[1]["provider"] == "openai"
    assert call_args[1]["model"] == "gpt-4"


class TestAuthenticationErrorHandling:
  """Test authentication error handling with provider-specific instructions."""
  
  @pytest.fixture
  def providers(self):
    """Create providers for testing authentication errors."""
    config = ProviderConfig(
      name="test",
      api_key="test-key",
      timeout=30,
      max_retries=3,
    )
    http_client = AsyncMock()
    
    return {
      "openai": OpenAIProvider(config, http_client),
      "anthropic": AnthropicProvider(config, http_client),
      "gemini": GeminiProvider(config, http_client),
      "openrouter": OpenRouterProvider(config, http_client),
    }
  
  def test_openai_authentication_error_instructions(self, providers):
    """Test OpenAI authentication error provides specific instructions."""
    provider = providers["openai"]
    error = provider._map_http_error(401, "Invalid API key", None)
    
    assert isinstance(error, AuthenticationError)
    assert error.provider == "openai"
    error_message = str(error)
    
    # Check for OpenAI-specific instructions
    assert "Check your OpenAI API key" in error_message
    assert "Verify the key is active" in error_message
    assert "Check your account status" in error_message
    assert "openai.com" in error_message.lower()
  
  def test_anthropic_authentication_error_instructions(self, providers):
    """Test Anthropic authentication error provides specific instructions."""
    provider = providers["anthropic"]
    error = provider._map_http_error(401, "Invalid API key", None)
    
    assert isinstance(error, AuthenticationError)
    assert error.provider == "anthropic"
    error_message = str(error)
    
    # Check for Anthropic-specific instructions
    assert "Check your Anthropic API key" in error_message
    assert "Verify the key format" in error_message
    assert "console.anthropic.com" in error_message.lower()
  
  def test_gemini_authentication_error_instructions(self, providers):
    """Test Gemini authentication error provides specific instructions."""
    provider = providers["gemini"]
    error = provider._map_http_error(401, "Invalid API key", None)
    
    assert isinstance(error, AuthenticationError)
    assert error.provider == "gemini"
    error_message = str(error)
    
    # Check for Gemini-specific instructions
    assert "Check your Google AI API key" in error_message
    assert "Google AI Studio" in error_message
    assert "makersuite.google.com" in error_message.lower()
  
  def test_openrouter_authentication_error_instructions(self, providers):
    """Test OpenRouter authentication error provides specific instructions."""
    provider = providers["openrouter"]
    error = provider._map_http_error(401, "Invalid API key", None)
    
    assert isinstance(error, AuthenticationError)
    assert error.provider == "openrouter"
    error_message = str(error)
    
    # Check for OpenRouter-specific instructions
    assert "Check your OpenRouter API key" in error_message
    assert "openrouter.ai" in error_message.lower()
  
  def test_authentication_error_context_preservation(self, providers):
    """Test that authentication errors preserve context information."""
    provider = providers["openai"]
    
    # Test with additional context
    error = provider._map_http_error(
      401, 
      "Invalid API key provided", 
      {"request_id": "req_123", "model": "gpt-4"}
    )
    
    assert isinstance(error, AuthenticationError)
    assert error.provider == "openai"
    # Context should be preserved in the error for debugging
    assert hasattr(error, 'cause') or "Invalid API key" in str(error)


class TestProviderErrorRecovery:
  """Test error recovery suggestions for different provider error types."""
  
  @pytest.fixture
  def openai_provider(self):
    """Create OpenAI provider for testing."""
    config = ProviderConfig(
      name="openai",
      api_key="test-key",
      base_url="https://api.openai.com/v1",
      timeout=30,
      max_retries=3,
    )
    http_client = AsyncMock()
    return OpenAIProvider(config, http_client)
  
  def test_rate_limit_recovery_suggestions(self, openai_provider):
    """Test rate limit error includes recovery suggestions."""
    error = openai_provider._map_http_error(429, "Rate limit exceeded", None)
    
    assert isinstance(error, RateLimitError)
    error_message = str(error)
    
    # Check for recovery suggestions
    assert "Implement exponential backoff" in error_message
    assert "Reduce your request frequency" in error_message
    assert "Consider upgrading to a higher tier" in error_message
    assert "Distribute requests across multiple API keys" in error_message
  
  def test_model_not_found_recovery_suggestions(self, openai_provider):
    """Test model not found error includes recovery suggestions."""
    error = openai_provider._map_http_error(404, "Model 'gpt-5' not found", None)
    
    assert isinstance(error, ModelNotFoundError)
    error_message = str(error)
    
    # Check for recovery suggestions
    assert "Check the model name spelling" in error_message
    assert "Verify the model is available" in error_message
    assert "Ensure your account has access" in error_message
    assert "Use list_models()" in error_message
  
  def test_quota_exceeded_recovery_suggestions(self, openai_provider):
    """Test quota exceeded error includes recovery suggestions."""
    error = openai_provider._map_http_error(429, "Quota exceeded", None)
    
    assert isinstance(error, RateLimitError)
    error_message = str(error)
    
    # Check for quota-specific recovery suggestions
    assert "Check your account usage" in error_message
    assert "Add payment method" in error_message or "billing" in error_message.lower()
  
  def test_server_error_recovery_suggestions(self, openai_provider):
    """Test server error includes recovery suggestions."""
    error = openai_provider._map_http_error(500, "Internal server error", None)
    
    assert isinstance(error, ProviderError)
    error_message = str(error)
    
    # Check for server error recovery suggestions
    assert "Try again" in error_message or "retry" in error_message.lower()
    assert "temporary" in error_message.lower() or "server" in error_message.lower()