"""Unit tests for LLMProvider abstract base class."""

import time
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from rationale_benchmark.llm.exceptions import (
    ProviderError,
    ResponseValidationError,
    StreamingNotSupportedError,
)
from rationale_benchmark.llm.http.client import HTTPClient
from rationale_benchmark.llm.models import ModelRequest, ModelResponse, ProviderConfig
from rationale_benchmark.llm.providers.base import LLMProvider


class TestLLMProvider:
  """Test suite for LLMProvider abstract base class."""

  @pytest.fixture
  def provider_config(self) -> ProviderConfig:
    """Create a test provider configuration."""
    return ProviderConfig(
      name="test-provider",
      api_key="test-api-key",
      base_url="https://api.test.com",
      timeout=30,
      max_retries=3,
      models=["test-model-1", "test-model-2"],
      default_params={"temperature": 0.7, "max_tokens": 1000},
      provider_specific={"custom_param": "value"}
    )

  @pytest.fixture
  def http_client(self) -> HTTPClient:
    """Create a mock HTTP client."""
    return MagicMock(spec=HTTPClient)

  @pytest.fixture
  def model_request(self) -> ModelRequest:
    """Create a test model request."""
    return ModelRequest(
      prompt="Test prompt",
      model="test-model",
      temperature=0.7,
      max_tokens=1000,
      system_prompt="You are a helpful assistant",
      stop_sequences=["END"],
      provider_specific={"custom_param": "value"}
    )

  @pytest.fixture
  def concrete_provider(self, provider_config: ProviderConfig, http_client: HTTPClient):
    """Create a concrete implementation of LLMProvider for testing."""
    
    class ConcreteProvider(LLMProvider):
      """Concrete implementation for testing."""
      
      async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Test implementation of generate_response."""
        return ModelResponse(
          text="Test response",
          model=request.model,
          provider=self.name,
          timestamp=datetime.now(),
          latency_ms=100,
          token_count=50,
          finish_reason="stop",
          metadata={"test": "data"}
        )
      
      async def list_models(self) -> List[str]:
        """Test implementation of list_models."""
        return self.config.models
      
      def validate_config(self) -> List[str]:
        """Test implementation of validate_config."""
        return []
      
      def _prepare_request(self, request: ModelRequest) -> Dict[str, Any]:
        """Test implementation of _prepare_request."""
        return {
          "model": request.model,
          "prompt": request.prompt,
          "temperature": request.temperature,
          "max_tokens": request.max_tokens
        }
      
      def _parse_response(
        self, 
        response_data: Dict[str, Any], 
        request: ModelRequest,
        latency_ms: int
      ) -> ModelResponse:
        """Test implementation of _parse_response."""
        return ModelResponse(
          text=response_data.get("text", ""),
          model=request.model,
          provider=self.name,
          timestamp=datetime.now(),
          latency_ms=latency_ms,
          token_count=response_data.get("token_count"),
          finish_reason=response_data.get("finish_reason"),
          metadata=response_data.get("metadata", {})
        )
    
    return ConcreteProvider(provider_config, http_client)

  def test_provider_initialization(
    self, 
    provider_config: ProviderConfig, 
    http_client: HTTPClient,
    concrete_provider: LLMProvider
  ):
    """Test that provider initializes correctly with config and HTTP client."""
    assert concrete_provider.config == provider_config
    assert concrete_provider.http_client == http_client
    assert concrete_provider.name == provider_config.name

  def test_provider_cannot_be_instantiated_directly(
    self, 
    provider_config: ProviderConfig, 
    http_client: HTTPClient
  ):
    """Test that LLMProvider cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
      LLMProvider(provider_config, http_client)

  @pytest.mark.asyncio
  async def test_abstract_methods_must_be_implemented(
    self, 
    provider_config: ProviderConfig, 
    http_client: HTTPClient
  ):
    """Test that all abstract methods must be implemented by subclasses."""
    
    class IncompleteProvider(LLMProvider):
      """Provider missing abstract method implementations."""
      pass
    
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
      IncompleteProvider(provider_config, http_client)

  @pytest.mark.asyncio
  async def test_generate_response_contract(
    self, 
    concrete_provider: LLMProvider, 
    model_request: ModelRequest
  ):
    """Test that generate_response returns ModelResponse."""
    response = await concrete_provider.generate_response(model_request)
    
    assert isinstance(response, ModelResponse)
    assert response.text == "Test response"
    assert response.model == model_request.model
    assert response.provider == concrete_provider.name
    assert isinstance(response.timestamp, datetime)
    assert response.latency_ms >= 0

  @pytest.mark.asyncio
  async def test_list_models_contract(self, concrete_provider: LLMProvider):
    """Test that list_models returns list of strings."""
    models = await concrete_provider.list_models()
    
    assert isinstance(models, list)
    assert all(isinstance(model, str) for model in models)
    assert models == concrete_provider.config.models

  def test_validate_config_contract(self, concrete_provider: LLMProvider):
    """Test that validate_config returns list of strings."""
    errors = concrete_provider.validate_config()
    
    assert isinstance(errors, list)
    assert all(isinstance(error, str) for error in errors)

  def test_validate_no_streaming_params_with_valid_params(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that valid parameters pass streaming validation."""
    valid_params = {
      "temperature": 0.7,
      "max_tokens": 1000,
      "model": "test-model"
    }
    
    # Should not raise any exception
    concrete_provider._validate_no_streaming_params(valid_params)

  def test_validate_no_streaming_params_with_streaming_params(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that streaming parameters are detected and rejected."""
    streaming_params = {
      "temperature": 0.7,
      "stream": True,
      "streaming": True,
      "stream_options": {"include_usage": True}
    }
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      concrete_provider._validate_no_streaming_params(streaming_params)
    
    assert "stream" in str(exc_info.value)
    assert "streaming" in str(exc_info.value)
    assert "stream_options" in str(exc_info.value)
    assert exc_info.value.blocked_params == ["stream", "streaming", "stream_options"]

  def test_validate_no_streaming_params_with_context(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that context is included in streaming parameter error messages."""
    streaming_params = {"stream": True}
    
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      concrete_provider._validate_no_streaming_params(
        streaming_params, 
        context="test context"
      )
    
    assert "test context" in str(exc_info.value)

  def test_filter_streaming_params_removes_streaming_params(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that streaming parameters are filtered out."""
    params_with_streaming = {
      "temperature": 0.7,
      "max_tokens": 1000,
      "stream": True,
      "streaming": True,
      "stream_options": {"include_usage": True},
      "stream_callback": lambda x: x,
      "custom_param": "value"
    }
    
    filtered = concrete_provider._filter_streaming_params(params_with_streaming)
    
    expected = {
      "temperature": 0.7,
      "max_tokens": 1000,
      "custom_param": "value"
    }
    
    assert filtered == expected

  def test_filter_streaming_params_with_no_streaming_params(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that non-streaming parameters are preserved."""
    params_without_streaming = {
      "temperature": 0.7,
      "max_tokens": 1000,
      "model": "test-model",
      "custom_param": "value"
    }
    
    filtered = concrete_provider._filter_streaming_params(params_without_streaming)
    
    assert filtered == params_without_streaming

  def test_measure_latency_calculates_correctly(self, concrete_provider: LLMProvider):
    """Test that latency measurement works correctly."""
    start_time = time.time()
    time.sleep(0.01)  # Sleep for 10ms
    latency = concrete_provider._measure_latency(start_time)
    
    assert isinstance(latency, int)
    assert latency >= 10  # Should be at least 10ms
    assert latency < 1000  # Should be less than 1 second

  def test_validate_response_not_empty_with_valid_response(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that valid response passes validation."""
    valid_response = {"text": "Test response", "model": "test-model"}
    
    # Should not raise any exception
    concrete_provider._validate_response_not_empty(valid_response)

  def test_validate_response_not_empty_with_empty_response(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that empty response raises validation error."""
    with pytest.raises(ResponseValidationError) as exc_info:
      concrete_provider._validate_response_not_empty({})
    
    assert "Empty response" in str(exc_info.value)
    assert exc_info.value.provider == concrete_provider.name

  def test_validate_response_not_empty_with_none_response(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that None response raises validation error."""
    with pytest.raises(ResponseValidationError) as exc_info:
      concrete_provider._validate_response_not_empty(None)
    
    assert "Empty response" in str(exc_info.value)
    assert exc_info.value.provider == concrete_provider.name

  def test_validate_response_not_empty_with_non_dict_response(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that non-dict response raises validation error."""
    with pytest.raises(ResponseValidationError) as exc_info:
      concrete_provider._validate_response_not_empty("not a dict")
    
    assert "must be a dictionary" in str(exc_info.value)
    assert exc_info.value.provider == concrete_provider.name

  def test_handle_http_error_with_400_status(self, concrete_provider: LLMProvider):
    """Test that HTTP 400 errors are handled correctly."""
    mock_response = MagicMock()
    mock_response.status = 400
    mock_response.json.return_value = {"error": "Bad request"}
    
    with pytest.raises(ProviderError) as exc_info:
      concrete_provider._handle_http_error(mock_response, "Test request")
    
    assert "Test request failed with status 400" in str(exc_info.value)
    assert "Bad request" in str(exc_info.value)
    assert exc_info.value.provider == concrete_provider.name

  def test_handle_http_error_with_500_status_no_error_details(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that HTTP 500 errors without error details are handled."""
    mock_response = MagicMock()
    mock_response.status = 500
    mock_response.json.side_effect = Exception("Cannot parse JSON")
    
    with pytest.raises(ProviderError) as exc_info:
      concrete_provider._handle_http_error(mock_response)
    
    assert "API request failed with status 500" in str(exc_info.value)
    assert exc_info.value.provider == concrete_provider.name

  def test_handle_http_error_with_200_status_does_not_raise(
    self, 
    concrete_provider: LLMProvider
  ):
    """Test that HTTP 200 status does not raise an error."""
    mock_response = MagicMock()
    mock_response.status = 200
    
    # Should not raise any exception
    concrete_provider._handle_http_error(mock_response)

  def test_str_representation(self, concrete_provider: LLMProvider):
    """Test string representation of provider."""
    str_repr = str(concrete_provider)
    assert "ConcreteProvider" in str_repr
    assert concrete_provider.name in str_repr

  def test_repr_representation(self, concrete_provider: LLMProvider):
    """Test detailed string representation of provider."""
    repr_str = repr(concrete_provider)
    assert "ConcreteProvider" in repr_str
    assert concrete_provider.name in repr_str
    assert str(concrete_provider.config.timeout) in repr_str
    assert concrete_provider.config.base_url in repr_str


class TestLLMProviderErrorHandling:
  """Test suite for LLMProvider error handling scenarios."""

  @pytest.fixture
  def provider_config(self) -> ProviderConfig:
    """Create a test provider configuration."""
    return ProviderConfig(
      name="error-test-provider",
      api_key="test-key",
      timeout=30,
      max_retries=3
    )

  @pytest.fixture
  def http_client(self) -> HTTPClient:
    """Create a mock HTTP client."""
    return MagicMock(spec=HTTPClient)

  @pytest.fixture
  def error_provider(self, provider_config: ProviderConfig, http_client: HTTPClient):
    """Create a provider that raises errors for testing."""
    
    class ErrorProvider(LLMProvider):
      """Provider that raises errors for testing."""
      
      async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Raises ProviderError."""
        raise ProviderError(self.name, "Test provider error")
      
      async def list_models(self) -> List[str]:
        """Raises ProviderError."""
        raise ProviderError(self.name, "Cannot list models")
      
      def validate_config(self) -> List[str]:
        """Returns validation errors."""
        return ["Invalid API key", "Missing required parameter"]
      
      def _prepare_request(self, request: ModelRequest) -> Dict[str, Any]:
        """Raises StreamingNotSupportedError."""
        raise StreamingNotSupportedError("Streaming not supported")
      
      def _parse_response(
        self, 
        response_data: Dict[str, Any], 
        request: ModelRequest,
        latency_ms: int
      ) -> ModelResponse:
        """Raises ResponseValidationError."""
        raise ResponseValidationError("Invalid response format")
    
    return ErrorProvider(provider_config, http_client)

  @pytest.mark.asyncio
  async def test_generate_response_error_propagation(
    self, 
    error_provider: LLMProvider, 
    model_request: ModelRequest
  ):
    """Test that generate_response errors are properly propagated."""
    with pytest.raises(ProviderError) as exc_info:
      await error_provider.generate_response(model_request)
    
    assert "Test provider error" in str(exc_info.value)
    assert exc_info.value.provider == error_provider.name

  @pytest.mark.asyncio
  async def test_list_models_error_propagation(self, error_provider: LLMProvider):
    """Test that list_models errors are properly propagated."""
    with pytest.raises(ProviderError) as exc_info:
      await error_provider.list_models()
    
    assert "Cannot list models" in str(exc_info.value)
    assert exc_info.value.provider == error_provider.name

  def test_validate_config_returns_errors(self, error_provider: LLMProvider):
    """Test that validate_config returns error messages."""
    errors = error_provider.validate_config()
    
    assert len(errors) == 2
    assert "Invalid API key" in errors
    assert "Missing required parameter" in errors

  def test_prepare_request_error_propagation(
    self, 
    error_provider: LLMProvider, 
    model_request: ModelRequest
  ):
    """Test that _prepare_request errors are properly propagated."""
    with pytest.raises(StreamingNotSupportedError) as exc_info:
      error_provider._prepare_request(model_request)
    
    assert "Streaming not supported" in str(exc_info.value)

  def test_parse_response_error_propagation(
    self, 
    error_provider: LLMProvider, 
    model_request: ModelRequest
  ):
    """Test that _parse_response errors are properly propagated."""
    response_data = {"text": "test"}
    
    with pytest.raises(ResponseValidationError) as exc_info:
      error_provider._parse_response(response_data, model_request, 100)
    
    assert "Invalid response format" in str(exc_info.value)


@pytest.fixture
def model_request() -> ModelRequest:
  """Create a test model request for module-level tests."""
  return ModelRequest(
    prompt="Test prompt",
    model="test-model",
    temperature=0.7,
    max_tokens=1000
  )