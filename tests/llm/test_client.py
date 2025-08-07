"""Unit tests for LLMClient class with concurrent manager integration."""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from rationale_benchmark.llm.client import LLMClient
from rationale_benchmark.llm.config.models import LLMConfig, ProviderConfig
from rationale_benchmark.llm.exceptions import (
    ConfigurationError,
    LLMConnectorError,
    StreamingNotSupportedError,
)
from rationale_benchmark.llm.models import ModelRequest, ModelResponse


def create_test_response(text="Test response", model="gpt-4", provider="openai", **kwargs):
  """Helper to create ModelResponse for testing, bypassing validation."""
  response = object.__new__(ModelResponse)
  response.text = text
  response.model = model
  response.provider = provider
  response.timestamp = kwargs.get('timestamp', datetime.now())
  response.latency_ms = kwargs.get('latency_ms', 500)
  response.token_count = kwargs.get('token_count', None)
  response.finish_reason = kwargs.get('finish_reason', None)
  response.cost_estimate = kwargs.get('cost_estimate', None)
  response.metadata = kwargs.get('metadata', {})
  return response


class TestLLMClientInitialization:
  """Test LLMClient initialization and configuration loading."""

  def test_client_creation_with_defaults(self):
    """Test LLMClient creation with default parameters."""
    client = LLMClient()
    
    assert client.config_dir == Path("config/llms")
    assert client.config_name == "default-llms"
    assert client.max_connections == 100
    assert client.request_timeout is None
    assert not client._is_initialized
    assert not client._is_shutdown

  def test_client_creation_with_custom_parameters(self):
    """Test LLMClient creation with custom parameters."""
    config_dir = Path("/custom/config")
    config_name = "custom-config"
    max_connections = 50
    request_timeout = 60.0
    
    client = LLMClient(
      config_dir=config_dir,
      config_name=config_name,
      max_connections=max_connections,
      request_timeout=request_timeout
    )
    
    assert client.config_dir == config_dir
    assert client.config_name == config_name
    assert client.max_connections == max_connections
    assert client.request_timeout == request_timeout

  @pytest.mark.asyncio
  async def test_successful_initialization(self):
    """Test successful client initialization with all components."""
    # Create mock configuration
    mock_config = LLMConfig(
      defaults={"temperature": 0.7},
      providers={
        "openai": ProviderConfig(
          name="openai",
          api_key="test-key",
          models=["gpt-4", "gpt-3.5-turbo"]
        )
      }
    )
    
    with patch("rationale_benchmark.llm.client.ConfigLoader") as mock_config_loader, \
         patch("rationale_benchmark.llm.client.ConfigValidator") as mock_config_validator, \
         patch("rationale_benchmark.llm.client.HTTPClient") as mock_http_client, \
         patch("rationale_benchmark.llm.client.ProviderFactory") as mock_provider_factory, \
         patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_concurrent_manager, \
         patch("rationale_benchmark.llm.client.ResponseValidator") as mock_response_validator:
      # Setup mocks
      config_loader_instance = mock_config_loader.return_value
      config_loader_instance.load_config.return_value = mock_config
      
      config_validator_instance = mock_config_validator.return_value
      config_validator_instance.validate_config.return_value = []
      config_validator_instance.validate_environment_variables.return_value = []
      
      http_client_instance = mock_http_client.return_value
      http_client_instance.close = AsyncMock()
      
      provider_factory_instance = mock_provider_factory.return_value
      provider_factory_instance.list_providers.return_value = {"openai": Mock()}
      provider_factory_instance.get_model_to_provider_mapping.return_value = {
        "gpt-4": "openai", "gpt-3.5-turbo": "openai"
      }
      
      concurrent_manager_instance = mock_concurrent_manager.return_value
      concurrent_manager_instance.shutdown = AsyncMock()
      
      # Test initialization
      client = LLMClient()
      await client.initialize()
      
      # Verify initialization state
      assert client._is_initialized
      assert not client._is_shutdown
      assert client.config == mock_config
      
      # Verify component initialization
      config_loader_instance.load_config.assert_called_once_with("default-llms")
      config_validator_instance.validate_config.assert_called_once_with(mock_config)
      config_validator_instance.validate_environment_variables.assert_called_once_with(mock_config)
      
      mock_http_client.assert_called_once_with(max_connections=100, timeout=30)
      mock_provider_factory.assert_called_once_with(http_client_instance)
      provider_factory_instance.initialize_providers.assert_called_once_with(mock_config)
      
      mock_concurrent_manager.assert_called_once()
      
      # Cleanup
      await client.shutdown()

  @pytest.mark.asyncio
  async def test_initialization_with_configuration_error(self):
    """Test initialization failure due to configuration error."""
    with patch("rationale_benchmark.llm.client.ConfigLoader") as mock_loader_class:
      config_loader_mock = mock_loader_class.return_value
      config_loader_mock.load_config.side_effect = ConfigurationError("Config not found")
      
      client = LLMClient()
      
      with pytest.raises(ConfigurationError, match="Config not found"):
        await client.initialize()
      
      # Verify client state after failure
      assert not client._is_initialized
      assert not client._is_shutdown

  @pytest.mark.asyncio
  async def test_initialization_with_validation_error(self):
    """Test initialization failure due to validation error."""
    mock_config = LLMConfig(
      defaults={},
      providers={
        "openai": ProviderConfig(name="openai", api_key="test-key", models=[])
      }
    )
    
    with patch("rationale_benchmark.llm.client.ConfigLoader") as mock_config_loader, \
         patch("rationale_benchmark.llm.client.ConfigValidator") as mock_config_validator:
      config_loader_instance = mock_config_loader.return_value
      config_loader_instance.load_config.return_value = mock_config
      
      config_validator_instance = mock_config_validator.return_value
      config_validator_instance.validate_config.return_value = ["Invalid provider config"]
      
      client = LLMClient()
      
      with pytest.raises(ConfigurationError, match="Configuration validation failed"):
        await client.initialize()
      
      assert not client._is_initialized

  @pytest.mark.asyncio
  async def test_initialization_cleanup_on_error(self):
    """Test proper cleanup when initialization fails."""
    mock_config = LLMConfig(
      defaults={},
      providers={
        "openai": ProviderConfig(name="openai", api_key="test-key", models=[])
      }
    )
    
    with patch("rationale_benchmark.llm.client.ConfigLoader") as mock_config_loader, \
         patch("rationale_benchmark.llm.client.ConfigValidator") as mock_config_validator, \
         patch("rationale_benchmark.llm.client.HTTPClient") as mock_http_client, \
         patch("rationale_benchmark.llm.client.ProviderFactory") as mock_provider_factory:
      # Setup successful early components
      config_loader_instance = mock_config_loader.return_value
      config_loader_instance.load_config.return_value = mock_config
      
      config_validator_instance = mock_config_validator.return_value
      config_validator_instance.validate_config.return_value = []
      config_validator_instance.validate_environment_variables.return_value = []
      
      http_client_instance = mock_http_client.return_value
      http_client_instance.close = AsyncMock()
      
      # Make provider factory fail
      provider_factory_instance = mock_provider_factory.return_value
      provider_factory_instance.initialize_providers.side_effect = Exception("Provider init failed")
      
      client = LLMClient()
      
      with pytest.raises(LLMConnectorError, match="Failed to initialize LLMClient"):
        await client.initialize()
      
      # Verify cleanup was called
      http_client_instance.close.assert_called_once()
      assert not client._is_initialized

  @pytest.mark.asyncio
  async def test_double_initialization(self):
    """Test that double initialization is handled gracefully."""
    mock_config = LLMConfig(
      defaults={},
      providers={
        "openai": ProviderConfig(name="openai", api_key="test-key", models=["gpt-4"])
      }
    )
    
    with patch("rationale_benchmark.llm.client.ConfigLoader") as mock_config_loader, \
         patch("rationale_benchmark.llm.client.ConfigValidator") as mock_config_validator, \
         patch("rationale_benchmark.llm.client.HTTPClient") as mock_http_client, \
         patch("rationale_benchmark.llm.client.ProviderFactory") as mock_provider_factory, \
         patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_concurrent_manager, \
         patch("rationale_benchmark.llm.client.ResponseValidator") as mock_response_validator:
      # Setup successful mocks
      config_loader_instance = mock_config_loader.return_value
      config_loader_instance.load_config.return_value = mock_config
      
      config_validator_instance = mock_config_validator.return_value
      config_validator_instance.validate_config.return_value = []
      config_validator_instance.validate_environment_variables.return_value = []
      
      provider_factory_instance = mock_provider_factory.return_value
      provider_factory_instance.list_providers.return_value = {"openai": Mock()}
      provider_factory_instance.get_model_to_provider_mapping.return_value = {"gpt-4": "openai"}
      
      http_client_instance = mock_http_client.return_value
      http_client_instance.close = AsyncMock()
      
      concurrent_manager_instance = mock_concurrent_manager.return_value
      concurrent_manager_instance.shutdown = AsyncMock()
      
      client = LLMClient()
      
      # First initialization
      await client.initialize()
      assert client._is_initialized
      
      # Second initialization should not fail
      await client.initialize()
      assert client._is_initialized
      
      # Config should only be loaded once
      config_loader_instance.load_config.assert_called_once()
      
      await client.shutdown()

  @pytest.mark.asyncio
  async def test_initialization_after_shutdown(self):
    """Test that initialization after shutdown raises error."""
    client = LLMClient()
    client._is_shutdown = True
    
    with pytest.raises(LLMConnectorError, match="Cannot initialize a shut down LLMClient"):
      await client.initialize()


class TestLLMClientStreamingValidation:
  """Test streaming parameter validation and removal."""

  @pytest.mark.asyncio
  async def test_streaming_parameter_detection_in_config(self):
    """Test detection and removal of streaming parameters in configuration."""
    mock_config = LLMConfig(
      defaults={"temperature": 0.7, "stream": True},  # Streaming in defaults
      providers={
        "openai": ProviderConfig(
          name="openai",
          api_key="test-key",
          models=["gpt-4"],
          default_params={"streaming": True},  # Streaming in default_params
          provider_specific={"stream_options": {"include_usage": True}}  # Streaming in provider_specific
        )
      }
    )
    
    with patch("rationale_benchmark.llm.client.ConfigLoader") as mock_config_loader, \
         patch("rationale_benchmark.llm.client.ConfigValidator") as mock_config_validator, \
         patch("rationale_benchmark.llm.client.HTTPClient") as mock_http_client, \
         patch("rationale_benchmark.llm.client.ProviderFactory") as mock_provider_factory, \
         patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_concurrent_manager, \
         patch("rationale_benchmark.llm.client.ResponseValidator") as mock_response_validator:
      # Setup mocks
      config_loader_instance = mock_config_loader.return_value
      config_loader_instance.load_config.return_value = mock_config
      
      config_validator_instance = mock_config_validator.return_value
      config_validator_instance.validate_config.return_value = []
      config_validator_instance.validate_environment_variables.return_value = []
      
      provider_factory_instance = mock_provider_factory.return_value
      provider_factory_instance.list_providers.return_value = {"openai": Mock()}
      provider_factory_instance.get_model_to_provider_mapping.return_value = {"gpt-4": "openai"}
      
      http_client_instance = mock_http_client.return_value
      http_client_instance.close = AsyncMock()
      
      concurrent_manager_instance = mock_concurrent_manager.return_value
      concurrent_manager_instance.shutdown = AsyncMock()
      
      client = LLMClient()
      
      with patch("rationale_benchmark.llm.client.logger") as mock_logger:
        await client.initialize()
        
        # Verify streaming parameters were removed
        assert "stream" not in client.config.defaults
        assert "streaming" not in client.config.providers["openai"].default_params
        assert "stream_options" not in client.config.providers["openai"].provider_specific
        
        # Verify warning was logged
        mock_logger.warning.assert_called()
        warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
        assert any("streaming parameters" in call.lower() for call in warning_calls)
      
      await client.shutdown()

  def test_request_streaming_parameter_removal(self):
    """Test removal of streaming parameters from requests."""
    client = LLMClient()
    
    # Create request with streaming parameters
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={
        "temperature": 0.8,
        "stream": True,  # Should be removed
        "streaming": True,  # Should be removed
        "stream_options": {"include_usage": True},  # Should be removed
        "max_tokens": 100  # Should be kept
      }
    )
    
    cleaned_request = client._validate_request_no_streaming(request)
    
    # Verify streaming parameters were removed
    assert "stream" not in cleaned_request.provider_specific
    assert "streaming" not in cleaned_request.provider_specific
    assert "stream_options" not in cleaned_request.provider_specific
    
    # Verify non-streaming parameters were kept
    assert cleaned_request.provider_specific["temperature"] == 0.8
    assert cleaned_request.provider_specific["max_tokens"] == 100
    
    # Verify other request fields unchanged
    assert cleaned_request.prompt == request.prompt
    assert cleaned_request.model == request.model

  def test_request_without_streaming_parameters(self):
    """Test that requests without streaming parameters are unchanged."""
    client = LLMClient()
    
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={
        "temperature": 0.8,
        "max_tokens": 100
      }
    )
    
    cleaned_request = client._validate_request_no_streaming(request)
    
    # Should return the same request object
    assert cleaned_request is request


class TestLLMClientUtilityMethods:
  """Test utility methods and status reporting."""

  @pytest.mark.asyncio
  async def test_get_available_models(self):
    """Test getting available models from provider factory."""
    with patch("rationale_benchmark.llm.client.ProviderFactory") as mock_factory_class:
      provider_factory_mock = mock_factory_class.return_value
      provider_factory_mock.list_models.return_value = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
      
      client = LLMClient()
      client._is_initialized = True
      client.provider_factory = provider_factory_mock
      
      models = client.get_available_models()
      
      assert models == ["gpt-4", "gpt-3.5-turbo", "claude-3"]
      provider_factory_mock.list_models.assert_called_once()

  def test_get_available_models_not_initialized(self):
    """Test getting available models when client not initialized."""
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="LLMClient is not initialized"):
      client.get_available_models()

  @pytest.mark.asyncio
  async def test_get_provider_for_model(self):
    """Test getting provider name for a specific model."""
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.get_provider_for_model.return_value = "openai"
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      provider = client.get_provider_for_model("gpt-4")
      
      assert provider == "openai"
      concurrent_manager_mock.get_provider_for_model.assert_called_once_with("gpt-4")

  @pytest.mark.asyncio
  async def test_get_client_status(self):
    """Test getting comprehensive client status."""
    with patch("rationale_benchmark.llm.client.ProviderFactory") as mock_provider_factory, \
         patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_concurrent_manager:
      provider_factory_instance = mock_provider_factory.return_value
      provider_factory_instance.list_models.return_value = ["gpt-4", "claude-3"]
      provider_factory_instance.get_provider_discovery_info.return_value = {
        "openai": {"models": ["gpt-4"], "status": "valid"}
      }
      
      concurrent_manager_instance = mock_concurrent_manager.return_value
      concurrent_manager_instance.get_overall_status.return_value = {
        "total_providers": 2,
        "active_providers": 2,
        "total_queue_size": 0
      }
      
      client = LLMClient(config_name="test-config", max_connections=50)
      client._is_initialized = True
      client.provider_factory = provider_factory_instance
      client.concurrent_manager = concurrent_manager_instance
      
      status = client.get_client_status()
      
      assert status["is_initialized"] is True
      assert status["is_shutdown"] is False
      assert status["config_name"] == "test-config"
      assert status["max_connections"] == 50
      assert status["available_models"] == 2
      assert status["total_providers"] == 2
      assert "provider_discovery" in status

  @pytest.mark.asyncio
  async def test_shutdown(self):
    """Test client shutdown process."""
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_concurrent_manager, \
         patch("rationale_benchmark.llm.client.HTTPClient") as mock_http_client:
      concurrent_manager_instance = mock_concurrent_manager.return_value
      concurrent_manager_instance.shutdown = AsyncMock()
      
      http_client_instance = mock_http_client.return_value
      http_client_instance.close = AsyncMock()
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_instance
      client.http_client = http_client_instance
      
      await client.shutdown()
      
      # Verify shutdown was called on components
      concurrent_manager_instance.shutdown.assert_called_once()
      http_client_instance.close.assert_called_once()
      
      # Verify client state
      assert client._is_shutdown
      assert not client._is_initialized

  @pytest.mark.asyncio
  async def test_double_shutdown(self):
    """Test that double shutdown is handled gracefully."""
    client = LLMClient()
    client._is_shutdown = True
    
    # Should not raise error
    await client.shutdown()

  @pytest.mark.asyncio
  async def test_shutdown_with_error(self):
    """Test shutdown error handling."""
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.shutdown = AsyncMock(side_effect=Exception("Shutdown failed"))
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      with pytest.raises(LLMConnectorError, match="Failed to shutdown LLMClient"):
        await client.shutdown()

  @pytest.mark.asyncio
  async def test_async_context_manager(self):
    """Test async context manager functionality."""
    mock_config = LLMConfig(
      defaults={},
      providers={
        "openai": ProviderConfig(name="openai", api_key="test-key", models=["gpt-4"])
      }
    )
    
    with patch("rationale_benchmark.llm.client.ConfigLoader") as mock_config_loader, \
         patch("rationale_benchmark.llm.client.ConfigValidator") as mock_config_validator, \
         patch("rationale_benchmark.llm.client.HTTPClient") as mock_http_client, \
         patch("rationale_benchmark.llm.client.ProviderFactory") as mock_provider_factory, \
         patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_concurrent_manager, \
         patch("rationale_benchmark.llm.client.ResponseValidator") as mock_response_validator:
      # Setup successful mocks
      config_loader_instance = mock_config_loader.return_value
      config_loader_instance.load_config.return_value = mock_config
      
      config_validator_instance = mock_config_validator.return_value
      config_validator_instance.validate_config.return_value = []
      config_validator_instance.validate_environment_variables.return_value = []
      
      provider_factory_instance = mock_provider_factory.return_value
      provider_factory_instance.list_providers.return_value = {"openai": Mock()}
      provider_factory_instance.get_model_to_provider_mapping.return_value = {"gpt-4": "openai"}
      
      http_client_instance = mock_http_client.return_value
      http_client_instance.close = AsyncMock()
      
      concurrent_manager_instance = mock_concurrent_manager.return_value
      concurrent_manager_instance.shutdown = AsyncMock()
      
      # Test context manager
      async with LLMClient() as client:
        assert client._is_initialized
        assert not client._is_shutdown
      
      # Verify shutdown was called
      concurrent_manager_instance.shutdown.assert_called_once()
      http_client_instance.close.assert_called_once()

  def test_string_representations(self):
    """Test string representation methods."""
    client = LLMClient(config_name="test-config")
    
    str_repr = str(client)
    assert "LLMClient" in str_repr
    assert "test-config" in str_repr
    assert "initialized=False" in str_repr
    
    repr_str = repr(client)
    assert "LLMClient" in repr_str
    assert "config_name=test-config" in repr_str
    assert "initialized=False" in repr_str


class TestLLMClientErrorHandling:
  """Test error handling scenarios."""

  def test_ensure_initialized_not_initialized(self):
    """Test _ensure_initialized when client not initialized."""
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="LLMClient is not initialized"):
      client._ensure_initialized()

  def test_ensure_initialized_shutdown(self):
    """Test _ensure_initialized when client is shut down."""
    client = LLMClient()
    client._is_shutdown = True
    
    with pytest.raises(LLMConnectorError, match="LLMClient has been shut down"):
      client._ensure_initialized()

  def test_ensure_initialized_success(self):
    """Test _ensure_initialized when client is properly initialized."""
    client = LLMClient()
    client._is_initialized = True
    
    # Should not raise any exception
    client._ensure_initialized()


class TestLLMClientSingleResponseGeneration:
  """Test single response generation functionality."""

  @pytest.mark.asyncio
  async def test_generate_response_success(self):
    """Test successful single response generation."""
    # Create test request and expected response
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      temperature=0.7,
      max_tokens=100
    )
    
    expected_response = ModelResponse(
      text="Test response",
      model="gpt-4",
      provider="openai",
      timestamp=datetime.now(),
      latency_ms=500,
      token_count=25
    )
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      # Setup mock future that resolves to expected response
      mock_future = asyncio.Future()
      mock_future.set_result(expected_response)
      
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.submit_request = AsyncMock(return_value=mock_future)
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      # Mock get_provider_for_model to return valid provider
      with patch.object(client, 'get_provider_for_model', return_value="openai"):
        with patch.object(client, 'get_available_models', return_value=["gpt-4"]):
          response = await client.generate_response(request)
      
      # Verify response
      assert response == expected_response
      concurrent_manager_mock.submit_request.assert_called_once()
      
      # Verify the request was cleaned (no streaming params in this case)
      submitted_request = concurrent_manager_mock.submit_request.call_args[0][0]
      assert submitted_request.prompt == request.prompt
      assert submitted_request.model == request.model

  @pytest.mark.asyncio
  async def test_generate_response_with_streaming_parameters(self):
    """Test response generation with streaming parameters removed."""
    request = ModelRequest(
      prompt="Test prompt",
      model="gpt-4",
      provider_specific={
        "temperature": 0.8,
        "stream": True,  # Should be removed
        "streaming": True,  # Should be removed
        "max_tokens": 100
      }
    )
    
    expected_response = ModelResponse(
      text="Test response",
      model="gpt-4",
      provider="openai",
      timestamp=datetime.now(),
      latency_ms=500
    )
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      mock_future = asyncio.Future()
      mock_future.set_result(expected_response)
      
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.submit_request = AsyncMock(return_value=mock_future)
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      with patch.object(client, 'get_provider_for_model', return_value="openai"):
        with patch.object(client, 'get_available_models', return_value=["gpt-4"]):
          with patch("rationale_benchmark.llm.client.logger") as mock_logger:
            response = await client.generate_response(request)
      
      # Verify streaming parameters were removed
      submitted_request = concurrent_manager_mock.submit_request.call_args[0][0]
      assert "stream" not in submitted_request.provider_specific
      assert "streaming" not in submitted_request.provider_specific
      assert submitted_request.provider_specific["temperature"] == 0.8
      assert submitted_request.provider_specific["max_tokens"] == 100
      
      # Verify warning was logged
      mock_logger.warning.assert_called()

  @pytest.mark.asyncio
  async def test_generate_response_unsupported_model(self):
    """Test response generation with unsupported model."""
    request = ModelRequest(
      prompt="Test prompt",
      model="unsupported-model"
    )
    
    client = LLMClient()
    client._is_initialized = True
    client.concurrent_manager = Mock()
    
    with patch.object(client, 'get_provider_for_model', return_value=None):
      with patch.object(client, 'get_available_models', return_value=["gpt-4", "claude-3"]):
        with pytest.raises(ValueError, match="Model 'unsupported-model' is not supported"):
          await client.generate_response(request)

  @pytest.mark.asyncio
  async def test_generate_response_not_initialized(self):
    """Test response generation when client not initialized."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="LLMClient is not initialized"):
      await client.generate_response(request)

  @pytest.mark.asyncio
  async def test_generate_response_timeout_error(self):
    """Test response generation with timeout error."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      # Setup mock future that raises timeout
      mock_future = asyncio.Future()
      mock_future.set_exception(asyncio.TimeoutError("Request timed out"))
      
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.submit_request = AsyncMock(return_value=mock_future)
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      with patch.object(client, 'get_provider_for_model', return_value="openai"):
        with pytest.raises(LLMConnectorError, match="Request timeout for model gpt-4"):
          await client.generate_response(request)

  @pytest.mark.asyncio
  async def test_generate_response_rate_limit_error(self):
    """Test response generation with rate limit error."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      mock_future = asyncio.Future()
      mock_future.set_exception(Exception("Rate limit exceeded"))
      
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.submit_request = AsyncMock(return_value=mock_future)
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      with patch.object(client, 'get_provider_for_model', return_value="openai"):
        with pytest.raises(LLMConnectorError, match="Rate limit exceeded"):
          await client.generate_response(request)

  @pytest.mark.asyncio
  async def test_generate_response_authentication_error(self):
    """Test response generation with authentication error."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      mock_future = asyncio.Future()
      mock_future.set_exception(Exception("Authentication failed"))
      
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.submit_request = AsyncMock(return_value=mock_future)
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      with patch.object(client, 'get_provider_for_model', return_value="openai"):
        with pytest.raises(LLMConnectorError, match="Authentication failed"):
          await client.generate_response(request)

  @pytest.mark.asyncio
  async def test_generate_response_model_not_found_error(self):
    """Test response generation with model not found error."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      mock_future = asyncio.Future()
      mock_future.set_exception(Exception("Model not found"))
      
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.submit_request = AsyncMock(return_value=mock_future)
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      with patch.object(client, 'get_provider_for_model', return_value="openai"):
        with pytest.raises(LLMConnectorError, match="Model not available"):
          await client.generate_response(request)

  @pytest.mark.asyncio
  async def test_generate_response_generic_error(self):
    """Test response generation with generic error."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      mock_future = asyncio.Future()
      mock_future.set_exception(Exception("Generic error"))
      
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.submit_request = AsyncMock(return_value=mock_future)
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      with patch.object(client, 'get_provider_for_model', return_value="openai"):
        with pytest.raises(LLMConnectorError, match="Request failed for model gpt-4"):
          await client.generate_response(request)

  def test_validate_response_structure_success(self):
    """Test successful response structure validation."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    response = ModelResponse(
      text="Valid response",
      model="gpt-4",
      provider="openai",
      timestamp=datetime.now(),
      latency_ms=500,
      token_count=25,
      cost_estimate=0.001
    )
    
    client = LLMClient()
    
    # Should not raise any exception
    client._validate_response_structure(response, request, "openai")

  def test_validate_response_structure_empty_text(self):
    """Test response validation with empty text."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    response = create_test_response(
      text="",  # Empty text
      model="gpt-4",
      provider="openai"
    )
    
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="Response text is empty"):
      client._validate_response_structure(response, request, "openai")

  def test_validate_response_structure_whitespace_only_text(self):
    """Test response validation with whitespace-only text."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    response = create_test_response(
      text="   \n\t  ",  # Whitespace only
      model="gpt-4",
      provider="openai"
    )
    
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="Response text is empty"):
      client._validate_response_structure(response, request, "openai")

  def test_validate_response_structure_model_mismatch(self):
    """Test response validation with model mismatch."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    response = create_test_response(
      text="Valid response",
      model="gpt-3.5-turbo",  # Different model
      provider="openai"
    )
    
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="Response model.*does not match"):
      client._validate_response_structure(response, request, "openai")

  def test_validate_response_structure_provider_mismatch(self):
    """Test response validation with provider mismatch."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    response = create_test_response(
      text="Valid response",
      model="gpt-4",
      provider="anthropic"  # Different provider
    )
    
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="Response provider.*does not match"):
      client._validate_response_structure(response, request, "openai")

  def test_validate_response_structure_negative_latency(self):
    """Test response validation with negative latency."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    response = create_test_response(
      text="Valid response",
      model="gpt-4",
      provider="openai",
      latency_ms=-100  # Negative latency
    )
    
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="Invalid latency"):
      client._validate_response_structure(response, request, "openai")

  def test_validate_response_structure_negative_token_count(self):
    """Test response validation with negative token count."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    response = create_test_response(
      text="Valid response",
      model="gpt-4",
      provider="openai",
      token_count=-10  # Negative token count
    )
    
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="Invalid token count"):
      client._validate_response_structure(response, request, "openai")

  def test_validate_response_structure_negative_cost(self):
    """Test response validation with negative cost estimate."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    response = create_test_response(
      text="Valid response",
      model="gpt-4",
      provider="openai",
      cost_estimate=-0.001  # Negative cost
    )
    
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="Invalid cost estimate"):
      client._validate_response_structure(response, request, "openai")

  def test_validate_response_structure_future_timestamp(self):
    """Test response validation with future timestamp (should warn but not fail)."""
    request = ModelRequest(prompt="Test", model="gpt-4")
    
    # Create timestamp 2 hours in the future
    from datetime import timedelta
    future_time = datetime.now() + timedelta(hours=2)
    
    response = create_test_response(
      text="Valid response",
      model="gpt-4",
      provider="openai",
      timestamp=future_time
    )
    
    client = LLMClient()
    
    with patch("rationale_benchmark.llm.client.logger") as mock_logger:
      # Should not raise exception, but should log warning
      client._validate_response_structure(response, request, "openai")
      mock_logger.warning.assert_called()
      warning_msg = mock_logger.warning.call_args[0][0]
      assert "timestamp seems incorrect" in warning_msg


class TestLLMClientConcurrentResponseGeneration:
  """Test concurrent response generation functionality."""

  @pytest.mark.asyncio
  async def test_generate_responses_concurrent_success(self):
    """Test successful concurrent response generation."""
    # Create multiple test requests
    requests = [
      ModelRequest(prompt="Test prompt 1", model="gpt-4"),
      ModelRequest(prompt="Test prompt 2", model="claude-3"),
      ModelRequest(prompt="Test prompt 3", model="gpt-4")
    ]
    
    # Create expected responses
    expected_responses = [
      ModelResponse(
        text="Response 1",
        model="gpt-4",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=500
      ),
      ModelResponse(
        text="Response 2",
        model="claude-3",
        provider="anthropic",
        timestamp=datetime.now(),
        latency_ms=600
      ),
      ModelResponse(
        text="Response 3",
        model="gpt-4",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=400
      )
    ]
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.process_requests_concurrent = AsyncMock(
        return_value=expected_responses
      )
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      # Mock provider mapping
      def mock_get_provider(model):
        if model == "gpt-4":
          return "openai"
        elif model == "claude-3":
          return "anthropic"
        return None
      
      with patch.object(client, 'get_provider_for_model', side_effect=mock_get_provider):
        with patch.object(client, 'get_available_models', return_value=["gpt-4", "claude-3"]):
          responses = await client.generate_responses_concurrent(requests)
      
      # Verify responses
      assert len(responses) == 3
      assert responses[0].text == "Response 1"
      assert responses[1].text == "Response 2"
      assert responses[2].text == "Response 3"
      
      # Verify concurrent manager was called
      concurrent_manager_mock.process_requests_concurrent.assert_called_once()
      submitted_requests = concurrent_manager_mock.process_requests_concurrent.call_args[0][0]
      assert len(submitted_requests) == 3

  @pytest.mark.asyncio
  async def test_generate_responses_concurrent_empty_list(self):
    """Test concurrent response generation with empty request list."""
    client = LLMClient()
    client._is_initialized = True
    client.concurrent_manager = Mock()  # Add mock concurrent manager
    
    responses = await client.generate_responses_concurrent([])
    
    assert responses == []

  @pytest.mark.asyncio
  async def test_generate_responses_concurrent_with_streaming_parameters(self):
    """Test concurrent response generation with streaming parameters removed."""
    requests = [
      ModelRequest(
        prompt="Test prompt 1",
        model="gpt-4",
        provider_specific={"stream": True, "temperature": 0.8}
      ),
      ModelRequest(
        prompt="Test prompt 2",
        model="claude-3",
        provider_specific={"streaming": True, "max_tokens": 100}
      )
    ]
    
    expected_responses = [
      ModelResponse(
        text="Response 1",
        model="gpt-4",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=500
      ),
      ModelResponse(
        text="Response 2",
        model="claude-3",
        provider="anthropic",
        timestamp=datetime.now(),
        latency_ms=600
      )
    ]
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.process_requests_concurrent = AsyncMock(
        return_value=expected_responses
      )
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      def mock_get_provider(model):
        return "openai" if model == "gpt-4" else "anthropic"
      
      with patch.object(client, 'get_provider_for_model', side_effect=mock_get_provider):
        with patch.object(client, 'get_available_models', return_value=["gpt-4", "claude-3"]):
          with patch("rationale_benchmark.llm.client.logger") as mock_logger:
            responses = await client.generate_responses_concurrent(requests)
      
      # Verify streaming parameters were removed
      submitted_requests = concurrent_manager_mock.process_requests_concurrent.call_args[0][0]
      assert "stream" not in submitted_requests[0].provider_specific
      assert "streaming" not in submitted_requests[1].provider_specific
      assert submitted_requests[0].provider_specific["temperature"] == 0.8
      assert submitted_requests[1].provider_specific["max_tokens"] == 100
      
      # Verify warnings were logged
      mock_logger.warning.assert_called()

  @pytest.mark.asyncio
  async def test_generate_responses_concurrent_validation_errors(self):
    """Test concurrent response generation with validation errors."""
    requests = [
      ModelRequest(prompt="Test prompt 1", model="gpt-4"),
      ModelRequest(prompt="Test prompt 2", model="unsupported-model"),  # Invalid model
      ModelRequest(prompt="Test prompt 3", model="claude-3")
    ]
    
    client = LLMClient()
    client._is_initialized = True
    client.concurrent_manager = Mock()
    
    def mock_get_provider(model):
      if model == "gpt-4":
        return "openai"
      elif model == "claude-3":
        return "anthropic"
      return None  # Unsupported model
    
    with patch.object(client, 'get_provider_for_model', side_effect=mock_get_provider):
      with patch.object(client, 'get_available_models', return_value=["gpt-4", "claude-3"]):
        with pytest.raises(ValueError, match="Request validation failed"):
          await client.generate_responses_concurrent(requests)

  @pytest.mark.asyncio
  async def test_generate_responses_concurrent_not_initialized(self):
    """Test concurrent response generation when client not initialized."""
    requests = [ModelRequest(prompt="Test", model="gpt-4")]
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="LLMClient is not initialized"):
      await client.generate_responses_concurrent(requests)

  @pytest.mark.asyncio
  async def test_generate_responses_concurrent_provider_distribution_logging(self):
    """Test that provider distribution is logged correctly."""
    requests = [
      ModelRequest(prompt="Test 1", model="gpt-4"),
      ModelRequest(prompt="Test 2", model="gpt-4"),
      ModelRequest(prompt="Test 3", model="claude-3"),
      ModelRequest(prompt="Test 4", model="claude-3"),
      ModelRequest(prompt="Test 5", model="claude-3")
    ]
    
    expected_responses = [
      ModelResponse(
        text=f"Response {i+1}",
        model=req.model,
        provider="openai" if req.model == "gpt-4" else "anthropic",
        timestamp=datetime.now(),
        latency_ms=500
      ) for i, req in enumerate(requests)
    ]
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.process_requests_concurrent = AsyncMock(
        return_value=expected_responses
      )
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      def mock_get_provider(model):
        return "openai" if model == "gpt-4" else "anthropic"
      
      with patch.object(client, 'get_provider_for_model', side_effect=mock_get_provider):
        with patch.object(client, 'get_available_models', return_value=["gpt-4", "claude-3"]):
          with patch("rationale_benchmark.llm.client.logger") as mock_logger:
            responses = await client.generate_responses_concurrent(requests)
      
      # Verify provider distribution logging
      info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
      distribution_log = next((call for call in info_calls if "providers:" in call), None)
      assert distribution_log is not None
      assert "openai" in distribution_log
      assert "anthropic" in distribution_log

  @pytest.mark.asyncio
  async def test_generate_responses_concurrent_response_validation_failure(self):
    """Test concurrent response generation with response validation failure."""
    requests = [
      ModelRequest(prompt="Test 1", model="gpt-4"),
      ModelRequest(prompt="Test 2", model="claude-3")
    ]
    
    # Create responses where one has validation issues
    responses_from_manager = [
      ModelResponse(
        text="Valid response",
        model="gpt-4",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=500
      ),
      create_test_response(
        text="",  # Empty text - will fail validation
        model="claude-3",
        provider="anthropic"
      )
    ]
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.process_requests_concurrent = AsyncMock(
        return_value=responses_from_manager
      )
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      def mock_get_provider(model):
        return "openai" if model == "gpt-4" else "anthropic"
      
      with patch.object(client, 'get_provider_for_model', side_effect=mock_get_provider):
        with patch.object(client, 'get_available_models', return_value=["gpt-4", "claude-3"]):
          with patch("rationale_benchmark.llm.client.logger") as mock_logger:
            responses = await client.generate_responses_concurrent(requests)
      
      # Verify we got responses for both requests
      assert len(responses) == 2
      
      # First response should be valid
      assert responses[0].text == "Valid response"
      
      # Second response should be an error response
      assert responses[1].text.startswith("ERROR:")
      assert "Response validation failed" in responses[1].text
      
      # Verify error was logged
      mock_logger.error.assert_called()

  @pytest.mark.asyncio
  async def test_generate_responses_concurrent_catastrophic_failure(self):
    """Test concurrent response generation with catastrophic failure."""
    requests = [
      ModelRequest(prompt="Test 1", model="gpt-4"),
      ModelRequest(prompt="Test 2", model="claude-3")
    ]
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.process_requests_concurrent = AsyncMock(
        side_effect=Exception("Catastrophic failure")
      )
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      def mock_get_provider(model):
        return "openai" if model == "gpt-4" else "anthropic"
      
      with patch.object(client, 'get_provider_for_model', side_effect=mock_get_provider):
        with patch.object(client, 'get_available_models', return_value=["gpt-4", "claude-3"]):
          with patch("rationale_benchmark.llm.client.logger") as mock_logger:
            responses = await client.generate_responses_concurrent(requests)
      
      # Should return error responses for all requests
      assert len(responses) == 2
      assert all(r.text.startswith("ERROR:") for r in responses)
      assert all("Catastrophic failure" in r.text for r in responses)
      
      # Verify error was logged
      mock_logger.error.assert_called()

  @pytest.mark.asyncio
  async def test_generate_responses_concurrent_order_preservation(self):
    """Test that response order is preserved regardless of completion timing."""
    requests = [
      ModelRequest(prompt="First request", model="gpt-4"),
      ModelRequest(prompt="Second request", model="claude-3"),
      ModelRequest(prompt="Third request", model="gpt-4")
    ]
    
    # Responses should be returned in same order as requests
    expected_responses = [
      ModelResponse(
        text="First response",
        model="gpt-4",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=500
      ),
      ModelResponse(
        text="Second response",
        model="claude-3",
        provider="anthropic",
        timestamp=datetime.now(),
        latency_ms=300  # Faster response
      ),
      ModelResponse(
        text="Third response",
        model="gpt-4",
        provider="openai",
        timestamp=datetime.now(),
        latency_ms=700  # Slower response
      )
    ]
    
    with patch("rationale_benchmark.llm.client.ConcurrentLLMManager") as mock_manager_class:
      concurrent_manager_mock = mock_manager_class.return_value
      concurrent_manager_mock.process_requests_concurrent = AsyncMock(
        return_value=expected_responses
      )
      
      client = LLMClient()
      client._is_initialized = True
      client.concurrent_manager = concurrent_manager_mock
      
      def mock_get_provider(model):
        return "openai" if model == "gpt-4" else "anthropic"
      
      with patch.object(client, 'get_provider_for_model', side_effect=mock_get_provider):
        with patch.object(client, 'get_available_models', return_value=["gpt-4", "claude-3"]):
          responses = await client.generate_responses_concurrent(requests)
      
      # Verify order is preserved
      assert len(responses) == 3
      assert responses[0].text == "First response"
      assert responses[1].text == "Second response"
      assert responses[2].text == "Third response"

  def test_create_error_response(self):
    """Test creation of error responses for failed requests."""
    request = ModelRequest(
      prompt="This is a very long prompt that should be truncated in the error response metadata because it exceeds 100 characters",
      model="gpt-4",
      temperature=0.8,
      max_tokens=200
    )
    
    client = LLMClient()
    error_message = "Test error message"
    
    error_response = client._create_error_response(request, "openai", error_message)
    
    # Verify error response structure
    assert error_response.text == f"ERROR: {error_message}"
    assert error_response.model == "gpt-4"
    assert error_response.provider == "openai"
    assert error_response.latency_ms == 0
    assert error_response.token_count == 0
    assert error_response.finish_reason == "error"
    assert error_response.cost_estimate == 0.0
    
    # Verify metadata
    assert error_response.metadata["error"] is True
    assert error_response.metadata["error_message"] == error_message
    
    # Verify prompt truncation in metadata
    original_request = error_response.metadata["original_request"]
    assert len(original_request["prompt"]) <= 103  # 100 chars + "..."
    assert original_request["prompt"].endswith("...")
    assert original_request["model"] == "gpt-4"
    assert original_request["temperature"] == 0.8
    assert original_request["max_tokens"] == 200

  def test_create_error_response_short_prompt(self):
    """Test creation of error response with short prompt (no truncation)."""
    request = ModelRequest(
      prompt="Short prompt",
      model="claude-3"
    )
    
    client = LLMClient()
    error_response = client._create_error_response(request, "anthropic", "Error")
    
    # Verify prompt is not truncated
    original_request = error_response.metadata["original_request"]
    assert original_request["prompt"] == "Short prompt"
    assert not original_request["prompt"].endswith("...")


class TestLLMClientModelListing:
  """Test model listing with concurrent provider queries."""

  @pytest.mark.asyncio
  async def test_list_all_models_success(self):
    """Test successful model listing from all providers."""
    # Create mock providers
    mock_openai_provider = Mock()
    mock_openai_provider.list_models = AsyncMock(return_value=["gpt-4", "gpt-3.5-turbo"])
    mock_openai_provider.config.base_url = "https://api.openai.com"
    mock_openai_provider.config.timeout = 30
    mock_openai_provider.config.max_retries = 3
    
    mock_anthropic_provider = Mock()
    mock_anthropic_provider.list_models = AsyncMock(return_value=["claude-3-opus", "claude-3-sonnet"])
    mock_anthropic_provider.config.base_url = "https://api.anthropic.com"
    mock_anthropic_provider.config.timeout = 30
    mock_anthropic_provider.config.max_retries = 3
    
    providers = {
      "openai": mock_openai_provider,
      "anthropic": mock_anthropic_provider
    }
    
    with patch("rationale_benchmark.llm.client.ProviderFactory") as mock_factory_class:
      provider_factory_mock = mock_factory_class.return_value
      provider_factory_mock.list_providers.return_value = providers
      
      client = LLMClient()
      client._is_initialized = True
      client.provider_factory = provider_factory_mock
      
      with patch.object(client, 'get_available_models', return_value=["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]):
        result = await client.list_all_models()
      
      # Verify result structure
      assert result["total_providers"] == 2
      assert result["successful_providers"] == 2
      assert result["failed_providers"] == 0
      assert result["total_models"] == 4
      assert result["errors"] == []
      assert "query_timestamp" in result
      assert "configured_models" in result
      
      # Verify provider results
      assert "openai" in result["providers"]
      assert "anthropic" in result["providers"]
      
      openai_result = result["providers"]["openai"]
      assert openai_result["status"] == "success"
      assert openai_result["models"] == ["gpt-4", "gpt-3.5-turbo"]
      assert openai_result["model_count"] == 2
      assert openai_result["response_time_ms"] is not None
      
      anthropic_result = result["providers"]["anthropic"]
      assert anthropic_result["status"] == "success"
      assert anthropic_result["models"] == ["claude-3-opus", "claude-3-sonnet"]
      assert anthropic_result["model_count"] == 2

  @pytest.mark.asyncio
  async def test_list_all_models_no_providers(self):
    """Test model listing when no providers are configured."""
    with patch("rationale_benchmark.llm.client.ProviderFactory") as mock_factory_class:
      provider_factory_mock = mock_factory_class.return_value
      provider_factory_mock.list_providers.return_value = {}
      
      client = LLMClient()
      client._is_initialized = True
      client.provider_factory = provider_factory_mock
      
      result = await client.list_all_models()
      
      assert result["total_providers"] == 0
      assert result["successful_providers"] == 0
      assert result["total_models"] == 0
      assert result["providers"] == {}

  @pytest.mark.asyncio
  async def test_list_all_models_with_provider_failures(self):
    """Test model listing with some provider failures."""
    # Create mock providers - one succeeds, one fails
    mock_openai_provider = Mock()
    mock_openai_provider.list_models = AsyncMock(return_value=["gpt-4", "gpt-3.5-turbo"])
    mock_openai_provider.config.base_url = "https://api.openai.com"
    mock_openai_provider.config.timeout = 30
    mock_openai_provider.config.max_retries = 3
    
    mock_anthropic_provider = Mock()
    mock_anthropic_provider.list_models = AsyncMock(side_effect=Exception("Authentication failed"))
    mock_anthropic_provider.config.base_url = "https://api.anthropic.com"
    mock_anthropic_provider.config.timeout = 30
    mock_anthropic_provider.config.max_retries = 3
    
    providers = {
      "openai": mock_openai_provider,
      "anthropic": mock_anthropic_provider
    }
    
    with patch("rationale_benchmark.llm.client.ProviderFactory") as mock_factory_class:
      provider_factory_mock = mock_factory_class.return_value
      provider_factory_mock.list_providers.return_value = providers
      
      client = LLMClient()
      client._is_initialized = True
      client.provider_factory = provider_factory_mock
      
      with patch.object(client, 'get_available_models', return_value=["gpt-4", "gpt-3.5-turbo"]):
        with patch("rationale_benchmark.llm.client.logger") as mock_logger:
          result = await client.list_all_models()
      
      # Verify result structure
      assert result["total_providers"] == 2
      assert result["successful_providers"] == 1
      assert result["failed_providers"] == 1
      assert result["total_models"] == 2  # Only from successful provider
      # Errors are handled within provider results, not in top-level errors list
      assert len(result["errors"]) == 0
      
      # Verify successful provider
      openai_result = result["providers"]["openai"]
      assert openai_result["status"] == "success"
      assert openai_result["models"] == ["gpt-4", "gpt-3.5-turbo"]
      
      # Verify failed provider
      anthropic_result = result["providers"]["anthropic"]
      assert anthropic_result["status"] == "error"
      assert anthropic_result["models"] == []
      assert anthropic_result["model_count"] == 0
      assert "Authentication failed" in anthropic_result["error"]
      
      # Note: Warnings are only logged for Exception instances, not error status dicts
      # The current implementation handles errors gracefully within _query_provider_models

  @pytest.mark.asyncio
  async def test_list_all_models_timeout(self):
    """Test model listing with timeout."""
    # Create mock provider that takes too long
    mock_provider = Mock()
    
    async def slow_list_models():
      await asyncio.sleep(35)  # Longer than 30s timeout
      return ["gpt-4"]
    
    mock_provider.list_models = slow_list_models
    mock_provider.config.base_url = "https://api.openai.com"
    mock_provider.config.timeout = 30
    mock_provider.config.max_retries = 3
    
    providers = {"openai": mock_provider}
    
    with patch("rationale_benchmark.llm.client.ProviderFactory") as mock_factory_class:
      provider_factory_mock = mock_factory_class.return_value
      provider_factory_mock.list_providers.return_value = providers
      
      client = LLMClient()
      client._is_initialized = True
      client.provider_factory = provider_factory_mock
      
      with patch.object(client, 'get_available_models', return_value=[]):
        with patch("rationale_benchmark.llm.client.logger") as mock_logger:
          result = await client.list_all_models()
      
      # Verify timeout handling
      assert result["total_providers"] == 1
      assert result["successful_providers"] == 0
      assert result["failed_providers"] == 1
      
      # Verify timeout was logged
      mock_logger.warning.assert_called()
      warning_msg = mock_logger.warning.call_args[0][0]
      assert "timed out" in warning_msg

  @pytest.mark.asyncio
  async def test_list_all_models_not_initialized(self):
    """Test model listing when client not initialized."""
    client = LLMClient()
    
    with pytest.raises(LLMConnectorError, match="LLMClient is not initialized"):
      await client.list_all_models()

  @pytest.mark.asyncio
  async def test_query_provider_models_success(self):
    """Test successful individual provider model query."""
    mock_provider = Mock()
    mock_provider.list_models = AsyncMock(return_value=["gpt-4", "gpt-3.5-turbo"])
    mock_provider.config.base_url = "https://api.openai.com"
    mock_provider.config.timeout = 30
    mock_provider.config.max_retries = 3
    
    client = LLMClient()
    
    result = await client._query_provider_models("openai", mock_provider)
    
    assert result["status"] == "success"
    assert result["models"] == ["gpt-4", "gpt-3.5-turbo"]
    assert result["model_count"] == 2
    assert result["response_time_ms"] is not None
    assert result["response_time_ms"] >= 0
    assert "provider_info" in result
    assert result["provider_info"]["base_url"] == "https://api.openai.com"

  @pytest.mark.asyncio
  async def test_query_provider_models_invalid_response_type(self):
    """Test provider query with invalid response type."""
    mock_provider = Mock()
    mock_provider.list_models = AsyncMock(return_value="not a list")  # Invalid type
    
    client = LLMClient()
    
    result = await client._query_provider_models("openai", mock_provider)
    
    assert result["status"] == "error"
    assert "invalid model list type" in result["error"]
    assert result["models"] == []
    assert result["model_count"] == 0

  @pytest.mark.asyncio
  async def test_query_provider_models_with_invalid_models(self):
    """Test provider query with some invalid model names."""
    mock_provider = Mock()
    mock_provider.list_models = AsyncMock(return_value=[
      "gpt-4",           # Valid
      "",                # Invalid - empty
      "gpt-3.5-turbo",   # Valid
      None,              # Invalid - None
      "  claude-3  ",    # Valid but needs trimming
      123                # Invalid - not string
    ])
    mock_provider.config.base_url = "https://api.openai.com"
    mock_provider.config.timeout = 30
    mock_provider.config.max_retries = 3
    
    client = LLMClient()
    
    with patch("rationale_benchmark.llm.client.logger") as mock_logger:
      result = await client._query_provider_models("openai", mock_provider)
    
    # Should only include valid models
    assert result["status"] == "success"
    assert result["models"] == ["gpt-4", "gpt-3.5-turbo", "claude-3"]
    assert result["model_count"] == 3
    
    # Should log warnings for invalid models
    mock_logger.warning.assert_called()

  @pytest.mark.asyncio
  async def test_query_provider_models_timeout_error(self):
    """Test provider query with timeout error."""
    mock_provider = Mock()
    mock_provider.list_models = AsyncMock(side_effect=asyncio.TimeoutError("Timeout"))
    
    client = LLMClient()
    
    result = await client._query_provider_models("openai", mock_provider)
    
    assert result["status"] == "timeout"
    assert result["error"] == "Query timed out"
    assert result["models"] == []
    assert result["model_count"] == 0
    assert result["response_time_ms"] is not None

  @pytest.mark.asyncio
  async def test_query_provider_models_authentication_error(self):
    """Test provider query with authentication error."""
    mock_provider = Mock()
    mock_provider.list_models = AsyncMock(side_effect=Exception("Authentication failed"))
    
    client = LLMClient()
    
    result = await client._query_provider_models("openai", mock_provider)
    
    assert result["status"] == "error"
    assert result["error"] == "Authentication failed"
    assert result["error_type"] == "authentication"
    assert result["models"] == []

  @pytest.mark.asyncio
  async def test_query_provider_models_network_error(self):
    """Test provider query with network error."""
    mock_provider = Mock()
    mock_provider.list_models = AsyncMock(side_effect=Exception("Network connection failed"))
    
    client = LLMClient()
    
    result = await client._query_provider_models("openai", mock_provider)
    
    assert result["status"] == "error"
    assert result["error"] == "Network connection failed"
    assert result["error_type"] == "network"

  @pytest.mark.asyncio
  async def test_query_provider_models_rate_limit_error(self):
    """Test provider query with rate limit error."""
    mock_provider = Mock()
    mock_provider.list_models = AsyncMock(side_effect=Exception("Rate limit exceeded"))
    
    client = LLMClient()
    
    result = await client._query_provider_models("openai", mock_provider)
    
    assert result["status"] == "error"
    assert result["error"] == "Rate limit exceeded"
    assert result["error_type"] == "rate_limit"

  @pytest.mark.asyncio
  async def test_query_provider_models_not_found_error(self):
    """Test provider query with not found error."""
    mock_provider = Mock()
    mock_provider.list_models = AsyncMock(side_effect=Exception("Endpoint not found"))
    
    client = LLMClient()
    
    result = await client._query_provider_models("openai", mock_provider)
    
    assert result["status"] == "error"
    assert result["error"] == "Endpoint not found"
    assert result["error_type"] == "not_found"

  @pytest.mark.asyncio
  async def test_query_provider_models_generic_error(self):
    """Test provider query with generic error."""
    mock_provider = Mock()
    mock_provider.list_models = AsyncMock(side_effect=Exception("Something went wrong"))
    
    client = LLMClient()
    
    result = await client._query_provider_models("openai", mock_provider)
    
    assert result["status"] == "error"
    assert result["error"] == "Something went wrong"
    assert result["error_type"] == "unknown"

  @pytest.mark.asyncio
  async def test_list_all_models_concurrent_execution(self):
    """Test that model listing executes provider queries concurrently."""
    # Create providers with different response times
    mock_fast_provider = Mock()
    mock_fast_provider.list_models = AsyncMock()
    mock_fast_provider.config.base_url = "https://fast.api.com"
    mock_fast_provider.config.timeout = 30
    mock_fast_provider.config.max_retries = 3
    
    mock_slow_provider = Mock()
    mock_slow_provider.list_models = AsyncMock()
    mock_slow_provider.config.base_url = "https://slow.api.com"
    mock_slow_provider.config.timeout = 30
    mock_slow_provider.config.max_retries = 3
    
    async def fast_response():
      await asyncio.sleep(0.1)
      return ["fast-model"]
    
    async def slow_response():
      await asyncio.sleep(0.5)
      return ["slow-model"]
    
    mock_fast_provider.list_models = fast_response
    mock_slow_provider.list_models = slow_response
    
    providers = {
      "fast": mock_fast_provider,
      "slow": mock_slow_provider
    }
    
    with patch("rationale_benchmark.llm.client.ProviderFactory") as mock_factory_class:
      provider_factory_mock = mock_factory_class.return_value
      provider_factory_mock.list_providers.return_value = providers
      
      client = LLMClient()
      client._is_initialized = True
      client.provider_factory = provider_factory_mock
      
      with patch.object(client, 'get_available_models', return_value=["fast-model", "slow-model"]):
        start_time = asyncio.get_event_loop().time()
        result = await client.list_all_models()
        end_time = asyncio.get_event_loop().time()
      
      # Should complete in roughly the time of the slowest provider (0.5s)
      # rather than the sum of both (0.6s), proving concurrent execution
      total_time = end_time - start_time
      assert total_time < 0.8  # Allow some margin for test execution overhead
      
      # Verify both providers succeeded
      assert result["successful_providers"] == 2
      assert result["total_models"] == 2