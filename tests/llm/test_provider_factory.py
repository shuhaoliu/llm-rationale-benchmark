"""Unit tests for ProviderFactory class."""

import pytest
from unittest.mock import Mock, patch

from rationale_benchmark.llm.models import LLMConfig, ProviderConfig
from rationale_benchmark.llm.exceptions import (
  ConfigurationError,
  ProviderError,
)
from rationale_benchmark.llm.factory import ProviderFactory
from rationale_benchmark.llm.http.client import HTTPClient
from rationale_benchmark.llm.providers.base import LLMProvider
from rationale_benchmark.llm.providers.openai import OpenAIProvider
from rationale_benchmark.llm.providers.anthropic import AnthropicProvider
from rationale_benchmark.llm.providers.gemini import GeminiProvider
from rationale_benchmark.llm.providers.openrouter import OpenRouterProvider


class TestProviderFactory:
  """Test cases for ProviderFactory class."""

  @pytest.fixture
  def mock_http_client(self):
    """Create a mock HTTP client for testing."""
    return Mock(spec=HTTPClient)

  @pytest.fixture
  def sample_provider_config(self):
    """Create a sample provider configuration for testing."""
    return ProviderConfig(
      name="openai",
      api_key="sk-test1234567890abcdef1234567890abcdef1234567890abcdef",
      base_url="https://api.openai.com/v1",
      timeout=30,
      max_retries=3,
      models=["gpt-4", "gpt-3.5-turbo"],
      default_params={"temperature": 0.7, "max_tokens": 1000},
      provider_specific={"organization": "test-org"}
    )

  @pytest.fixture
  def sample_llm_config(self, sample_provider_config):
    """Create a sample LLM configuration with multiple providers."""
    anthropic_config = ProviderConfig(
      name="anthropic",
      api_key="sk-ant-api03-test1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
      models=["claude-3-opus", "claude-3-sonnet"],
      default_params={"temperature": 0.5, "max_tokens": 2000}
    )
    
    gemini_config = ProviderConfig(
      name="gemini",
      api_key="AIzaSyTest1234567890abcdef1234567890abc",
      models=["gemini-pro", "gemini-pro-vision"],
      default_params={"temperature": 0.8}
    )
    
    return LLMConfig(
      defaults={"temperature": 0.7, "max_tokens": 1000},
      providers={
        "openai": sample_provider_config,
        "anthropic": anthropic_config,
        "gemini": gemini_config
      }
    )

  def test_provider_factory_initialization(self, mock_http_client):
    """Test ProviderFactory initialization with HTTP client."""
    factory = ProviderFactory(mock_http_client)
    
    assert factory.http_client is mock_http_client
    assert factory._providers == {}
    assert factory._model_to_provider == {}
    assert len(factory._provider_registry) > 0  # Should have built-in providers

  def test_provider_factory_has_built_in_providers(self, mock_http_client):
    """Test that ProviderFactory has built-in provider types registered."""
    factory = ProviderFactory(mock_http_client)
    
    # Check that built-in providers are registered
    assert "openai" in factory._provider_registry
    assert "anthropic" in factory._provider_registry
    assert "gemini" in factory._provider_registry
    assert "openrouter" in factory._provider_registry
    
    # Verify the registered classes
    assert factory._provider_registry["openai"] == OpenAIProvider
    assert factory._provider_registry["anthropic"] == AnthropicProvider
    assert factory._provider_registry["gemini"] == GeminiProvider
    assert factory._provider_registry["openrouter"] == OpenRouterProvider

  def test_register_provider_success(self, mock_http_client):
    """Test successful provider registration."""
    factory = ProviderFactory(mock_http_client)
    
    class CustomProvider(LLMProvider):
      pass
    
    factory.register_provider("custom", CustomProvider)
    
    assert "custom" in factory._provider_registry
    assert factory._provider_registry["custom"] == CustomProvider

  def test_register_provider_invalid_class(self, mock_http_client):
    """Test provider registration with invalid provider class."""
    factory = ProviderFactory(mock_http_client)
    
    class NotAProvider:
      pass
    
    with pytest.raises(ConfigurationError, match="must be a subclass of LLMProvider"):
      factory.register_provider("invalid", NotAProvider)

  def test_register_provider_duplicate_name(self, mock_http_client):
    """Test provider registration with duplicate name."""
    factory = ProviderFactory(mock_http_client)
    
    class CustomProvider(LLMProvider):
      pass
    
    # First registration should succeed
    factory.register_provider("custom", CustomProvider)
    
    # Second registration with same name should raise error
    with pytest.raises(ConfigurationError, match="Provider 'custom' is already registered"):
      factory.register_provider("custom", CustomProvider)

  def test_create_provider_success(self, mock_http_client, sample_provider_config):
    """Test successful provider creation."""
    factory = ProviderFactory(mock_http_client)
    
    provider = factory.create_provider(sample_provider_config)
    
    assert isinstance(provider, OpenAIProvider)
    assert provider.config == sample_provider_config
    assert provider.http_client == mock_http_client

  def test_create_provider_unknown_type(self, mock_http_client):
    """Test provider creation with unknown provider type."""
    factory = ProviderFactory(mock_http_client)
    
    unknown_config = ProviderConfig(
      name="unknown",
      api_key="test-key"
    )
    
    with pytest.raises(ConfigurationError, match="Unknown provider type: unknown"):
      factory.create_provider(unknown_config)

  def test_create_provider_instantiation_error(self, mock_http_client):
    """Test provider creation when instantiation fails."""
    factory = ProviderFactory(mock_http_client)
    
    # Replace the provider class in the registry with a mock that raises an exception
    original_provider = factory._provider_registry["openai"]
    mock_provider_class = Mock(side_effect=Exception("Instantiation failed"))
    factory._provider_registry["openai"] = mock_provider_class
    
    try:
      config = ProviderConfig(name="openai", api_key="sk-test1234567890abcdef1234567890abcdef1234567890abcdef")
      
      with pytest.raises(ProviderError, match="Failed to create provider 'openai'"):
        factory.create_provider(config)
    finally:
      # Restore the original provider
      factory._provider_registry["openai"] = original_provider

  def test_initialize_providers_success(self, mock_http_client, sample_llm_config):
    """Test successful initialization of all providers from configuration."""
    factory = ProviderFactory(mock_http_client)
    
    factory.initialize_providers(sample_llm_config)
    
    # Check that all providers were created
    assert len(factory._providers) == 3
    assert "openai" in factory._providers
    assert "anthropic" in factory._providers
    assert "gemini" in factory._providers
    
    # Check provider types
    assert isinstance(factory._providers["openai"], OpenAIProvider)
    assert isinstance(factory._providers["anthropic"], AnthropicProvider)
    assert isinstance(factory._providers["gemini"], GeminiProvider)

  def test_initialize_providers_builds_model_mapping(self, mock_http_client, sample_llm_config):
    """Test that provider initialization builds correct model-to-provider mapping."""
    factory = ProviderFactory(mock_http_client)
    
    factory.initialize_providers(sample_llm_config)
    
    # Check model-to-provider mapping
    expected_mappings = {
      "gpt-4": "openai",
      "gpt-3.5-turbo": "openai",
      "claude-3-opus": "anthropic",
      "claude-3-sonnet": "anthropic",
      "gemini-pro": "gemini",
      "gemini-pro-vision": "gemini"
    }
    
    assert factory._model_to_provider == expected_mappings

  def test_initialize_providers_duplicate_models(self, mock_http_client):
    """Test provider initialization with duplicate model names across providers."""
    # Create config with duplicate model names
    config1 = ProviderConfig(
      name="openai",
      api_key="sk-test1234567890abcdef1234567890abcdef1234567890abcdef",
      models=["gpt-4", "shared-model"]
    )
    config2 = ProviderConfig(
      name="anthropic",
      api_key="sk-ant-api03-test1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
      models=["claude-3-opus", "shared-model"]
    )
    
    llm_config = LLMConfig(
      defaults={},
      providers={"openai": config1, "anthropic": config2}
    )
    
    factory = ProviderFactory(mock_http_client)
    
    with pytest.raises(ConfigurationError, match="Model 'shared-model' is configured for multiple providers"):
      factory.initialize_providers(llm_config)

  def test_initialize_providers_validation_error(self, mock_http_client, sample_llm_config):
    """Test provider initialization when provider validation fails."""
    factory = ProviderFactory(mock_http_client)
    
    # Mock provider validation to return errors
    with patch.object(OpenAIProvider, 'validate_config') as mock_validate:
      mock_validate.return_value = ["Invalid API key format"]
      
      with pytest.raises(ConfigurationError, match="Provider 'openai': Invalid API key format"):
        factory.initialize_providers(sample_llm_config)

  def test_get_provider_success(self, mock_http_client, sample_llm_config):
    """Test successful provider retrieval by name."""
    factory = ProviderFactory(mock_http_client)
    factory.initialize_providers(sample_llm_config)
    
    provider = factory.get_provider("openai")
    
    assert isinstance(provider, OpenAIProvider)
    assert provider.config.name == "openai"

  def test_get_provider_not_found(self, mock_http_client):
    """Test provider retrieval with non-existent provider name."""
    factory = ProviderFactory(mock_http_client)
    
    with pytest.raises(ProviderError, match="Provider 'nonexistent' not found"):
      factory.get_provider("nonexistent")

  def test_get_provider_for_model_success(self, mock_http_client, sample_llm_config):
    """Test successful provider retrieval by model name."""
    factory = ProviderFactory(mock_http_client)
    factory.initialize_providers(sample_llm_config)
    
    provider = factory.get_provider_for_model("gpt-4")
    
    assert isinstance(provider, OpenAIProvider)
    assert provider.config.name == "openai"

  def test_get_provider_for_model_not_found(self, mock_http_client, sample_llm_config):
    """Test provider retrieval with non-existent model name."""
    factory = ProviderFactory(mock_http_client)
    factory.initialize_providers(sample_llm_config)
    
    with pytest.raises(ProviderError, match="No provider found for model 'unknown-model'"):
      factory.get_provider_for_model("unknown-model")

  def test_list_providers(self, mock_http_client, sample_llm_config):
    """Test listing all initialized providers."""
    factory = ProviderFactory(mock_http_client)
    factory.initialize_providers(sample_llm_config)
    
    providers = factory.list_providers()
    
    assert len(providers) == 3
    assert "openai" in providers
    assert "anthropic" in providers
    assert "gemini" in providers
    
    # Check that returned providers are the actual instances
    assert isinstance(providers["openai"], OpenAIProvider)
    assert isinstance(providers["anthropic"], AnthropicProvider)
    assert isinstance(providers["gemini"], GeminiProvider)

  def test_list_models(self, mock_http_client, sample_llm_config):
    """Test listing all available models across providers."""
    factory = ProviderFactory(mock_http_client)
    factory.initialize_providers(sample_llm_config)
    
    models = factory.list_models()
    
    expected_models = {
      "gpt-4", "gpt-3.5-turbo",
      "claude-3-opus", "claude-3-sonnet",
      "gemini-pro", "gemini-pro-vision"
    }
    
    assert set(models) == expected_models

  def test_get_model_to_provider_mapping(self, mock_http_client, sample_llm_config):
    """Test getting the complete model-to-provider mapping."""
    factory = ProviderFactory(mock_http_client)
    factory.initialize_providers(sample_llm_config)
    
    mapping = factory.get_model_to_provider_mapping()
    
    expected_mapping = {
      "gpt-4": "openai",
      "gpt-3.5-turbo": "openai",
      "claude-3-opus": "anthropic",
      "claude-3-sonnet": "anthropic",
      "gemini-pro": "gemini",
      "gemini-pro-vision": "gemini"
    }
    
    assert mapping == expected_mapping

  def test_validate_all_providers_success(self, mock_http_client, sample_llm_config):
    """Test successful validation of all providers."""
    factory = ProviderFactory(mock_http_client)
    factory.initialize_providers(sample_llm_config)
    
    # Mock all providers to return no validation errors
    for provider in factory._providers.values():
      provider.validate_config = Mock(return_value=[])
    
    errors = factory.validate_all_providers()
    
    assert errors == []

  def test_validate_all_providers_with_errors(self, mock_http_client, sample_llm_config):
    """Test validation when some providers have errors."""
    factory = ProviderFactory(mock_http_client)
    factory.initialize_providers(sample_llm_config)
    
    # Mock providers to return validation errors
    factory._providers["openai"].validate_config = Mock(return_value=["OpenAI error"])
    factory._providers["anthropic"].validate_config = Mock(return_value=[])
    factory._providers["gemini"].validate_config = Mock(return_value=["Gemini error 1", "Gemini error 2"])
    
    errors = factory.validate_all_providers()
    
    expected_errors = [
      "Provider 'openai': OpenAI error",
      "Provider 'gemini': Gemini error 1",
      "Provider 'gemini': Gemini error 2"
    ]
    
    assert errors == expected_errors

  def test_provider_factory_empty_initialization(self, mock_http_client):
    """Test ProviderFactory behavior before provider initialization."""
    factory = ProviderFactory(mock_http_client)
    
    # Should have empty providers but registry should be populated
    assert factory._providers == {}
    assert factory._model_to_provider == {}
    assert len(factory._provider_registry) > 0
    
    # Operations on uninitialized factory should raise appropriate errors
    with pytest.raises(ProviderError):
      factory.get_provider("openai")
    
    with pytest.raises(ProviderError):
      factory.get_provider_for_model("gpt-4")
    
    # List operations should return empty results
    assert factory.list_providers() == {}
    assert factory.list_models() == []
    assert factory.get_model_to_provider_mapping() == {}

  def test_provider_factory_custom_provider_integration(self, mock_http_client):
    """Test integration of custom provider through registration and initialization."""
    factory = ProviderFactory(mock_http_client)
    
    # Create a custom provider class
    class CustomProvider(LLMProvider):
      def validate_config(self):
        return []
      
      async def generate_response(self, request):
        pass
      
      async def list_models(self):
        return ["custom-model-1", "custom-model-2"]
      
      def _prepare_request(self, request):
        return {}
      
      def _parse_response(self, response_data, request, latency_ms):
        pass
    
    # Register the custom provider
    factory.register_provider("custom", CustomProvider)
    
    # Create configuration with custom provider
    custom_config = ProviderConfig(
      name="custom",
      api_key="custom-key",
      models=["custom-model-1", "custom-model-2"]
    )
    
    llm_config = LLMConfig(
      defaults={},
      providers={"custom": custom_config}
    )
    
    # Initialize providers
    factory.initialize_providers(llm_config)
    
    # Verify custom provider is available
    assert "custom" in factory.list_providers()
    assert isinstance(factory.get_provider("custom"), CustomProvider)
    assert factory.get_provider_for_model("custom-model-1").config.name == "custom"
    
    # Verify models are mapped correctly
    models = factory.list_models()
    assert "custom-model-1" in models
    assert "custom-model-2" in models
    
    mapping = factory.get_model_to_provider_mapping()
    assert mapping["custom-model-1"] == "custom"
    assert mapping["custom-model-2"] == "custom"