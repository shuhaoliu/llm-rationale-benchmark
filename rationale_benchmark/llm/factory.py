"""Provider factory for creating and managing LLM provider instances."""

import logging
from typing import Dict, List, Type

from .config.models import LLMConfig, ProviderConfig
from .exceptions import ConfigurationError, ProviderError
from .http.client import HTTPClient
from .providers.base import LLMProvider
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .providers.openai import OpenAIProvider
from .providers.openrouter import OpenRouterProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
  """Factory class for creating and managing LLM provider instances.
  
  This factory provides a centralized way to create provider instances based on
  configuration, register new provider types, and manage the mapping between
  models and their corresponding providers for request routing.
  """

  def __init__(self, http_client: HTTPClient):
    """Initialize the provider factory with an HTTP client.
    
    Args:
      http_client: HTTP client instance to be shared across all providers
    """
    self.http_client = http_client
    self._providers: Dict[str, LLMProvider] = {}
    self._model_to_provider: Dict[str, str] = {}
    
    # Registry of available provider types
    self._provider_registry: Dict[str, Type[LLMProvider]] = {
      "openai": OpenAIProvider,
      "anthropic": AnthropicProvider,
      "gemini": GeminiProvider,
      "openrouter": OpenRouterProvider,
    }
    
    logger.debug(f"ProviderFactory initialized with {len(self._provider_registry)} built-in provider types")

  def register_provider(self, provider_type: str, provider_class: Type[LLMProvider]) -> None:
    """Register a new provider type for easy extension.
    
    This method allows registration of custom provider implementations that
    extend the LLMProvider base class.
    
    Args:
      provider_type: Unique identifier for the provider type
      provider_class: Provider class that implements LLMProvider interface
      
    Raises:
      ConfigurationError: If provider_class is not a valid LLMProvider subclass
                         or if provider_type is already registered
    """
    if not issubclass(provider_class, LLMProvider):
      raise ConfigurationError(
        f"Provider class for '{provider_type}' must be a subclass of LLMProvider",
        field=f"provider_registry.{provider_type}"
      )
    
    if provider_type in self._provider_registry:
      raise ConfigurationError(
        f"Provider '{provider_type}' is already registered",
        field=f"provider_registry.{provider_type}"
      )
    
    self._provider_registry[provider_type] = provider_class
    logger.info(f"Registered custom provider type: {provider_type}")

  def create_provider(self, config: ProviderConfig) -> LLMProvider:
    """Create a provider instance based on configuration.
    
    Args:
      config: Provider configuration containing type and settings
      
    Returns:
      Initialized provider instance
      
    Raises:
      ConfigurationError: If provider type is unknown
      ProviderError: If provider instantiation fails
    """
    provider_type = config.name
    
    if provider_type not in self._provider_registry:
      available_types = list(self._provider_registry.keys())
      raise ConfigurationError(
        f"Unknown provider type: {provider_type}. Available types: {available_types}",
        field=f"providers.{provider_type}"
      )
    
    provider_class = self._provider_registry[provider_type]
    
    try:
      provider = provider_class(config, self.http_client)
      logger.debug(f"Created provider instance: {provider_type}")
      return provider
    except Exception as e:
      raise ProviderError(
        provider_type,
        f"Failed to create provider '{provider_type}': {str(e)}",
        cause=e
      )

  def initialize_providers(self, config: LLMConfig) -> None:
    """Initialize all providers from configuration and build model mapping.
    
    This method creates provider instances for all configured providers,
    validates their configurations, and builds the model-to-provider mapping
    for request routing.
    
    Args:
      config: Complete LLM configuration with all providers
      
    Raises:
      ConfigurationError: If provider validation fails or duplicate models found
      ProviderError: If provider creation fails
    """
    logger.info(f"Initializing {len(config.providers)} providers")
    
    # Clear existing providers and mappings
    self._providers.clear()
    self._model_to_provider.clear()
    
    # Create all provider instances
    for provider_name, provider_config in config.providers.items():
      try:
        provider = self.create_provider(provider_config)
        self._providers[provider_name] = provider
        logger.debug(f"Initialized provider: {provider_name}")
      except Exception as e:
        logger.error(f"Failed to initialize provider '{provider_name}': {e}")
        raise
    
    # Validate all providers
    validation_errors = []
    for provider_name, provider in self._providers.items():
      try:
        errors = provider.validate_config()
        if errors:
          validation_errors.extend([f"Provider '{provider_name}': {error}" for error in errors])
      except Exception as e:
        validation_errors.append(f"Provider '{provider_name}' validation failed: {str(e)}")
    
    if validation_errors:
      raise ConfigurationError(
        f"Provider validation failed:\n" + "\n".join(validation_errors)
      )
    
    # Build model-to-provider mapping
    self._build_model_mapping(config)
    
    logger.info(f"Successfully initialized {len(self._providers)} providers with {len(self._model_to_provider)} models")

  def _build_model_mapping(self, config: LLMConfig) -> None:
    """Build mapping from model names to provider names for request routing.
    
    Args:
      config: LLM configuration containing provider model lists
      
    Raises:
      ConfigurationError: If duplicate model names are found across providers
    """
    model_conflicts = {}
    
    for provider_name, provider_config in config.providers.items():
      for model in provider_config.models:
        if model in self._model_to_provider:
          # Track conflicts for detailed error reporting
          if model not in model_conflicts:
            model_conflicts[model] = [self._model_to_provider[model]]
          model_conflicts[model].append(provider_name)
        else:
          self._model_to_provider[model] = provider_name
    
    if model_conflicts:
      conflict_details = []
      for model, providers in model_conflicts.items():
        conflict_details.append(f"Model '{model}' is configured for multiple providers: {providers}")
      
      raise ConfigurationError(
        f"Duplicate model names found across providers:\n" + "\n".join(conflict_details)
      )
    
    logger.debug(f"Built model mapping for {len(self._model_to_provider)} models")

  def get_provider(self, provider_name: str) -> LLMProvider:
    """Get provider instance by name.
    
    Args:
      provider_name: Name of the provider to retrieve
      
    Returns:
      Provider instance
      
    Raises:
      ProviderError: If provider is not found
    """
    if provider_name not in self._providers:
      available_providers = list(self._providers.keys())
      raise ProviderError(
        provider_name,
        f"Provider '{provider_name}' not found. Available providers: {available_providers}"
      )
    
    return self._providers[provider_name]

  def get_provider_for_model(self, model: str) -> LLMProvider:
    """Get provider instance that handles the specified model.
    
    Args:
      model: Model name to find provider for
      
    Returns:
      Provider instance that handles the model
      
    Raises:
      ProviderError: If no provider is found for the model
    """
    if model not in self._model_to_provider:
      available_models = list(self._model_to_provider.keys())
      raise ProviderError(
        "unknown",
        f"No provider found for model '{model}'. Available models: {available_models}"
      )
    
    provider_name = self._model_to_provider[model]
    return self.get_provider(provider_name)

  def list_providers(self) -> Dict[str, LLMProvider]:
    """List all initialized provider instances.
    
    Returns:
      Dictionary mapping provider names to provider instances
    """
    return self._providers.copy()

  def list_models(self) -> List[str]:
    """List all available models across all providers.
    
    Returns:
      List of model names available across all providers
    """
    return list(self._model_to_provider.keys())

  def get_model_to_provider_mapping(self) -> Dict[str, str]:
    """Get the complete model-to-provider mapping for request routing.
    
    Returns:
      Dictionary mapping model names to provider names
    """
    return self._model_to_provider.copy()

  def validate_all_providers(self) -> List[str]:
    """Validate configuration of all initialized providers.
    
    Returns:
      List of validation error messages (empty if all valid)
    """
    validation_errors = []
    
    for provider_name, provider in self._providers.items():
      try:
        errors = provider.validate_config()
        for error in errors:
          validation_errors.append(f"Provider '{provider_name}': {error}")
      except Exception as e:
        validation_errors.append(f"Provider '{provider_name}' validation failed: {str(e)}")
    
    return validation_errors

  def get_provider_discovery_info(self) -> Dict[str, Dict[str, any]]:
    """Get discovery information about all initialized providers.
    
    Returns:
      Dictionary with provider discovery information including models and status
    """
    discovery_info = {}
    
    for provider_name, provider in self._providers.items():
      provider_models = [
        model for model, prov_name in self._model_to_provider.items()
        if prov_name == provider_name
      ]
      
      discovery_info[provider_name] = {
        "provider_class": provider.__class__.__name__,
        "models": provider_models,
        "base_url": getattr(provider.config, 'base_url', None),
        "timeout": provider.config.timeout,
        "max_retries": provider.config.max_retries,
        "default_params": provider.config.default_params,
        "validation_status": "valid" if not provider.validate_config() else "invalid"
      }
    
    return discovery_info

  def __str__(self) -> str:
    """String representation of the factory."""
    return f"ProviderFactory(providers={len(self._providers)}, models={len(self._model_to_provider)})"

  def __repr__(self) -> str:
    """Detailed string representation of the factory."""
    provider_names = list(self._providers.keys())
    return (
      f"ProviderFactory("
      f"providers={provider_names}, "
      f"models={len(self._model_to_provider)}, "
      f"registered_types={list(self._provider_registry.keys())})"
    )