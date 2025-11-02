"""Registry for provider client factories."""

from __future__ import annotations

from typing import Callable, Dict

from .config.connector_models import LLMConnectorConfig, ProviderType
from .exceptions import ConfigurationError
from .provider_client import BaseProviderClient


ProviderFactoryFn = Callable[[LLMConnectorConfig], BaseProviderClient]


class ProviderRegistry:
  """Maintain a registry of provider client factory functions."""

  def __init__(self) -> None:
    self._registry: Dict[ProviderType, ProviderFactoryFn] = {}

  def register(
    self, provider_type: ProviderType, factory: ProviderFactoryFn
  ) -> None:
    """Register a factory for ``provider_type``."""

    if provider_type in self._registry:
      raise ConfigurationError(
        f"Provider '{provider_type.value}' already registered"
      )
    self._registry[provider_type] = factory

  def create(self, config: LLMConnectorConfig) -> BaseProviderClient:
    """Create a provider client for ``config``."""

    try:
      factory = self._registry[config.provider]
    except KeyError as exc:
      raise ConfigurationError(
        f"No provider registered for '{config.provider.value}'"
      ) from exc
    return factory(config)

  def available_providers(self) -> Dict[ProviderType, ProviderFactoryFn]:
    """Return the registered provider factories."""

    return dict(self._registry)

