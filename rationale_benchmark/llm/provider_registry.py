"""Registry for provider client factories."""

from __future__ import annotations

from typing import Callable, Dict

from .config.connector_models import LLMConnectorConfig, ProviderType
from .exceptions import ConfigurationError
from .provider_client import BaseProviderClient
from .providers import (
  AliyunClient,
  AnthropicClient,
  GeminiClient,
  OpenAIChatClient,
  OpenAICompatibleClient,
)


ProviderFactoryFn = Callable[[LLMConnectorConfig], BaseProviderClient]


class ProviderRegistry:
  """Maintain a registry of provider client factory functions."""

  def __init__(self, *, register_builtin: bool = True) -> None:
    self._registry: Dict[ProviderType, ProviderFactoryFn] = {}
    if register_builtin:
      self._install_builtin()

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

  def _install_builtin(self) -> None:
    self._registry[ProviderType.OPENAI] = OpenAIChatClient
    self._registry[ProviderType.OPENAI_COMPATIBLE] = OpenAICompatibleClient
    self._registry[ProviderType.ALIYUN] = AliyunClient
    self._registry[ProviderType.ANTHROPIC] = AnthropicClient
    self._registry[ProviderType.GEMINI] = GeminiClient
