"""Factory producing :class:`LLMConversation` instances."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .config.connector_models import LLMConnectorConfig
from .config.connector_loader import ConnectorConfigLoader
from .conversation import LLMConversation
from .exceptions import ConfigurationError
from .provider_client import BaseProviderClient
from .provider_registry import ProviderRegistry


class LLMConversationFactory:
  """Create conversations from configuration files."""

  def __init__(
    self,
    *,
    loader: ConnectorConfigLoader | None = None,
    registry: ProviderRegistry | None = None,
  ) -> None:
    self.loader = loader or ConnectorConfigLoader()
    self.registry = registry or ProviderRegistry()
    self._client_cache: Dict[str, BaseProviderClient] = {}

  def create_from_config(
    self,
    config_path: Path,
    target_model: str,
    system_prompt: Optional[str] = None,
  ) -> LLMConversation:
    """Create a conversation from ``config_path`` selecting ``target_model``."""

    configs = self.loader.load(config_path)

    if target_model not in configs:
      available = ", ".join(sorted(configs.keys()))
      raise ConfigurationError(
        f"Model '{target_model}' not found. Available models: {available}",
        config_file=str(config_path),
        field="models",
      )

    config = configs[target_model]
    client = self._client_for_config(config)

    prompt = system_prompt if system_prompt is not None else config.system_prompt

    return LLMConversation(
      config=config,
      provider_client=client,
      system_prompt=prompt,
    )

  def _client_for_config(self, config: LLMConnectorConfig) -> BaseProviderClient:
    cache_key = config.cache_key()
    if cache_key not in self._client_cache:
      self._client_cache[cache_key] = self.registry.create(config)
    return self._client_cache[cache_key]

