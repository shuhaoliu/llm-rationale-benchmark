"""OpenAI-compatible provider client with configurable auth headers."""

from __future__ import annotations

from typing import Dict

from ..config.connector_models import LLMConnectorConfig
from ..exceptions import ConfigurationError
from .openai import OpenAIChatClient


class OpenAICompatibleClient(OpenAIChatClient):
  """Variant of :class:`OpenAIChatClient` for OpenAI-compatible services.

  Many OpenAI-compatible endpoints require custom base URLs or authentication
  headers. This implementation lets providers override the header name and
  prefix used for the API key via ``provider_specific`` options while reusing
  the OpenAI payload/response logic.
  """

  def __init__(self, config: LLMConnectorConfig):
    super().__init__(config)
    base = config.base_url or config.endpoint
    if not base:
      raise ConfigurationError(
        "openai_compatible providers require 'base_url' or 'endpoint'",
        field=f"providers.{config.provider.value}.base_url",
      )
    self.base_url = base.rstrip("/")
    self.headers = self._build_headers()

  def _build_headers(self) -> Dict[str, str]:
    provider_specific = self.config.provider_specific
    header_name = provider_specific.get("api_key_header", "Authorization")
    prefix = provider_specific.get("api_key_prefix", "Bearer ")

    api_key = self._require_api_key()
    auth_value = f"{prefix}{api_key}" if prefix else api_key

    headers = {
      header_name: auth_value,
      "Content-Type": "application/json",
    }

    custom_headers = provider_specific.get("headers", {})
    if isinstance(custom_headers, dict):
      for key, value in custom_headers.items():
        headers[str(key)] = str(value)

    return headers
