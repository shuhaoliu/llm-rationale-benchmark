"""Tests for provider registry built-in registrations."""

from __future__ import annotations

import pytest

from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
)
from rationale_benchmark.llm.exceptions import ConfigurationError
from rationale_benchmark.llm.provider_registry import ProviderRegistry
from rationale_benchmark.llm.providers.openai_compatible import (
  OpenAICompatibleClient,
)


def make_config(**overrides) -> LLMConnectorConfig:
  base = {
    "provider": ProviderType.OPENAI_COMPATIBLE,
    "model": "demo-model",
    "api_key": "secret",
    "base_url": "https://example.test/v1",
  }
  base.update(overrides)
  return LLMConnectorConfig.model_validate(base)


def test_registry_provides_openai_compatible_client():
  registry = ProviderRegistry()
  config = make_config(
    provider_specific={
      "api_key_header": "X-API-Key",
      "api_key_prefix": "",
      "headers": {"Custom": "value"},
    }
  )

  client = registry.create(config)

  assert isinstance(client, OpenAICompatibleClient)
  assert client.base_url == "https://example.test/v1"
  assert client.headers["X-API-Key"] == "secret"
  assert "Authorization" not in client.headers
  assert client.headers["Custom"] == "value"


def test_openai_compatible_requires_endpoint():
  config = LLMConnectorConfig(
    provider=ProviderType.OPENAI_COMPATIBLE,
    model="demo",
    api_key="secret",
  )

  with pytest.raises(ConfigurationError):
    OpenAICompatibleClient(config)
