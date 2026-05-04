"""Tests for :mod:`rationale_benchmark.llm.config.connector_loader`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rationale_benchmark.llm.config.connector_loader import ConnectorConfigLoader
from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
  ResponseFormat,
)
from rationale_benchmark.llm.exceptions import ConfigurationError


def write_config(tmp_path, contents: str) -> Path:
  path = tmp_path / "config.yaml"
  path.write_text(textwrap.dedent(contents), encoding="utf-8")
  return path


def test_load_merges_defaults_and_resolves_env(tmp_path, monkeypatch):
  monkeypatch.setenv("TEST_API_KEY", "secret-token")
  default_path = tmp_path / "default-llms.yaml"
  default_path.write_text(
    textwrap.dedent(
      """
      defaults:
        timeout: 45
        response_format: json
        max_retries: 2
        default_params:
          temperature: 0.5
      providers:
        openai:
          api_key: ${TEST_API_KEY}
          models:
            - gpt-eval
            - gpt-backup
          default_params:
            temperature: 0.3
      """
    ),
    encoding="utf-8",
  )

  path = write_config(
    tmp_path,
    """
    defaults:
      response_format: text
      default_params:
        top_p: 0.9
    providers:
      openai:
        models:
          - name: gpt-custom
            default_params:
              temperature: 0.2
              extra: true
        max_tokens: 256
      anthropic:
        api_key: plain-token
        models:
          - claude-mini
        response_format: json
        retry:
          max_attempts: 4
    """,
  )

  loader = ConnectorConfigLoader()
  configs = loader.load(path)

  assert set(configs.keys()) == {
    "openai/gpt-custom",
    "anthropic/claude-mini",
  }

  openai_config = configs["openai/gpt-custom"]
  assert isinstance(openai_config, LLMConnectorConfig)
  assert openai_config.provider is ProviderType.OPENAI
  assert openai_config.timeout_seconds == 45
  assert openai_config.response_format is ResponseFormat.TEXT
  assert openai_config.retry.max_attempts == 2
  assert openai_config.max_tokens == 256
  assert openai_config.api_key == "secret-token"
  assert openai_config.default_params["temperature"] == 0.2
  assert openai_config.default_params["top_p"] == 0.9
  assert openai_config.default_params["extra"] is True

  assert "openai/gpt-backup" not in configs

  anthropic_config = configs["anthropic/claude-mini"]
  assert anthropic_config.response_format is ResponseFormat.JSON
  assert anthropic_config.retry.max_attempts == 4


def test_invalid_models_section_raises(tmp_path):
  path = write_config(
    tmp_path,
    """
    providers:
      openai:
        models: null
    """,
  )

  loader = ConnectorConfigLoader()

  with pytest.raises(ConfigurationError) as exc:
    loader.load(path)

  assert "models" in str(exc.value)


def test_missing_environment_variable(tmp_path):
  path = write_config(
    tmp_path,
    """
    providers:
      openai:
        api_key: ${MISSING_KEY}
        models:
          - gpt-test
    """,
  )

  loader = ConnectorConfigLoader()

  with pytest.raises(ConfigurationError) as exc:
    loader.load(path)

  assert "MISSING_KEY" in str(exc.value)


def test_suffix_openai_compatible_provider(tmp_path):
  path = write_config(
    tmp_path,
    """
    providers:
      openrouter_openai_compatible:
        api_key: token
        base_url: https://openrouter.ai/api/v1
        models:
          - qwen3-max
    """,
  )

  loader = ConnectorConfigLoader()
  configs = loader.load(path)

  selector = "openrouter_openai_compatible/qwen3-max"
  assert selector in configs
  config = configs[selector]
  assert config.provider is ProviderType.OPENAI_COMPATIBLE
  assert config.base_url == "https://openrouter.ai/api/v1"


def test_zero_timeout_means_no_client_side_timeout(tmp_path):
  path = write_config(
    tmp_path,
    """
    defaults:
      timeout: 0
    providers:
      openrouter_openai_compatible:
        api_key: token
        base_url: https://openrouter.ai/api/v1
        models:
          - qwen3-max
    """,
  )

  loader = ConnectorConfigLoader()
  configs = loader.load(path)

  assert configs["openrouter_openai_compatible/qwen3-max"].timeout_seconds == 0


def test_streaming_parameters_are_rejected(tmp_path):
  path = write_config(
    tmp_path,
    """
    providers:
      openai:
        api_key: token
        models:
          - name: gpt-4
            default_params:
              stream: true
    """,
  )

  loader = ConnectorConfigLoader()

  with pytest.raises(ConfigurationError) as exc:
    loader.load(path)

  assert "Streaming parameters" in str(exc.value)
