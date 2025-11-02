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
  path = write_config(
    tmp_path,
    """
    defaults:
      timeout_seconds: 45
      response_format: text
      retry:
        max_attempts: 2
        initial_delay: 0.1
        multiplier: 1.0
        max_delay: 0.1
    models:
      openai/gpt-eval:
        api_key: ${TEST_API_KEY}
        max_tokens: 256
      anthropic/claude-mini:
        api_key: plain-token
        response_format: json
        retry:
          max_attempts: 4
    """,
  )

  loader = ConnectorConfigLoader()
  configs = loader.load(path)

  assert set(configs.keys()) == {"openai/gpt-eval", "anthropic/claude-mini"}

  openai_config = configs["openai/gpt-eval"]
  assert isinstance(openai_config, LLMConnectorConfig)
  assert openai_config.provider is ProviderType.OPENAI
  assert openai_config.timeout_seconds == 45
  assert openai_config.response_format is ResponseFormat.TEXT
  assert openai_config.retry.max_attempts == 2
  assert openai_config.max_tokens == 256
  assert openai_config.api_key == "secret-token"

  anthropic_config = configs["anthropic/claude-mini"]
  assert anthropic_config.response_format is ResponseFormat.JSON
  assert anthropic_config.retry.max_attempts == 4


def test_invalid_model_selector_raises(tmp_path):
  path = write_config(
    tmp_path,
    """
    models:
      gpt-4:
        api_key: token
    """,
  )

  loader = ConnectorConfigLoader()

  with pytest.raises(ConfigurationError) as exc:
    loader.load(path)

  assert "provider/model" in str(exc.value)


def test_missing_environment_variable(tmp_path):
  path = write_config(
    tmp_path,
    """
    models:
      openai/gpt-test:
        api_key: ${MISSING_KEY}
    """,
  )

  loader = ConnectorConfigLoader()

  with pytest.raises(ConfigurationError) as exc:
    loader.load(path)

  assert "MISSING_KEY" in str(exc.value)

