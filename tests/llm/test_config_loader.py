"""Tests for :mod:`rationale_benchmark.llm.config.connector_loader`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rationale_benchmark.llm.config.connector_loader import ConnectorConfigLoader
from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
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
  assert openai_config.retry.max_attempts == 2
  assert openai_config.max_tokens == 256
  assert openai_config.api_key == "secret-token"
  assert openai_config.default_params["temperature"] == 0.2
  assert openai_config.default_params["top_p"] == 0.9
  assert openai_config.default_params["extra"] is True

  assert "openai/gpt-backup" not in configs

  anthropic_config = configs["anthropic/claude-mini"]
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


def test_aliyun_provider_name_is_supported_as_dedicated_provider(tmp_path):
  path = write_config(
    tmp_path,
    """
    providers:
      aliyun:
        api_key: token
        base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
        models:
          - qwen3.6-plus
    """,
  )

  loader = ConnectorConfigLoader()
  configs = loader.load(path)

  config = configs["aliyun/qwen3.6-plus"]
  assert config.provider is ProviderType.ALIYUN
  assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_suffix_aliyun_provider_is_rejected(tmp_path):
  path = write_config(
    tmp_path,
    """
    providers:
      dashscope_aliyun:
        api_key: token
        base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
        models:
          - qwen3.6-plus
    """,
  )

  loader = ConnectorConfigLoader()

  with pytest.raises(ConfigurationError) as exc:
    loader.load(path)

  assert "Unknown provider" in str(exc.value)


def test_aliyun_model_suffix_zero_disables_thinking(tmp_path):
  path = write_config(
    tmp_path,
    """
    providers:
      aliyun:
        api_key: token
        base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
        models:
          - qwen3.6-plus (0)
    """,
  )

  loader = ConnectorConfigLoader()
  configs = loader.load(path)

  config = configs["aliyun/qwen3.6-plus (0)"]
  assert config.default_params["enable_thinking"] is False
  assert "thinking_budget" not in config.default_params


def test_aliyun_model_suffix_positive_sets_budget(tmp_path):
  path = write_config(
    tmp_path,
    """
    providers:
      aliyun:
        api_key: token
        base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
        models:
          - qwen3.6-plus (10)
    """,
  )

  loader = ConnectorConfigLoader()
  configs = loader.load(path)

  config = configs["aliyun/qwen3.6-plus (10)"]
  assert config.default_params["enable_thinking"] is True
  assert config.default_params["thinking_budget"] == 10
  assert config.model == "qwen3.6-plus"


def test_aliyun_model_suffix_negative_one_enables_thinking_without_budget(
  tmp_path,
):
  path = write_config(
    tmp_path,
    """
    providers:
      aliyun:
        api_key: token
        base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
        models:
          - qwen3.6-plus (-1)
    """,
  )

  loader = ConnectorConfigLoader()
  configs = loader.load(path)

  config = configs["aliyun/qwen3.6-plus (-1)"]
  assert config.default_params["enable_thinking"] is True
  assert "thinking_budget" not in config.default_params


def test_aliyun_model_suffix_rejects_other_negative_values(tmp_path):
  path = write_config(
    tmp_path,
    """
    providers:
      aliyun:
        api_key: token
        base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
        models:
          - qwen3.6-plus (-2)
    """,
  )

  loader = ConnectorConfigLoader()

  with pytest.raises(ConfigurationError) as exc:
    loader.load(path)

  assert "thinking suffix" in str(exc.value)


def test_thinking_config_keeps_reasoning_levels_as_distinct_models(
  monkeypatch,
):
  monkeypatch.setenv("DASHSCOPE_API_KEY", "token")
  loader = ConnectorConfigLoader()

  configs = loader.load(
    Path("config/llms/thinking.yaml"),
    merge_default=False,
  )

  assert list(configs.keys()) == [
    "aliyun/qwen3.6-plus (0)",
    "aliyun/qwen3.6-plus (-1)",
    "aliyun/qwen3.5-plus (0)",
    "aliyun/qwen3.5-plus (-1)",
    "aliyun/deepseek-v4-pro (0)",
    "aliyun/deepseek-v4-pro (-1)",
    "aliyun/glm-5.1 (0)",
    "aliyun/glm-5.1 (-1)",
  ]
  assert configs["aliyun/qwen3.6-plus (0)"].default_params == {
    "enable_thinking": False,
  }
  assert configs["aliyun/qwen3.6-plus (-1)"].default_params == {
    "enable_thinking": True,
  }


def test_deepseek_config_defaults_to_runnable_models(monkeypatch):
  monkeypatch.setenv("DASHSCOPE_API_KEY", "token")
  loader = ConnectorConfigLoader()

  configs = loader.load(
    Path("config/llms/deepseek.yaml"),
    merge_default=False,
  )

  assert list(configs.keys()) == [
    "aliyun/deepseek-v4-pro (0)",
    "aliyun/deepseek-v3.2 (0)",
  ]
  assert configs["aliyun/deepseek-v4-pro (0)"].default_params == {
    "enable_thinking": False,
  }
  assert configs["aliyun/deepseek-v3.2 (0)"].default_params == {
    "enable_thinking": False,
  }


def test_default_llms_does_not_set_a_global_temperature(monkeypatch):
  monkeypatch.setenv("DASHSCOPE_API_KEY", "token")
  loader = ConnectorConfigLoader()

  configs = loader.load(
    Path("config/llms/default-llms.yaml"),
    merge_default=False,
  )

  assert configs
  assert all(config.temperature is None for config in configs.values())


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


def test_response_format_configuration_is_rejected(tmp_path):
  path = write_config(
    tmp_path,
    """
    defaults:
      response_format: text
    providers:
      openai:
        api_key: token
        models:
          - gpt-4
    """,
  )

  loader = ConnectorConfigLoader()

  with pytest.raises(ConfigurationError) as exc:
    loader.load(path)

  assert "response_format" in str(exc.value)
