"""Tests for provider request payload construction."""

from __future__ import annotations

from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
)
from rationale_benchmark.llm.providers.aliyun import AliyunClient
from rationale_benchmark.llm.providers.anthropic import AnthropicClient
from rationale_benchmark.llm.providers.gemini import GeminiClient
from rationale_benchmark.llm.providers.openai import OpenAIChatClient
from rationale_benchmark.llm.providers.openai_compatible import (
  OpenAICompatibleClient,
)


OUTPUT_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "answer": {
      "type": "integer",
      "minimum": 1,
      "maximum": 5,
    }
  },
  "required": ["answer"],
  "title": "rating_01",
}


def test_openai_payload_omits_temperature_when_unset():
  config = LLMConnectorConfig(
    provider=ProviderType.OPENAI,
    model="gpt-4o-mini",
    api_key="token",
  )
  client = OpenAIChatClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    OUTPUT_SCHEMA,
  )

  assert "temperature" not in payload


def test_anthropic_payload_omits_temperature_when_unset():
  config = LLMConnectorConfig(
    provider=ProviderType.ANTHROPIC,
    model="claude-3-5-sonnet",
    api_key="token",
  )
  client = AnthropicClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    OUTPUT_SCHEMA,
  )

  assert "temperature" not in payload


def test_gemini_payload_omits_temperature_when_unset():
  config = LLMConnectorConfig(
    provider=ProviderType.GEMINI,
    model="gemini-2.5-pro",
    api_key="token",
  )
  client = GeminiClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    OUTPUT_SCHEMA,
  )

  assert "generationConfig" not in payload


def test_openai_payload_keeps_explicit_temperature():
  config = LLMConnectorConfig(
    provider=ProviderType.OPENAI,
    model="gpt-4o-mini",
    api_key="token",
    temperature=0.4,
  )
  client = OpenAIChatClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    OUTPUT_SCHEMA,
  )

  assert payload["temperature"] == 0.4


def test_openai_payload_uses_json_schema_response_format():
  config = LLMConnectorConfig(
    provider=ProviderType.OPENAI,
    model="gpt-4o-mini",
    api_key="token",
  )
  client = OpenAIChatClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    OUTPUT_SCHEMA,
  )

  assert payload["response_format"] == {
    "type": "json_schema",
    "json_schema": {
      "name": "rating_01",
      "strict": True,
      "schema": OUTPUT_SCHEMA,
    },
  }


def test_anthropic_payload_uses_output_config_format():
  config = LLMConnectorConfig(
    provider=ProviderType.ANTHROPIC,
    model="claude-sonnet-4-5",
    api_key="token",
  )
  client = AnthropicClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    OUTPUT_SCHEMA,
  )

  assert payload["output_config"] == {
    "format": {
      "type": "json_schema",
      "schema": OUTPUT_SCHEMA,
    }
  }


def test_openai_compatible_payload_uses_json_schema_for_standard_backends():
  config = LLMConnectorConfig(
    provider=ProviderType.OPENAI_COMPATIBLE,
    model="qwen3-max",
    api_key="token",
    base_url="https://openrouter.ai/api/v1",
  )
  client = OpenAICompatibleClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    OUTPUT_SCHEMA,
  )

  assert payload["response_format"]["type"] == "json_schema"


def test_aliyun_payload_omits_native_structured_output_for_reasoning_models():
  config = LLMConnectorConfig(
    provider=ProviderType.ALIYUN,
    model="qwen3.6-plus",
    api_key="token",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    default_params={"enable_thinking": True},
  )
  client = AliyunClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    OUTPUT_SCHEMA,
  )

  assert "response_format" not in payload


def test_aliyun_payload_omits_native_structured_output_by_default():
  config = LLMConnectorConfig(
    provider=ProviderType.ALIYUN,
    model="qwen3.6-plus",
    api_key="token",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  )
  client = AliyunClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    OUTPUT_SCHEMA,
  )

  assert "response_format" not in payload


def test_aliyun_payload_uses_native_structured_output_when_thinking_disabled():
  config = LLMConnectorConfig(
    provider=ProviderType.ALIYUN,
    model="qwen3.6-plus",
    api_key="token",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    default_params={"enable_thinking": False},
  )
  client = AliyunClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    OUTPUT_SCHEMA,
  )

  assert payload["response_format"]["type"] == "json_schema"
