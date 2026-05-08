"""Tests for provider request payload construction."""

from __future__ import annotations

from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
  ResponseFormat,
)
from rationale_benchmark.llm.providers.anthropic import AnthropicClient
from rationale_benchmark.llm.providers.gemini import GeminiClient
from rationale_benchmark.llm.providers.openai import OpenAIChatClient


def test_openai_payload_omits_temperature_when_unset():
  config = LLMConnectorConfig(
    provider=ProviderType.OPENAI,
    model="gpt-4o-mini",
    api_key="token",
  )
  client = OpenAIChatClient(config)

  payload = client._build_payload(
    [{"role": "user", "content": "hello"}],
    ResponseFormat.TEXT,
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
    ResponseFormat.TEXT,
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
    ResponseFormat.TEXT,
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
    ResponseFormat.TEXT,
  )

  assert payload["temperature"] == 0.4
