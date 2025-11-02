"""Tests for :class:`rationale_benchmark.llm.conversation_factory.LLMConversationFactory`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rationale_benchmark.llm.config.connector_models import (
  ProviderType,
  ResponseFormat,
)
from rationale_benchmark.llm.conversation_factory import LLMConversationFactory
from rationale_benchmark.llm.exceptions import ConfigurationError
from rationale_benchmark.llm.provider_client import BaseProviderClient, ProviderResponse
from rationale_benchmark.llm.provider_registry import ProviderRegistry


class EchoProvider(BaseProviderClient):
  def __init__(self, config):
    super().__init__(config)
    self.calls = 0

  def _generate(self, messages, *, response_format):
    self.calls += 1
    reply = messages[-1]["content"].upper()
    if response_format is ResponseFormat.JSON:
      reply = "{\"echo\": \"" + reply + "\"}"
    return ProviderResponse(content=reply, raw={"messages": messages})


def register_echo_provider(registry: ProviderRegistry) -> None:
  registry.register(ProviderType.OPENAI, lambda config: EchoProvider(config))


def write_config(tmp_path: Path) -> Path:
  contents = textwrap.dedent(
    """
    providers:
      openai:
        api_key: token
        models:
          - gpt-sim
    """
  )
  path = tmp_path / "llms.yaml"
  path.write_text(contents, encoding="utf-8")
  return path


def test_factory_creates_conversation_and_reuses_client(tmp_path):
  registry = ProviderRegistry(register_builtin=False)
  register_echo_provider(registry)
  factory = LLMConversationFactory(registry=registry)

  config_path = write_config(tmp_path)

  conversation_one = factory.create_from_config(
    config_path,
    "openai/gpt-sim",
    system_prompt="echo",
  )
  conversation_two = factory.create_from_config(
    config_path,
    "openai/gpt-sim",
  )

  assert conversation_one.provider_client is conversation_two.provider_client
  assert conversation_one.system_prompt == "echo"

  response = conversation_two.ask("hello", validator=lambda r: True)
  assert response.parsed == {"echo": "HELLO"}


def test_factory_raises_for_unknown_model(tmp_path):
  registry = ProviderRegistry(register_builtin=False)
  register_echo_provider(registry)
  factory = LLMConversationFactory(registry=registry)

  config_path = write_config(tmp_path)

  with pytest.raises(ConfigurationError):
    factory.create_from_config(config_path, "openai/unknown")
