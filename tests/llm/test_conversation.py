"""Unit tests for :class:`rationale_benchmark.llm.conversation.LLMConversation`."""

from __future__ import annotations

from typing import List

import pytest

from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
  ResponseFormat,
  RetryPolicy,
)
from rationale_benchmark.llm.conversation import LLMConversation
from rationale_benchmark.llm.exceptions import (
  ConversationArchivedError,
  RetryableProviderError,
  TimeoutError,
  ValidationFailedError,
)
from rationale_benchmark.llm.provider_client import BaseProviderClient, ProviderResponse


def build_config(
  *,
  response_format: ResponseFormat = ResponseFormat.JSON,
  requires_streaming: bool = False,
  max_attempts: int = 3,
) -> LLMConnectorConfig:
  return LLMConnectorConfig(
    provider=ProviderType.OPENAI,
    model="unit-test",
    api_key="dummy",
    response_format=response_format,
    requires_streaming=requires_streaming,
    retry=RetryPolicy(
      max_attempts=max_attempts,
      initial_delay=0.01,
      multiplier=1.0,
      max_delay=0.01,
      jitter=0.0,
    ),
  )


class ToggleProvider(BaseProviderClient):
  def __init__(self, config: LLMConnectorConfig, responses: List[object]):
    super().__init__(config)
    self._responses = responses
    self.calls: int = 0

  def _generate(
    self,
    messages: List[dict[str, str]],
    *,
    response_format: ResponseFormat,
  ) -> ProviderResponse:
    response = self._responses[self.calls]
    self.calls += 1
    if isinstance(response, Exception):
      raise response
    return ProviderResponse(content=response, raw={"messages": messages})


class StreamingProvider(BaseProviderClient):
  def __init__(self, config: LLMConnectorConfig, chunks: List[str]):
    super().__init__(config)
    self.chunks = chunks
    self.calls = 0

  def supports_streaming(self) -> bool:
    return True

  def _generate(
    self,
    messages: List[dict[str, str]],
    *,
    response_format: ResponseFormat,
  ) -> ProviderResponse:
    raise AssertionError("Non-streaming path should not be used")

  def stream_generate(
    self,
    messages: List[dict[str, str]],
    *,
    response_format: ResponseFormat,
  ):
    self.calls += 1
    return list(self.chunks)


def test_json_parse_failure_retries_until_success():
  config = build_config()
  provider = ToggleProvider(
    config,
    responses=["{", "{\"value\": 1}"]
  )
  conversation = LLMConversation(
    config=config,
    provider_client=provider,
    system_prompt="behave",
    sleep_fn=lambda _delay: None,
  )

  response = conversation.ask("Question?")

  assert response.parsed == {"value": 1}
  assert provider.calls == 2
  assert conversation.history[-1].verification_errors == []
  assert conversation.history[-2].verification_errors == [
    "json_parse_error: Expecting property name enclosed in double quotes"
  ]

  archive = conversation.archive()
  assert archive.metadata["json_parse_failures"] == 1

  with pytest.raises(ConversationArchivedError):
    conversation.ask("still there?")


def test_retryable_provider_error_retries():
  config = build_config()

  class FlakyProvider(BaseProviderClient):
    def __init__(self, config: LLMConnectorConfig):
      super().__init__(config)
      self.calls = 0

    def _generate(self, messages, *, response_format):
      self.calls += 1
      if self.calls == 1:
        raise RetryableProviderError(
          self.config.provider.value, "try later", retry_after=0
        )
      return ProviderResponse(content="{\"ok\": true}", raw={"messages": messages})

  provider = FlakyProvider(config)
  conversation = LLMConversation(
    config=config,
    provider_client=provider,
    sleep_fn=lambda _delay: None,
  )

  response = conversation.ask("hello")
  assert response.parsed == {"ok": True}
  assert provider.calls == 2


def test_timeout_error_retries_until_success():
  config = build_config()
  provider = ToggleProvider(
    config,
    responses=[
      TimeoutError("Request timed out", timeout_seconds=30),
      "{\"ok\": true}",
    ],
  )
  conversation = LLMConversation(
    config=config,
    provider_client=provider,
    sleep_fn=lambda _delay: None,
  )

  response = conversation.ask("hello")

  assert response.parsed == {"ok": True}
  assert provider.calls == 2


def test_validator_failure_raises_after_retries():
  config = build_config(max_attempts=2)
  provider = ToggleProvider(
    config,
    responses=["{\"value\": 1}", "{\"value\": 2}"]
  )
  conversation = LLMConversation(
    config=config,
    provider_client=provider,
    sleep_fn=lambda _delay: None,
  )

  def validator(response):
    return response.parsed.get("value") == 3

  with pytest.raises(ValidationFailedError) as exc:
    conversation.ask("check", validator=validator)

  assert "Validator rejected response" in str(exc.value)
  assert provider.calls == 2


def test_streaming_provider_buffers_chunks():
  config = build_config(
    response_format=ResponseFormat.TEXT,
    requires_streaming=True,
  )
  provider = StreamingProvider(config, chunks=["Hello", " ", "world"])
  conversation = LLMConversation(
    config=config,
    provider_client=provider,
    sleep_fn=lambda _delay: None,
  )

  response = conversation.ask("compose")
  assert response.text == "Hello world"
  assert provider.calls == 1
