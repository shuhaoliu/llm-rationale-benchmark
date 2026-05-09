"""Focused tests for per-call output schema propagation."""

from __future__ import annotations

from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
  RetryPolicy,
)
from rationale_benchmark.llm.conversation import LLMConversation
from rationale_benchmark.llm.provider_client import BaseProviderClient, ProviderResponse


class RecordingProvider(BaseProviderClient):
  def __init__(self, config: LLMConnectorConfig) -> None:
    super().__init__(config)
    self.last_output_schema: dict[str, object] | None = None

  def _generate(
    self,
    messages,
    *,
    output_schema,
  ) -> ProviderResponse:
    self.last_output_schema = output_schema
    return ProviderResponse(content='{"answer": 5}', raw={"messages": messages})


def test_conversation_passes_output_schema_per_request() -> None:
  config = LLMConnectorConfig(
    provider=ProviderType.OPENAI,
    model="unit-test",
    api_key="dummy",
    retry=RetryPolicy(
      max_attempts=1,
      initial_delay=0.01,
      multiplier=1.0,
      max_delay=0.01,
      jitter=0.0,
    ),
  )
  provider = RecordingProvider(config)
  conversation = LLMConversation(
    config=config,
    provider_client=provider,
    sleep_fn=lambda _delay: None,
  )
  output_schema = {
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

  response = conversation.ask("Prompt", output_schema=output_schema)

  assert response.parsed == {"answer": 5}
  assert provider.last_output_schema == output_schema
