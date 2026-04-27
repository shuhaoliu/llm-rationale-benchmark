from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
  ResponseFormat,
)
from rationale_benchmark.llm.conversation import LLMConversation
from rationale_benchmark.llm.exceptions import ConfigurationError
from rationale_benchmark.llm.provider_client import BaseProviderClient, ProviderResponse
from rationale_benchmark.questionnaire.models import (
  Question,
  Questionnaire,
  QuestionType,
  ScoringRule,
  Section,
)
from rationale_benchmark.runner.executor import BenchmarkRunner, RunnerConfigError


class RecordingProvider(BaseProviderClient):
  """Provider returning configured responses and recording request messages."""

  def __init__(
    self,
    config: LLMConnectorConfig,
    responses: list[str],
    requests: list[list[dict[str, str]]],
  ) -> None:
    super().__init__(config)
    self._responses = responses
    self._requests = requests
    self._index = 0

  def _generate(
    self,
    messages: list[dict[str, str]],
    *,
    response_format: ResponseFormat,
  ) -> ProviderResponse:
    self._requests.append(list(messages))
    content = self._responses[self._index]
    self._index += 1
    return ProviderResponse(
      content=content,
      raw={"messages": messages},
      metadata={"request_index": self._index},
    )


class StubConversationFactory:
  """Factory creating conversations with static responses."""

  def __init__(self, responses_by_llm_id: dict[str, list[str]]) -> None:
    self._responses_by_llm_id = {
      llm_id: list(responses) for llm_id, responses in responses_by_llm_id.items()
    }
    self.created_system_prompts: list[str | None] = []
    self.requests_by_llm_id: dict[str, list[list[dict[str, str]]]] = {}

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation:
    if model not in self._responses_by_llm_id:
      raise ConfigurationError(f"Unknown model '{model}'", config_file=None)

    self.created_system_prompts.append(system_prompt)
    requests = self.requests_by_llm_id.setdefault(model, [])
    config = LLMConnectorConfig(
      provider=ProviderType.OPENAI,
      model=model,
      api_key="test-key",
      system_prompt=system_prompt,
      response_format=ResponseFormat.JSON,
    )
    provider = RecordingProvider(
      config,
      self._responses_by_llm_id[model],
      requests,
    )
    return LLMConversation(
      config=config,
      provider_client=provider,
      system_prompt=system_prompt,
    )


def build_questionnaire(default_population: int = 1) -> Questionnaire:
  rating_question = Question(
    id="rating-1",
    type=QuestionType.RATING_5,
    prompt="Rate helpfulness from 1 to 5.",
    options=None,
    scoring=ScoringRule(
      total=5,
      weights={str(value): value for value in range(1, 6)},
    ),
  )
  choice_question = Question(
    id="choice-1",
    type=QuestionType.CHOICE,
    prompt="Pick the preferred option.",
    options={"a": "Option A", "b": "Option B"},
    scoring=ScoringRule(
      total=3,
      weights={"a": 3, "b": 1},
    ),
  )
  section = Section(
    name="General",
    instructions="Answer each question carefully.",
    questions=[rating_question, choice_question],
  )
  return Questionnaire(
    id="test-questionnaire",
    name="Test Questionnaire",
    description=None,
    version=1,
    metadata={"default_population": default_population},
    default_population=default_population,
    system_prompt="System prompt",
    sections=[section],
  )


def build_two_section_questionnaire() -> Questionnaire:
  return Questionnaire(
    id="two-section",
    name="Two Section Questionnaire",
    description=None,
    version=1,
    metadata={"default_population": 1},
    default_population=1,
    system_prompt="System prompt",
    sections=[
      Section(
        name="First",
        instructions=None,
        questions=[
          Question(
            id="first-1",
            type=QuestionType.RATING_5,
            prompt="First section question.",
            options=None,
            scoring=ScoringRule(total=5, weights={"5": 5}),
          )
        ],
      ),
      Section(
        name="Second",
        instructions=None,
        questions=[
          Question(
            id="second-1",
            type=QuestionType.RATING_5,
            prompt="Second section question.",
            options=None,
            scoring=ScoringRule(total=5, weights={"5": 5}),
          )
        ],
      ),
    ],
  )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  return [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line
  ]


def test_runner_writes_raw_jsonl_records(tmp_path: Path) -> None:
  questionnaire = build_questionnaire()
  output_path = tmp_path / "responses.jsonl"
  factory = StubConversationFactory(
    {
      "openai/gpt-4": [
        json.dumps({"answer": 5, "reasoning": "Helpful."}),
        json.dumps({"answer": "a", "reasoning": "Option A."}),
      ]
    },
  )
  runner = BenchmarkRunner(factory, max_concurrency=2)

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4"],
    output_path=output_path,
  )

  assert result.errors == []
  assert len(result.records) == 1
  records = read_jsonl(output_path)
  assert len(records) == 1
  record = records[0]
  assert record["questionnaire"]["name"] == "test-questionnaire"
  assert record["llm_id"] == "openai/gpt-4"
  assert record["population_index"] == 0
  assert record["query_time"]
  assert "summary" not in record
  assert "score" not in json.dumps(record)
  assert record["response"]["sections"][0]["id"] == "General"
  questions = record["response"]["sections"][0]["questions"]
  assert [question["id"] for question in questions] == ["rating-1", "choice-1"]
  assert questions[0]["response"]["raw"] == json.dumps(
    {"answer": 5, "reasoning": "Helpful."}
  )
  assert questions[0]["response"]["parsed"]["answer"] == 5


def test_runner_creates_records_for_each_llm_and_population() -> None:
  questionnaire = build_questionnaire(default_population=2)
  factory = StubConversationFactory(
    {
      "openai/gpt-4": [json.dumps({"answer": 5})] * 4,
      "anthropic/claude": [json.dumps({"answer": 4})] * 4,
    },
  )
  runner = BenchmarkRunner(factory, max_concurrency=3)

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4", "anthropic/claude"],
  )

  assert len(result.records) == 4
  assert {
    (record.llm_id, record.population_index) for record in result.records
  } == {
    ("openai/gpt-4", 0),
    ("openai/gpt-4", 1),
    ("anthropic/claude", 0),
    ("anthropic/claude", 1),
  }


def test_runner_rejects_multiple_questionnaires() -> None:
  runner = BenchmarkRunner(StubConversationFactory({}))

  with pytest.raises(RunnerConfigError, match="exactly one questionnaire"):
    runner.run_sync(
      questionnaires=[build_questionnaire(), build_questionnaire()],
      llm_ids=["openai/gpt-4"],
    )


def test_sections_use_independent_contexts_and_questions_keep_history() -> None:
  questionnaire = build_two_section_questionnaire()
  factory = StubConversationFactory(
    {"openai/gpt-4": [json.dumps({"answer": 5}), json.dumps({"answer": 4})]},
  )
  runner = BenchmarkRunner(factory, max_concurrency=2)

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4"],
  )

  assert result.errors == []
  requests = factory.requests_by_llm_id["openai/gpt-4"]
  assert len(requests) == 2
  user_messages = [
    [message["content"] for message in request if message["role"] == "user"]
    for request in requests
  ]
  assert ["First section question."] in user_messages
  assert ["Second section question."] in user_messages


def test_runner_records_configuration_errors() -> None:
  questionnaire = build_questionnaire()
  factory = StubConversationFactory({})
  runner = BenchmarkRunner(factory, max_concurrency=1)

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["missing/model"],
  )

  assert len(result.records) == 1
  assert result.records[0].llm_id == "missing/model"
  assert result.records[0].errors
  assert result.records[0].errors[0].stage == "configuration"
