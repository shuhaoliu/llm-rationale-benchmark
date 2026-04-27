from __future__ import annotations

import json

import pytest

from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
  ResponseFormat,
)
from rationale_benchmark.llm.conversation import LLMConversation
from rationale_benchmark.llm.exceptions import ConfigurationError
from rationale_benchmark.llm.provider_client import StaticResponseProvider
from rationale_benchmark.questionnaire.models import (
  Question,
  Questionnaire,
  QuestionType,
  ScoringRule,
  Section,
)
from rationale_benchmark.runner.executor import BenchmarkRunner


class StubConversationFactory:
  """Factory creating conversations with static responses."""

  def __init__(self, responses_by_model: dict[str, list[str]]):
    self._responses_by_model = {
      model: list(responses) for model, responses in responses_by_model.items()
    }
    self.created_system_prompts: list[str | None] = []

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation:
    if model not in self._responses_by_model:
      raise ConfigurationError(f"Unknown model '{model}'", config_file=None)

    self.created_system_prompts.append(system_prompt)
    responses = list(self._responses_by_model[model])
    config = LLMConnectorConfig(
      provider=ProviderType.OPENAI,
      model=model,
      api_key="test-key",
      system_prompt=system_prompt,
      response_format=ResponseFormat.JSON,
    )
    provider = StaticResponseProvider(config, responses)
    return LLMConversation(
      config=config,
      provider_client=provider,
      system_prompt=system_prompt,
    )


def build_questionnaire() -> Questionnaire:
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
    metadata={"default_population": 1},
    default_population=1,
    system_prompt="System prompt",
    sections=[section],
  )


def build_two_section_questionnaire() -> Questionnaire:
  first_question = Question(
    id="first-1",
    type=QuestionType.RATING_5,
    prompt="First section question.",
    options=None,
    scoring=ScoringRule(
      total=5,
      weights={str(value): value for value in range(1, 6)},
    ),
  )
  second_question = Question(
    id="second-1",
    type=QuestionType.RATING_5,
    prompt="Second section question.",
    options=None,
    scoring=ScoringRule(
      total=5,
      weights={str(value): value for value in range(1, 6)},
    ),
  )
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
        questions=[first_question],
      ),
      Section(
        name="Second",
        instructions=None,
        questions=[second_question],
      ),
    ],
  )


def test_benchmark_runner_successful_run():
  questionnaire = build_questionnaire()
  responses = [
    json.dumps(
      {"answer": 5, "reasoning": "Strongly agree with the statement."}
    ),
    json.dumps(
      {"answer": "a", "reasoning": "Option A aligns with the rationale."}
    ),
  ]
  factory = StubConversationFactory(
    {"openai/gpt-4": responses},
  )
  runner = BenchmarkRunner(factory, max_concurrency=2)

  result = runner.run_sync(
    questionnaires=[questionnaire],
    models=["openai/gpt-4"],
  )

  assert result.info.questionnaires == ("test-questionnaire",)
  assert result.info.models_tested == ("openai/gpt-4",)
  assert result.errors == []

  assert len(result.model_results) == 1
  model_result = result.model_results[0]
  assert model_result.model == "openai/gpt-4"
  assert len(model_result.questions) == 2
  assert model_result.questions[0].score.awarded == 5
  assert model_result.questions[1].score.awarded == 3
  assert model_result.errors == []
  assert pytest.approx(model_result.cost_estimate, abs=1e-6) == 0.0

  transcript = model_result.transcript
  assert transcript[0].role == "system"
  assert transcript[0].content == questionnaire.system_prompt
  # First user prompt should match question config exactly
  assert transcript[1].role == "user"
  assert transcript[1].content == questionnaire.sections[0].questions[0].prompt
  assert transcript[3].role == "user"
  assert transcript[3].content == questionnaire.sections[0].questions[1].prompt

  questionnaire_scores = model_result.questionnaire_scores
  assert len(questionnaire_scores) == 1
  questionnaire_score = questionnaire_scores[0]
  assert questionnaire_score.awarded == questionnaire_score.total == 8

  summary = result.summary
  assert summary.total_questions == 2
  assert summary.models_tested == 1
  assert summary.average_scores_by_questionnaire["test-questionnaire"] == pytest.approx(
    1.0
  )
  assert summary.average_scores_by_model["openai/gpt-4"] == pytest.approx(1.0)
  assert summary.cost_estimates["openai/gpt-4"] == pytest.approx(0.0)


def test_sections_use_independent_conversation_contexts():
  questionnaire = build_two_section_questionnaire()
  factory = StubConversationFactory(
    {"openai/gpt-4": [json.dumps({"answer": 5})]},
  )
  runner = BenchmarkRunner(factory, max_concurrency=2)

  result = runner.run_sync(
    questionnaires=[questionnaire],
    models=["openai/gpt-4"],
  )

  assert factory.created_system_prompts == ["System prompt", "System prompt"]
  model_result = result.model_results[0]
  first_transcript = model_result.section_transcripts["First"]
  second_transcript = model_result.section_transcripts["Second"]
  assert [turn.content for turn in first_transcript if turn.role == "user"] == [
    "First section question."
  ]
  assert [turn.content for turn in second_transcript if turn.role == "user"] == [
    "Second section question."
  ]
  assert all(
    "Second section question." not in turn.content
    for turn in first_transcript
  )
  assert all(
    "First section question." not in turn.content
    for turn in second_transcript
  )


def test_benchmark_runner_handles_configuration_error():
  questionnaire = build_questionnaire()
  factory = StubConversationFactory({})
  runner = BenchmarkRunner(factory, max_concurrency=1)

  result = runner.run_sync(
    questionnaires=[questionnaire],
    models=["missing/model"],
  )

  assert any(error.stage == "configuration" for error in result.errors)
  for error in result.errors:
    assert error.model == "missing/model"

  model_result = result.model_results[0]
  assert model_result.errors
  assert model_result.questionnaire_scores[0].awarded == 0
  assert result.summary.average_scores_by_model["missing/model"] == pytest.approx(0.0)
