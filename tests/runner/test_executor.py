from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ProviderType,
  ResponseFormat,
)
from rationale_benchmark.llm.conversation import LLMConversation
from rationale_benchmark.llm.exceptions import (
  ConfigurationError,
  LLMConnectorError,
  ProviderError,
  TimeoutError,
)
from rationale_benchmark.llm.provider_client import BaseProviderClient, ProviderResponse
from rationale_benchmark.questionnaire.models import (
  Question,
  Questionnaire,
  QuestionType,
  ScoringRule,
  Section,
)
from rationale_benchmark.runner.executor import BenchmarkRunner, RunnerConfigError
from rationale_benchmark.runner.progress_display import QueryProgressSnapshot


@pytest.fixture(autouse=True)
def isolate_default_results_dir(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)


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
    output_schema,
  ) -> ProviderResponse:
    self._requests.append(list(messages))
    content = self._responses[self._index]
    self._index += 1
    return ProviderResponse(
      content=content,
      raw={"messages": messages},
      metadata={"request_index": self._index},
    )


class ErrorProvider(BaseProviderClient):
  """Provider raising a connector error for progress error-state tests."""

  def _generate(
    self,
    messages: list[dict[str, str]],
    *,
    output_schema,
  ) -> ProviderResponse:
    raise LLMConnectorError("provider failed after retries")


class ProviderErrorProvider(BaseProviderClient):
  """Provider raising a non-retryable provider error."""

  def _generate(
    self,
    messages: list[dict[str, str]],
    *,
    output_schema,
  ) -> ProviderResponse:
    raise ProviderError(self.config.provider.value, "Model access denied.")


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


class ErrorConversationFactory:
  """Factory creating conversations that fail every provider request."""

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation:
    config = LLMConnectorConfig(
      provider=ProviderType.OPENAI,
      model=model,
      api_key="test-key",
      system_prompt=system_prompt,
      response_format=ResponseFormat.JSON,
    )
    return LLMConversation(
      config=config,
      provider_client=ErrorProvider(config),
      system_prompt=system_prompt,
    )


class ProviderErrorConversationFactory:
  """Factory creating conversations that hit a provider-side rejection."""

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation:
    config = LLMConnectorConfig(
      provider=ProviderType.OPENAI,
      model=model,
      api_key="test-key",
      system_prompt=system_prompt,
      response_format=ResponseFormat.JSON,
    )
    return LLMConversation(
      config=config,
      provider_client=ProviderErrorProvider(config),
      system_prompt=system_prompt,
    )


class TimeoutThenSuccessProvider(BaseProviderClient):
  """Provider timing out once before returning a successful answer."""

  def __init__(self, config: LLMConnectorConfig) -> None:
    super().__init__(config)
    self.calls = 0

  def _generate(
    self,
    messages: list[dict[str, str]],
    *,
    output_schema,
  ) -> ProviderResponse:
    self.calls += 1
    if self.calls == 1:
      raise TimeoutError("Request timed out", timeout_seconds=30)
    return ProviderResponse(
      content=json.dumps({"answer": 5}),
      raw={"messages": messages},
      metadata={"request_index": self.calls},
    )


class AlwaysTimeoutProvider(BaseProviderClient):
  """Provider timing out on every request."""

  def __init__(self, config: LLMConnectorConfig) -> None:
    super().__init__(config)
    self.calls = 0

  def _generate(
    self,
    messages: list[dict[str, str]],
    *,
    output_schema,
  ) -> ProviderResponse:
    self.calls += 1
    raise TimeoutError("The read operation timed out", timeout_seconds=30)


class TimeoutThenSuccessConversationFactory:
  """Factory creating conversations that recover after a timeout."""

  def __init__(self) -> None:
    self.providers: list[TimeoutThenSuccessProvider] = []

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation:
    config = LLMConnectorConfig(
      provider=ProviderType.OPENAI,
      model=model,
      api_key="test-key",
      system_prompt=system_prompt,
      response_format=ResponseFormat.JSON,
    )
    provider = TimeoutThenSuccessProvider(config)
    self.providers.append(provider)
    return LLMConversation(
      config=config,
      provider_client=provider,
      system_prompt=system_prompt,
      sleep_fn=lambda _delay: None,
    )


class AlwaysTimeoutConversationFactory:
  """Factory creating conversations that exhaust timeout retries."""

  def __init__(self) -> None:
    self.providers: list[AlwaysTimeoutProvider] = []

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation:
    config = LLMConnectorConfig(
      provider=ProviderType.OPENAI,
      model=model,
      api_key="test-key",
      system_prompt=system_prompt,
      response_format=ResponseFormat.JSON,
    )
    provider = AlwaysTimeoutProvider(config)
    self.providers.append(provider)
    return LLMConversation(
      config=config,
      provider_client=provider,
      system_prompt=system_prompt,
      sleep_fn=lambda _delay: None,
    )


class RecordingProgressDisplay:
  """Progress display test double recording runner status transitions."""

  def __init__(self) -> None:
    self.starts: list[QueryProgressSnapshot] = []
    self.started_sections: list[tuple[str, int, int]] = []
    self.completed_sections: list[tuple[str, int, int]] = []
    self.error_sections: list[tuple[str, int, int]] = []
    self.unfinished_error_messages: list[str] = []
    self.stopped = False
    self._lock = threading.Lock()

  def start(
    self,
    *,
    llm_ids: list[str],
    population_size: int,
    section_count: int,
  ) -> None:
    with self._lock:
      self.starts.append(
        QueryProgressSnapshot.empty(
          llm_ids=llm_ids,
          population_size=population_size,
          section_count=section_count,
        )
      )

  def mark_section_started(
    self,
    llm_id: str,
    population_index: int,
    section_index: int,
  ) -> None:
    with self._lock:
      self.started_sections.append((llm_id, population_index, section_index))

  def mark_section_completed(
    self,
    llm_id: str,
    population_index: int,
    section_index: int,
  ) -> None:
    with self._lock:
      self.completed_sections.append((llm_id, population_index, section_index))

  def mark_section_error(
    self,
    llm_id: str,
    population_index: int,
    section_index: int,
  ) -> None:
    with self._lock:
      self.error_sections.append((llm_id, population_index, section_index))

  def mark_unfinished_error(self, message: str) -> None:
    with self._lock:
      self.unfinished_error_messages.append(message)

  def stop(self) -> None:
    with self._lock:
      self.stopped = True

  def started_count(self) -> int:
    with self._lock:
      return len(self.started_sections)


class BlockingProvider(BaseProviderClient):
  """Provider blocking the first request so progress state can be inspected."""

  def __init__(
    self,
    config: LLMConnectorConfig,
    first_request_started: threading.Event,
    release_first_request: threading.Event,
  ) -> None:
    super().__init__(config)
    self._first_request_started = first_request_started
    self._release_first_request = release_first_request

  def _generate(
    self,
    messages: list[dict[str, str]],
    *,
    output_schema,
  ) -> ProviderResponse:
    if not self._first_request_started.is_set():
      self._first_request_started.set()
      self._release_first_request.wait(timeout=5)
    return ProviderResponse(
      content=json.dumps({"answer": 5}),
      raw={"messages": messages},
      metadata={},
    )


class AlwaysBlockingProvider(BaseProviderClient):
  """Provider blocking every request so scheduler order can be inspected."""

  def __init__(
    self,
    config: LLMConnectorConfig,
    release_requests: threading.Event,
  ) -> None:
    super().__init__(config)
    self._release_requests = release_requests

  def _generate(
    self,
    messages: list[dict[str, str]],
    *,
    output_schema,
  ) -> ProviderResponse:
    self._release_requests.wait(timeout=5)
    return ProviderResponse(
      content=json.dumps({"answer": 5}),
      raw={"messages": messages},
      metadata={},
    )


class AlwaysBlockingConversationFactory:
  """Factory creating conversations that block until tests release them."""

  def __init__(self, release_requests: threading.Event) -> None:
    self._release_requests = release_requests

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation:
    config = LLMConnectorConfig(
      provider=ProviderType.OPENAI,
      model=model,
      api_key="test-key",
      system_prompt=system_prompt,
      response_format=ResponseFormat.JSON,
    )
    return LLMConversation(
      config=config,
      provider_client=AlwaysBlockingProvider(config, self._release_requests),
      system_prompt=system_prompt,
    )


class BlockingConversationFactory:
  """Factory creating blocking conversations for concurrency progress tests."""

  def __init__(
    self,
    first_request_started: threading.Event,
    release_first_request: threading.Event,
  ) -> None:
    self._first_request_started = first_request_started
    self._release_first_request = release_first_request

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation:
    config = LLMConnectorConfig(
      provider=ProviderType.OPENAI,
      model=model,
      api_key="test-key",
      system_prompt=system_prompt,
      response_format=ResponseFormat.JSON,
    )
    return LLMConversation(
      config=config,
      provider_client=BlockingProvider(
        config,
        self._first_request_started,
        self._release_first_request,
      ),
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


def build_single_rating_questionnaire() -> Questionnaire:
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
  section = Section(
    name="General",
    instructions="Answer each question carefully.",
    questions=[rating_question],
  )
  return Questionnaire(
    id="single-rating-questionnaire",
    name="Single Rating Questionnaire",
    description=None,
    version=1,
    metadata={"default_population": 1},
    default_population=1,
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


def build_multi_question_section_questionnaire() -> Questionnaire:
  return Questionnaire(
    id="multi-question-sections",
    name="Multi Question Sections",
    description=None,
    version=1,
    metadata={"default_population": 1},
    default_population=1,
    system_prompt="System prompt",
    sections=[
      Section(
        name="First",
        instructions="Use first section instructions.",
        questions=[
          Question(
            id="first-1",
            type=QuestionType.RATING_5,
            prompt="First section question one.",
            options=None,
            scoring=ScoringRule(total=5, weights={"5": 5}),
          ),
          Question(
            id="first-2",
            type=QuestionType.RATING_5,
            prompt="First section question two.",
            options=None,
            scoring=ScoringRule(total=5, weights={"4": 4}),
          ),
        ],
      ),
      Section(
        name="Second",
        instructions="Use second section instructions.",
        questions=[
          Question(
            id="second-1",
            type=QuestionType.RATING_5,
            prompt="Second section question one.",
            options=None,
            scoring=ScoringRule(total=5, weights={"3": 3}),
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


def test_runner_writes_split_jsonl_records(tmp_path: Path) -> None:
  questionnaire = build_questionnaire()
  output_dir = tmp_path / "run-output"
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
    output_path=output_dir,
  )

  assert result.errors == []
  assert len(result.records) == 1
  records = read_jsonl(output_dir / "responses.jsonl")
  metadata_records = read_jsonl(output_dir / "metadata.jsonl")
  assert len(records) == 1
  assert len(metadata_records) == 1
  record = records[0]
  metadata_record = metadata_records[0]
  assert record["questionnaire"]["name"] == "test-questionnaire"
  assert metadata_record["questionnaire"] == record["questionnaire"]
  assert record["llm_id"] == "openai/gpt-4"
  assert metadata_record["llm_id"] == record["llm_id"]
  assert record["population_index"] == 0
  assert metadata_record["population_index"] == record["population_index"]
  assert record["query_time"]
  assert metadata_record["query_time"] == record["query_time"]
  assert "summary" not in record
  assert "score" not in json.dumps(record)
  assert record["response"]["sections"][0]["id"] == "General"
  questions = record["response"]["sections"][0]["questions"]
  assert [question["id"] for question in questions] == ["rating-1", "choice-1"]
  assert questions[0]["response"] == "5"
  assert questions[1]["response"] == "a"
  attempts = metadata_record["metadata"]["sections"][0]["questions"][0][
    "attempts"
  ]
  assert attempts[0]["raw_response"] == json.dumps(
    {"answer": 5, "reasoning": "Helpful."}
  )
  assert attempts[0]["canonical_response"] == "5"
  assert attempts[0]["provider_metadata"]["request_index"] == 1


def test_runner_default_output_dir_includes_llm_profile() -> None:
  questionnaire = build_questionnaire()
  factory = StubConversationFactory(
    {
      "openai/gpt-4": [
        json.dumps({"answer": 5}),
        json.dumps({"answer": "a"}),
      ]
    },
  )
  runner = BenchmarkRunner(factory, max_concurrency=1)

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4"],
    llm_profile="default-llms",
  )

  assert result.output_dir is not None
  assert result.output_dir.parent == Path("results")
  assert result.output_dir.name.startswith("test-questionnaire-default-llms-")
  assert (result.output_dir / "responses.jsonl").is_file()
  assert (result.output_dir / "metadata.jsonl").is_file()


def test_runner_retries_when_canonicalization_fails(tmp_path: Path) -> None:
  questionnaire = build_questionnaire()
  output_dir = tmp_path / "retry-output"
  factory = StubConversationFactory(
    {
      "openai/gpt-4": [
        json.dumps({"answer": 5}),
        json.dumps({"answer": "definitely not an option"}),
        json.dumps({"answer": "Option A"}),
      ]
    },
  )
  runner = BenchmarkRunner(factory, max_concurrency=1)

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4"],
    output_path=output_dir,
  )

  assert result.errors == []
  record = read_jsonl(output_dir / "responses.jsonl")[0]
  metadata_record = read_jsonl(output_dir / "metadata.jsonl")[0]
  questions = record["response"]["sections"][0]["questions"]
  assert questions[1]["response"] == "a"
  choice_attempts = metadata_record["metadata"]["sections"][0]["questions"][1][
    "attempts"
  ]
  assert [attempt["canonical_response"] for attempt in choice_attempts] == [
    None,
    "a",
  ]
  assert "not allowed" in choice_attempts[0]["validation_error"]


def test_runner_retries_timeouts_before_recording_error(tmp_path: Path) -> None:
  questionnaire = build_single_rating_questionnaire()
  output_dir = tmp_path / "timeout-retry-output"
  factory = TimeoutThenSuccessConversationFactory()
  runner = BenchmarkRunner(factory, max_concurrency=1)

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4"],
    output_path=output_dir,
  )

  assert result.errors == []
  assert len(factory.providers) == 1
  assert factory.providers[0].calls == 2
  record = read_jsonl(output_dir / "responses.jsonl")[0]
  assert record["response"]["sections"][0]["questions"][0]["response"] == "5"
  assert record["response"]["sections"][0]["questions"][0]["errors"] == []


def test_runner_records_retry_count_when_timeouts_are_exhausted(
  tmp_path: Path,
) -> None:
  questionnaire = build_single_rating_questionnaire()
  output_dir = tmp_path / "timeout-error-output"
  factory = AlwaysTimeoutConversationFactory()
  runner = BenchmarkRunner(factory, max_concurrency=1)

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4"],
    output_path=output_dir,
  )

  assert len(factory.providers) == 1
  assert factory.providers[0].calls == 3
  assert len(result.errors) == 1
  assert result.errors[0].retry_count == 2
  record = read_jsonl(output_dir / "responses.jsonl")[0]
  question_error = record["response"]["sections"][0]["questions"][0]["errors"][0]
  assert question_error["message"] == "The read operation timed out"
  assert question_error["retry_count"] == 2


def test_runner_records_nonretryable_provider_errors(
  tmp_path: Path,
) -> None:
  questionnaire = build_single_rating_questionnaire()
  output_dir = tmp_path / "provider-error-output"
  runner = BenchmarkRunner(
    ProviderErrorConversationFactory(),
    max_concurrency=1,
  )

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4"],
    output_path=output_dir,
  )

  assert len(result.errors) == 1
  assert result.errors[0].stage == "provider"
  assert result.errors[0].message == "[openai] Model access denied."
  record = read_jsonl(output_dir / "responses.jsonl")[0]
  question_error = record["response"]["sections"][0]["questions"][0]["errors"][0]
  assert question_error["stage"] == "provider"


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


def test_sections_start_clean_and_later_questions_keep_section_history() -> None:
  questionnaire = build_multi_question_section_questionnaire()
  factory = StubConversationFactory(
    {
      "openai/gpt-4": [
        json.dumps({"answer": 5}),
        json.dumps({"answer": 4}),
        json.dumps({"answer": 3}),
      ]
    },
  )
  runner = BenchmarkRunner(factory, max_concurrency=1)

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4"],
  )

  assert result.errors == []
  requests = factory.requests_by_llm_id["openai/gpt-4"]
  assert len(requests) == 3

  first_section_prompt_one = requests[0][1]["content"]
  first_section_prompt_two = next(
    request[-1]["content"]
    for request in requests
    if "First section question two." in request[-1]["content"]
  )
  second_section_prompt_one = next(
    request[1]["content"]
    for request in requests
    if "Second section question one." in request[1]["content"]
  )

  assert "Use first section instructions." in first_section_prompt_one
  assert "First section question one." in first_section_prompt_one
  assert "Respond with JSON that matches this schema exactly" in first_section_prompt_one
  assert '"title": "first-1"' in first_section_prompt_one
  assert "Use second section instructions." in second_section_prompt_one
  assert '"title": "second-1"' in second_section_prompt_one

  request_signatures = [
    [(message["role"], message["content"]) for message in request]
    for request in requests
  ]

  assert [
    ("system", "System prompt"),
    ("user", first_section_prompt_one),
  ] in request_signatures
  assert [
    ("system", "System prompt"),
    ("user", second_section_prompt_one),
  ] in request_signatures
  assert [
    ("system", "System prompt"),
    ("user", first_section_prompt_one),
    ("assistant", json.dumps({"answer": 5})),
    ("user", first_section_prompt_two),
  ] in request_signatures


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


def test_runner_updates_progress_display_for_sections() -> None:
  questionnaire = build_two_section_questionnaire()
  display = RecordingProgressDisplay()
  factory = StubConversationFactory(
    {
      "openai/gpt-4": [json.dumps({"answer": 5}), json.dumps({"answer": 4})],
    },
  )
  runner = BenchmarkRunner(factory, max_concurrency=2, progress_display=display)

  runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4"],
    total_population=1,
  )

  assert len(display.starts) == 1
  assert display.starts[0].llm_ids == ["openai/gpt-4"]
  assert display.starts[0].population_size == 1
  assert display.starts[0].section_count == 2
  assert set(display.started_sections) == {
    ("openai/gpt-4", 0, 0),
    ("openai/gpt-4", 0, 1),
  }
  assert set(display.completed_sections) == {
    ("openai/gpt-4", 0, 0),
    ("openai/gpt-4", 0, 1),
  }
  assert display.stopped is True


def test_progress_marks_only_queries_that_are_in_flight() -> None:
  questionnaire = build_two_section_questionnaire()
  display = RecordingProgressDisplay()
  first_request_started = threading.Event()
  release_first_request = threading.Event()
  runner = BenchmarkRunner(
    BlockingConversationFactory(first_request_started, release_first_request),
    max_concurrency=1,
    progress_display=display,
  )
  thread_errors: list[BaseException] = []

  def run_runner() -> None:
    try:
      runner.run_sync(
        questionnaire=questionnaire,
        llm_ids=["openai/gpt-4"],
        total_population=1,
      )
    except BaseException as exc:
      thread_errors.append(exc)

  thread = threading.Thread(target=run_runner)
  thread.start()
  try:
    assert first_request_started.wait(timeout=5)
    time.sleep(0.1)
    assert display.started_count() == 1
  finally:
    release_first_request.set()
    thread.join(timeout=5)

  assert not thread.is_alive()
  assert thread_errors == []


def test_runner_prioritizes_population_then_sections_before_llms() -> None:
  questionnaire = build_two_section_questionnaire()
  display = RecordingProgressDisplay()
  release_requests = threading.Event()
  runner = BenchmarkRunner(
    AlwaysBlockingConversationFactory(release_requests),
    max_concurrency=6,
    progress_display=display,
  )
  llm_ids = ["openai/gpt-4", "anthropic/claude", "gemini/pro"]
  thread_errors: list[BaseException] = []

  def run_runner() -> None:
    try:
      runner.run_sync(
        questionnaire=questionnaire,
        llm_ids=llm_ids,
        total_population=2,
      )
    except BaseException as exc:
      thread_errors.append(exc)

  thread = threading.Thread(target=run_runner)
  thread.start()
  try:
    deadline = time.time() + 5
    while display.started_count() < 6 and time.time() < deadline:
      time.sleep(0.01)

    assert display.started_count() == 6
    assert set(display.started_sections) == {
      (llm_id, 0, section_index)
      for section_index in range(2)
      for llm_id in llm_ids
    }
  finally:
    release_requests.set()
    thread.join(timeout=5)

  assert not thread.is_alive()
  assert thread_errors == []


def test_progress_marks_sections_with_response_errors_as_error() -> None:
  questionnaire = build_two_section_questionnaire()
  display = RecordingProgressDisplay()
  runner = BenchmarkRunner(
    ErrorConversationFactory(),
    max_concurrency=2,
    progress_display=display,
  )

  result = runner.run_sync(
    questionnaire=questionnaire,
    llm_ids=["openai/gpt-4"],
    total_population=1,
  )

  assert result.errors
  assert display.completed_sections == []
  assert set(display.error_sections) == {
    ("openai/gpt-4", 0, 0),
    ("openai/gpt-4", 0, 1),
  }


def test_progress_marks_unfinished_sections_as_error_on_keyboard_interrupt(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  questionnaire = build_two_section_questionnaire()
  display = RecordingProgressDisplay()
  runner = BenchmarkRunner(
    StubConversationFactory({}),
    max_concurrency=2,
    progress_display=display,
  )

  def raise_keyboard_interrupt(coro: Any) -> None:
    coro.close()
    raise KeyboardInterrupt

  monkeypatch.setattr(
    "rationale_benchmark.runner.executor.asyncio.run",
    raise_keyboard_interrupt,
  )

  with pytest.raises(KeyboardInterrupt):
    runner.run_sync(
      questionnaire=questionnaire,
      llm_ids=["openai/gpt-4"],
      total_population=1,
    )

  assert display.error_sections == []
  assert display.unfinished_error_messages == [
    (
      "Interrupted: all remaining populations and sections are marked as "
      "error because they were not queried."
    )
  ]
  assert display.stopped is True
