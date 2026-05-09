"""Asynchronous execution orchestrator for raw questionnaire runs."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from rationale_benchmark.llm.conversation import LLMConversation, LLMResponse
from rationale_benchmark.llm.exceptions import (
  ConfigurationError,
  ConversationArchivedError,
  LLMConnectorError,
  ProviderError,
  ValidationFailedError,
)
from rationale_benchmark.questionnaire.errors import AnswerValidationError
from rationale_benchmark.questionnaire.models import (
  Question,
  Questionnaire,
  QuestionType,
  Section,
)
from rationale_benchmark.questionnaire.output_schema import resolve_output_schema
from rationale_benchmark.questionnaire.scoring import validate_answer
from rationale_benchmark.runner.progress_display import QueryProgressDisplayProtocol
from rationale_benchmark.runner.prompts import build_prompt
from rationale_benchmark.runner.types import (
  MetadataRecord,
  OutputRecordPair,
  PromptContext,
  QuestionAnswer,
  QuestionAttemptTrace,
  RawRunResult,
  ResponseRecord,
  RunnerError,
  now_utc,
)

INTERRUPTED_PROGRESS_MESSAGE = (
  "Interrupted: all remaining populations and sections are marked as error "
  "because they were not queried."
)


class ConversationFactoryProtocol(Protocol):
  """Protocol describing conversation factories consumed by the runner."""

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation: ...


class RunnerConfigError(RuntimeError):
  """Raised when runner configuration is invalid."""


class BenchmarkRunner:
  """Coordinate raw questionnaire execution for one questionnaire."""

  def __init__(
    self,
    conversation_factory: ConversationFactoryProtocol,
    *,
    max_concurrency: int = 5,
    progress_display: QueryProgressDisplayProtocol | None = None,
  ) -> None:
    if max_concurrency < 1:
      raise RunnerConfigError("max_concurrency must be >= 1")
    self._conversation_factory = conversation_factory
    self._max_concurrency = max_concurrency
    self._progress_display = progress_display

  async def run(
    self,
    questionnaire: Questionnaire | None = None,
    llm_ids: Sequence[str] | None = None,
    *,
    questionnaires: Sequence[Questionnaire] | None = None,
    total_population: int | None = None,
    output_path: Path | str | None = None,
    questionnaire_path: Path | str | None = None,
    llm_profile: str | None = None,
  ) -> RawRunResult:
    """Execute one questionnaire and return raw response records."""

    questionnaire = self._resolve_single_questionnaire(
      questionnaire,
      questionnaires,
    )
    if not llm_ids:
      raise RunnerConfigError("At least one LLM ID must be provided")
    if total_population is not None and total_population < 1:
      raise RunnerConfigError("total_population must be >= 1")

    population = total_population or questionnaire.default_population
    if population < 1:
      raise RunnerConfigError("effective population must be >= 1")

    destination = self._resolve_output_dir(
      questionnaire,
      output_path,
      llm_profile=llm_profile,
    )
    source_path = (
      Path(questionnaire_path) if questionnaire_path is not None else None
    )
    self._validate_output_dir(destination)

    provider_semaphore = asyncio.Semaphore(self._max_concurrency)
    write_lock = asyncio.Lock()
    if self._progress_display is not None:
      self._progress_display.start(
        llm_ids=list(llm_ids),
        population_size=population,
        section_count=len(questionnaire.sections),
      )

    try:
      records: list[ResponseRecord] = []
      metadata_records: list[MetadataRecord] = []
      errors: list[RunnerError] = []
      record_state: dict[tuple[str, int], dict[str, Any]] = {
        (llm_id, population_index): {
          "response_sections": [None] * len(questionnaire.sections),
          "metadata_sections": [None] * len(questionnaire.sections),
          "errors": [],
          "query_times": [],
          "completed_sections": 0,
        }
        for llm_id in llm_ids
        for population_index in range(population)
      }
      tasks = [
        asyncio.create_task(
          self._execute_section_job(
            questionnaire=questionnaire,
            section=section,
            section_index=section_index,
            llm_id=llm_id,
            population_index=population_index,
            provider_semaphore=provider_semaphore,
          )
        )
        for population_index in range(population)
        for section_index, section in enumerate(questionnaire.sections)
        for llm_id in llm_ids
      ]

      for completed_task in asyncio.as_completed(tasks):
        try:
          (
            llm_id,
            population_index,
            section_index,
            response_section,
            metadata_section,
            section_errors,
            first_query_time,
          ) = await completed_task
        except Exception as exc:
          error = RunnerError(
            llm_id="",
            stage="runtime",
            message=str(exc),
          )
          errors.append(error)
          continue

        state = record_state[(llm_id, population_index)]
        state["response_sections"][section_index] = response_section
        state["metadata_sections"][section_index] = metadata_section
        state["errors"].extend(section_errors)
        if first_query_time is not None:
          state["query_times"].append(first_query_time)
        state["completed_sections"] += 1
        if state["completed_sections"] != len(questionnaire.sections):
          continue

        query_times = state["query_times"]
        query_time = min(query_times) if query_times else now_utc()
        response_record = ResponseRecord(
          questionnaire_name=questionnaire.id,
          questionnaire_path=source_path,
          llm_id=llm_id,
          population_index=population_index,
          query_time=query_time,
          response={"sections": state["response_sections"]},
          errors=state["errors"],
        )
        metadata_record = MetadataRecord(
          questionnaire_name=questionnaire.id,
          questionnaire_path=source_path,
          llm_id=llm_id,
          population_index=population_index,
          query_time=query_time,
          metadata={"sections": state["metadata_sections"]},
          errors=state["errors"],
        )
        await self._append_jsonl_pair(
          OutputRecordPair(
            response_record=response_record,
            metadata_record=metadata_record,
          ),
          destination,
          write_lock,
        )
        records.append(response_record)
        metadata_records.append(metadata_record)
        errors.extend(response_record.errors)
    except asyncio.CancelledError:
      if self._progress_display is not None:
        self._progress_display.mark_unfinished_error(INTERRUPTED_PROGRESS_MESSAGE)
      raise
    finally:
      if self._progress_display is not None:
        self._progress_display.stop()

    records.sort(key=lambda record: (record.llm_id, record.population_index))
    metadata_records.sort(key=lambda record: (record.llm_id, record.population_index))
    return RawRunResult(
      records=records,
      metadata_records=metadata_records,
      output_dir=destination,
      errors=errors,
    )

  def run_sync(
    self,
    questionnaire: Questionnaire | None = None,
    llm_ids: Sequence[str] | None = None,
    *,
    questionnaires: Sequence[Questionnaire] | None = None,
    total_population: int | None = None,
    output_path: Path | str | None = None,
    questionnaire_path: Path | str | None = None,
    llm_profile: str | None = None,
  ) -> RawRunResult:
    """Synchronous wrapper executing the runner in a new event loop."""

    try:
      return asyncio.run(
        self.run(
          questionnaire=questionnaire,
          llm_ids=llm_ids,
          questionnaires=questionnaires,
          total_population=total_population,
          output_path=output_path,
          questionnaire_path=questionnaire_path,
          llm_profile=llm_profile,
        )
      )
    except KeyboardInterrupt:
      if self._progress_display is not None:
        self._progress_display.mark_unfinished_error(INTERRUPTED_PROGRESS_MESSAGE)
        self._progress_display.stop()
      raise

  def _resolve_single_questionnaire(
    self,
    questionnaire: Questionnaire | None,
    questionnaires: Sequence[Questionnaire] | None,
  ) -> Questionnaire:
    if questionnaire is not None and questionnaires is not None:
      raise RunnerConfigError("Provide exactly one questionnaire")
    if questionnaire is not None:
      return questionnaire
    if questionnaires is None:
      raise RunnerConfigError("Provide exactly one questionnaire")
    if len(questionnaires) != 1:
      raise RunnerConfigError("Runner execution accepts exactly one questionnaire")
    return questionnaires[0]

  def _resolve_output_dir(
    self,
    questionnaire: Questionnaire,
    output_path: Path | str | None,
    *,
    llm_profile: str | None,
  ) -> Path:
    if output_path is not None:
      return Path(output_path)
    timestamp = now_utc().strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    profile_slug = self._slugify_llm_profile(llm_profile)
    return Path("results") / f"{questionnaire.id}-{profile_slug}-{timestamp}"

  def _slugify_llm_profile(self, llm_profile: str | None) -> str:
    profile = llm_profile or "llm"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", profile).strip("-").lower()
    return slug or "llm"

  def _validate_output_dir(self, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "responses.jsonl").open("a", encoding="utf-8"):
      pass
    with (output_dir / "metadata.jsonl").open("a", encoding="utf-8"):
      pass

  async def _execute_section_job(
    self,
    *,
    questionnaire: Questionnaire,
    section: Section,
    section_index: int,
    llm_id: str,
    population_index: int,
    provider_semaphore: asyncio.Semaphore,
  ) -> tuple[
    str,
    int,
    int,
    dict[str, Any],
    dict[str, Any],
    list[RunnerError],
    Any,
  ]:
    response_section, metadata_section, section_errors, first_query_time = (
      await self._execute_section_with_progress(
        questionnaire=questionnaire,
        section=section,
        section_index=section_index,
        llm_id=llm_id,
        population_index=population_index,
        provider_semaphore=provider_semaphore,
      )
    )
    return (
      llm_id,
      population_index,
      section_index,
      response_section,
      metadata_section,
      section_errors,
      first_query_time,
    )

  async def _execute_section(
    self,
    *,
    questionnaire: Questionnaire,
    section: Section,
    section_index: int,
    llm_id: str,
    population_index: int,
    provider_semaphore: asyncio.Semaphore,
  ) -> tuple[dict[str, Any], dict[str, Any], list[RunnerError], Any]:
    section_errors: list[RunnerError] = []
    question_entries: list[dict[str, Any]] = []
    metadata_question_entries: list[dict[str, Any]] = []
    prior_answers: list[QuestionAnswer] = []
    first_query_time = None

    try:
      conversation = self._conversation_factory.create(
        llm_id,
        system_prompt=questionnaire.system_prompt,
      )
    except ConfigurationError as exc:
      error = RunnerError(
        llm_id=llm_id,
        questionnaire_id=questionnaire.id,
        population_index=population_index,
        section_id=section.name,
        stage="configuration",
        message=str(exc),
      )
      section_errors.append(error)
      return (
        {"id": section.name, "questions": [
          {
            "id": question.id,
            "response": None,
            "errors": [error.to_json_dict()],
          }
          for question in section.questions
        ]},
        {
          "id": section.name,
          "questions": [
            {
              "id": question.id,
              "attempts": [],
              "errors": [error.to_json_dict()],
            }
            for question in section.questions
          ],
        },
        section_errors,
        first_query_time,
      )

    for question in section.questions:
      prompt_context = PromptContext(
        questionnaire_id=questionnaire.id,
        section_name=section.name,
        section_instructions=section.instructions,
        question=question,
        system_prompt=questionnaire.system_prompt,
        prior_answers=list(prior_answers),
      )
      prompt = build_prompt(prompt_context)
      question_errors: list[RunnerError] = []
      attempts: list[QuestionAttemptTrace] = []
      attempts_before = conversation.provider_attempts
      try:
        canonical_response, response, query_time = await self._ask_question(
          conversation,
          prompt,
          provider_semaphore,
          question=question,
          attempts=attempts,
          llm_id=llm_id,
          population_index=population_index,
          section_index=section_index,
        )
      except ValidationFailedError as exc:
        retry_count = self._retry_count_for_question(
          conversation,
          attempts_before,
        )
        error = self._question_error(
          llm_id,
          questionnaire.id,
          population_index,
          section.name,
          question.id,
          "validation",
          str(exc),
          retry_count=retry_count,
        )
        canonical_response = None
        question_errors.append(error)
        section_errors.append(error)
      except ConversationArchivedError as exc:
        error = self._question_error(
          llm_id,
          questionnaire.id,
          population_index,
          section.name,
          question.id,
          "conversation",
          str(exc),
        )
        canonical_response = None
        question_errors.append(error)
        section_errors.append(error)
      except LLMConnectorError as exc:
        retry_count = self._retry_count_for_question(
          conversation,
          attempts_before,
        )
        error = self._question_error(
          llm_id,
          questionnaire.id,
          population_index,
          section.name,
          question.id,
          "network",
          str(exc),
          retry_count=retry_count,
        )
        canonical_response = None
        question_errors.append(error)
        section_errors.append(error)
      except ProviderError as exc:
        retry_count = self._retry_count_for_question(
          conversation,
          attempts_before,
        )
        error = self._question_error(
          llm_id,
          questionnaire.id,
          population_index,
          section.name,
          question.id,
          "provider",
          str(exc),
          retry_count=retry_count,
        )
        canonical_response = None
        question_errors.append(error)
        section_errors.append(error)
      except Exception as exc:  # pragma: no cover - defensive
        retry_count = self._retry_count_for_question(
          conversation,
          attempts_before,
        )
        error = self._question_error(
          llm_id,
          questionnaire.id,
          population_index,
          section.name,
          question.id,
          "runtime",
          str(exc),
          retry_count=retry_count,
        )
        canonical_response = None
        question_errors.append(error)
        section_errors.append(error)
      else:
        if canonical_response is None:
          raise RuntimeError("Accepted response missing canonical answer")
        if first_query_time is None:
          first_query_time = query_time
        prior_answers.append(
          QuestionAnswer(
            question_id=question.id,
            prompt=prompt,
            response_text=canonical_response,
          )
        )

      question_entries.append(
        {
          "id": question.id,
          "response": canonical_response,
          "errors": [error.to_json_dict() for error in question_errors],
        }
      )
      metadata_question_entries.append(
        {
          "id": question.id,
          "attempts": [attempt.to_json_dict() for attempt in attempts],
          "errors": [error.to_json_dict() for error in question_errors],
        }
      )

    return (
      {"id": section.name, "questions": question_entries},
      {"id": section.name, "questions": metadata_question_entries},
      section_errors,
      first_query_time,
    )

  async def _execute_section_with_progress(
    self,
    *,
    questionnaire: Questionnaire,
    section: Section,
    section_index: int,
    llm_id: str,
    population_index: int,
    provider_semaphore: asyncio.Semaphore,
  ) -> tuple[dict[str, Any], dict[str, Any], list[RunnerError], Any]:
    result = await self._execute_section(
      questionnaire=questionnaire,
      section=section,
      section_index=section_index,
      llm_id=llm_id,
      population_index=population_index,
      provider_semaphore=provider_semaphore,
    )
    _response_section, _metadata_section, section_errors, _first_query_time = result
    if self._progress_display is not None:
      if section_errors:
        self._progress_display.mark_section_error(
          llm_id,
          population_index,
          section_index,
        )
      else:
        self._progress_display.mark_section_completed(
          llm_id,
          population_index,
          section_index,
        )
    return result

  def _question_error(
    self,
    llm_id: str,
    questionnaire_id: str,
    population_index: int,
    section_id: str,
    question_id: str,
    stage: str,
    message: str,
    *,
    retry_count: int | None = None,
  ) -> RunnerError:
    return RunnerError(
      llm_id=llm_id,
      questionnaire_id=questionnaire_id,
      population_index=population_index,
      section_id=section_id,
      question_id=question_id,
      stage=stage,
      message=message,
      retry_count=retry_count,
    )

  async def _ask_question(
    self,
    conversation: LLMConversation,
    prompt: str,
    provider_semaphore: asyncio.Semaphore,
    *,
    question: Question,
    attempts: list[QuestionAttemptTrace],
    llm_id: str,
    population_index: int,
    section_index: int,
  ) -> tuple[str, LLMResponse, Any]:
    canonical_response: str | None = None

    def validator(response: LLMResponse) -> bool:
      nonlocal canonical_response
      attempt_number = len(attempts) + 1
      provider_metadata = self._provider_metadata(response)
      try:
        canonical_response = self._canonicalize_response(question, response)
      except AnswerValidationError as exc:
        attempts.append(
          QuestionAttemptTrace(
            attempt=attempt_number,
            raw_response=response,
            canonical_response=None,
            validation_error=str(exc),
            provider_metadata=provider_metadata,
          )
        )
        return False
      attempts.append(
        QuestionAttemptTrace(
          attempt=attempt_number,
          raw_response=response,
          canonical_response=canonical_response,
          validation_error=None,
          provider_metadata=provider_metadata,
        )
      )
      return True

    async with provider_semaphore:
      if self._progress_display is not None:
        self._progress_display.mark_section_started(
          llm_id,
          population_index,
          section_index,
        )
      query_time = now_utc()
      response = await asyncio.to_thread(
        conversation.ask,
        prompt,
        resolve_output_schema(question),
        validator,
      )
    if canonical_response is None:
      canonical_response = self._canonicalize_response(question, response)
    return canonical_response, response, query_time

  def _retry_count_for_question(
    self,
    conversation: LLMConversation,
    attempts_before: int,
  ) -> int:
    attempts_used = conversation.provider_attempts - attempts_before
    return max(0, attempts_used - 1)

  def _canonicalize_response(
    self,
    question: Question,
    response: LLMResponse,
  ) -> str:
    answer = self._extract_answer(response)
    if question.type is QuestionType.CHOICE:
      answer = self._canonicalize_choice_answer(question, answer)
    elif isinstance(answer, str) and answer.strip().isdigit():
      answer = int(answer.strip())
    return validate_answer(question, answer)

  def _extract_answer(self, response: LLMResponse) -> Any:
    if isinstance(response.parsed, dict) and "answer" in response.parsed:
      return response.parsed["answer"]
    if isinstance(response.parsed, str):
      try:
        decoded = json.loads(response.parsed)
      except json.JSONDecodeError:
        return response.parsed
      if isinstance(decoded, dict) and "answer" in decoded:
        return decoded["answer"]
      return decoded
    if response.parsed is not None and not isinstance(response.parsed, dict):
      return response.parsed
    return response.text

  def _canonicalize_choice_answer(self, question: Question, answer: Any) -> Any:
    if not isinstance(answer, str):
      return answer
    if question.options is None or answer in question.options:
      return answer

    normalized_answer = self._normalize_choice_text(answer)
    for key, label in question.options.items():
      if normalized_answer == self._normalize_choice_text(label):
        return key
      if normalized_answer == self._normalize_choice_text(key):
        return key

    key_matches = [
      key
      for key in question.options
      if re.search(
        rf"(?<![A-Za-z0-9]){re.escape(key)}(?![A-Za-z0-9])",
        answer,
        flags=re.IGNORECASE,
      )
    ]
    if len(key_matches) == 1:
      return key_matches[0]
    return answer

  def _normalize_choice_text(self, value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"\s+", " ", lowered.strip(" \t\r\n.:;,-_!()[]{}"))

  def _provider_metadata(self, response: LLMResponse) -> dict[str, Any]:
    return {
      "provider": response.provider,
      "model": response.model,
      **response.metadata,
    }

  async def _append_jsonl_pair(
    self,
    record_pair: OutputRecordPair,
    output_dir: Path,
    write_lock: asyncio.Lock,
  ) -> None:
    response_line = json.dumps(
      record_pair.response_record.to_json_dict(),
      ensure_ascii=False,
    )
    metadata_line = json.dumps(
      record_pair.metadata_record.to_json_dict(),
      ensure_ascii=False,
    )
    async with write_lock:
      await asyncio.to_thread(
        self._append_lines,
        output_dir / "responses.jsonl",
        response_line,
        output_dir / "metadata.jsonl",
        metadata_line,
      )

  def _append_lines(
    self,
    response_path: Path,
    response_line: str,
    metadata_path: Path,
    metadata_line: str,
  ) -> None:
    with response_path.open("a", encoding="utf-8") as response_handle:
      response_handle.write(response_line + "\n")
    with metadata_path.open("a", encoding="utf-8") as metadata_handle:
      metadata_handle.write(metadata_line + "\n")


__all__ = ["BenchmarkRunner", "RunnerConfigError"]
