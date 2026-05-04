"""Asynchronous execution orchestrator for raw questionnaire runs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from rationale_benchmark.llm.conversation import LLMConversation, LLMResponse
from rationale_benchmark.llm.exceptions import (
  ConfigurationError,
  ConversationArchivedError,
  LLMConnectorError,
  ValidationFailedError,
)
from rationale_benchmark.questionnaire.models import Questionnaire, Section
from rationale_benchmark.runner.progress_display import QueryProgressDisplayProtocol
from rationale_benchmark.runner.prompts import build_prompt
from rationale_benchmark.runner.types import (
  PromptContext,
  QuestionAnswer,
  RawResponseRecord,
  RawRunResult,
  RunnerError,
  now_utc,
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

    destination = Path(output_path) if output_path is not None else None
    source_path = (
      Path(questionnaire_path) if questionnaire_path is not None else None
    )
    if destination is not None:
      self._validate_output_path(destination)

    provider_semaphore = asyncio.Semaphore(self._max_concurrency)
    write_lock = asyncio.Lock()
    if self._progress_display is not None:
      self._progress_display.start(
        llm_ids=list(llm_ids),
        population_size=population,
        section_count=len(questionnaire.sections),
      )

    try:
      tasks = [
        asyncio.create_task(
          self._execute_questionnaire(
            questionnaire=questionnaire,
            questionnaire_path=source_path,
            llm_id=llm_id,
            population_index=population_index,
            provider_semaphore=provider_semaphore,
            output_path=destination,
            write_lock=write_lock,
          )
        )
        for llm_id in llm_ids
        for population_index in range(population)
      ]

      records: list[RawResponseRecord] = []
      errors: list[RunnerError] = []
      for outcome in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(outcome, Exception):
          error = RunnerError(
            llm_id="",
            stage="runtime",
            message=str(outcome),
          )
          errors.append(error)
          continue
        records.append(outcome)
        errors.extend(outcome.errors)
    finally:
      if self._progress_display is not None:
        self._progress_display.stop()

    records.sort(key=lambda record: (record.llm_id, record.population_index))
    return RawRunResult(records=records, errors=errors)

  def run_sync(
    self,
    questionnaire: Questionnaire | None = None,
    llm_ids: Sequence[str] | None = None,
    *,
    questionnaires: Sequence[Questionnaire] | None = None,
    total_population: int | None = None,
    output_path: Path | str | None = None,
    questionnaire_path: Path | str | None = None,
  ) -> RawRunResult:
    """Synchronous wrapper executing the runner in a new event loop."""

    return asyncio.run(
      self.run(
        questionnaire=questionnaire,
        llm_ids=llm_ids,
        questionnaires=questionnaires,
        total_population=total_population,
        output_path=output_path,
        questionnaire_path=questionnaire_path,
      )
    )

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

  def _validate_output_path(self, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8"):
      pass

  async def _execute_questionnaire(
    self,
    *,
    questionnaire: Questionnaire,
    questionnaire_path: Path | None,
    llm_id: str,
    population_index: int,
    provider_semaphore: asyncio.Semaphore,
    output_path: Path | None,
    write_lock: asyncio.Lock,
  ) -> RawResponseRecord:
    section_tasks = [
      asyncio.create_task(
        self._execute_section_with_progress(
          questionnaire=questionnaire,
          section=section,
          section_index=section_index,
          llm_id=llm_id,
          population_index=population_index,
          provider_semaphore=provider_semaphore,
        )
      )
      for section_index, section in enumerate(questionnaire.sections)
    ]
    section_outcomes = await asyncio.gather(*section_tasks)

    sections: list[dict[str, Any]] = []
    errors: list[RunnerError] = []
    query_times = []
    for section_payload, section_errors, first_query_time in section_outcomes:
      sections.append(section_payload)
      errors.extend(section_errors)
      if first_query_time is not None:
        query_times.append(first_query_time)

    query_time = min(query_times) if query_times else now_utc()
    record = RawResponseRecord(
      questionnaire_name=questionnaire.id,
      questionnaire_path=questionnaire_path,
      llm_id=llm_id,
      population_index=population_index,
      query_time=query_time,
      response={"sections": sections},
      errors=errors,
    )
    if output_path is not None:
      await self._append_jsonl(record, output_path, write_lock)
    return record

  async def _execute_section(
    self,
    *,
    questionnaire: Questionnaire,
    section: Section,
    section_index: int,
    llm_id: str,
    population_index: int,
    provider_semaphore: asyncio.Semaphore,
  ) -> tuple[dict[str, Any], list[RunnerError], Any]:
    section_errors: list[RunnerError] = []
    question_entries: list[dict[str, Any]] = []
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
        {
          "id": section.name,
          "questions": [
            {
              "id": question.id,
              "response": None,
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
      try:
        response, query_time = await self._ask_question(
          conversation,
          prompt,
          provider_semaphore,
          llm_id=llm_id,
          population_index=population_index,
          section_index=section_index,
        )
      except ValidationFailedError as exc:
        error = self._question_error(
          llm_id,
          questionnaire.id,
          population_index,
          section.name,
          question.id,
          "validation",
          str(exc),
        )
        response = None
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
        response = None
        question_errors.append(error)
        section_errors.append(error)
      except LLMConnectorError as exc:
        error = self._question_error(
          llm_id,
          questionnaire.id,
          population_index,
          section.name,
          question.id,
          "network",
          str(exc),
        )
        response = None
        question_errors.append(error)
        section_errors.append(error)
      except Exception as exc:  # pragma: no cover - defensive
        error = self._question_error(
          llm_id,
          questionnaire.id,
          population_index,
          section.name,
          question.id,
          "runtime",
          str(exc),
        )
        response = None
        question_errors.append(error)
        section_errors.append(error)
      else:
        if first_query_time is None:
          first_query_time = query_time
        prior_answers.append(
          QuestionAnswer(
            question_id=question.id,
            prompt=prompt,
            response_text=response.text,
          )
        )

      question_entries.append(
        {
          "id": question.id,
          "response": (
            self._response_to_json_dict(response)
            if response is not None
            else None
          ),
          "errors": [error.to_json_dict() for error in question_errors],
        }
      )

    return (
      {"id": section.name, "questions": question_entries},
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
  ) -> tuple[dict[str, Any], list[RunnerError], Any]:
    try:
      return await self._execute_section(
        questionnaire=questionnaire,
        section=section,
        section_index=section_index,
        llm_id=llm_id,
        population_index=population_index,
        provider_semaphore=provider_semaphore,
      )
    finally:
      if self._progress_display is not None:
        self._progress_display.mark_section_completed(
          llm_id,
          population_index,
          section_index,
        )

  def _question_error(
    self,
    llm_id: str,
    questionnaire_id: str,
    population_index: int,
    section_id: str,
    question_id: str,
    stage: str,
    message: str,
  ) -> RunnerError:
    return RunnerError(
      llm_id=llm_id,
      questionnaire_id=questionnaire_id,
      population_index=population_index,
      section_id=section_id,
      question_id=question_id,
      stage=stage,
      message=message,
    )

  async def _ask_question(
    self,
    conversation: LLMConversation,
    prompt: str,
    provider_semaphore: asyncio.Semaphore,
    *,
    llm_id: str,
    population_index: int,
    section_index: int,
  ) -> tuple[LLMResponse, Any]:
    async with provider_semaphore:
      if self._progress_display is not None:
        self._progress_display.mark_section_started(
          llm_id,
          population_index,
          section_index,
        )
      query_time = now_utc()
      response = await asyncio.to_thread(conversation.ask, prompt)
    return response, query_time

  def _response_to_json_dict(self, response: LLMResponse) -> dict[str, Any]:
    return {
      "raw": response.text,
      "parsed": response.parsed,
      "metadata": {
        "provider": response.provider,
        "model": response.model,
        **response.metadata,
      },
    }

  async def _append_jsonl(
    self,
    record: RawResponseRecord,
    output_path: Path,
    write_lock: asyncio.Lock,
  ) -> None:
    line = json.dumps(record.to_json_dict(), ensure_ascii=False)
    async with write_lock:
      await asyncio.to_thread(
        self._append_line,
        output_path,
        line,
      )

  def _append_line(self, output_path: Path, line: str) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
      handle.write(line + "\n")


__all__ = ["BenchmarkRunner", "RunnerConfigError"]
