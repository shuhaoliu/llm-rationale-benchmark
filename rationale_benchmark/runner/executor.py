"""Asynchronous execution orchestrator for benchmark runs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rationale_benchmark.llm.conversation import ConversationTurn, LLMConversation
from rationale_benchmark.llm.exceptions import (
  ConfigurationError,
  ConversationArchivedError,
  LLMConnectorError,
  ValidationFailedError,
)
from rationale_benchmark.questionnaire.models import Questionnaire, Section
from rationale_benchmark.runner.prompts import build_prompt
from rationale_benchmark.runner.types import (
  BenchmarkResult,
  ModelExecutionResult,
  PromptContext,
  QuestionAnswer,
  QuestionRunTrace,
  RunnerError,
  now_utc,
)

from .evaluator import BenchmarkEvaluator


class ConversationFactoryProtocol(Protocol):
  """Protocol describing conversation factories consumed by the runner."""

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation: ...


@dataclass(frozen=True)
class ModelExecutionPlan:
  """Immutable execution plan for a single model."""

  model: str
  questionnaires: tuple[Questionnaire, ...]


@dataclass
class ModelRunOutcome:
  """Aggregated outcome for a model after execution."""

  model: str
  results: list[ModelExecutionResult]
  errors: list[RunnerError]


class RunnerConfigError(RuntimeError):
  """Raised when runner configuration is invalid."""


class BenchmarkRunner:
  """Coordinate questionnaire execution across multiple models."""

  def __init__(
    self,
    conversation_factory: ConversationFactoryProtocol,
    *,
    max_concurrency: int = 4,
  ) -> None:
    if max_concurrency < 1:
      raise RunnerConfigError("max_concurrency must be >= 1")
    self._conversation_factory = conversation_factory
    self._max_concurrency = max_concurrency

  async def run(
    self,
    questionnaires: Sequence[Questionnaire],
    models: Sequence[str],
    *,
    llm_config: str | None = None,
    total_population: int | None = None,
    parallel_sessions: int = 1,
  ) -> BenchmarkResult:
    if not questionnaires:
      raise RunnerConfigError("At least one questionnaire must be provided")
    if not models:
      raise RunnerConfigError("At least one model must be provided")
    if total_population is not None and total_population < 1:
      raise RunnerConfigError("total_population must be >= 1")
    if parallel_sessions < 1:
      raise RunnerConfigError("parallel_sessions must be >= 1")

    population_by_questionnaire = {
      questionnaire.id: (
        total_population
        if total_population is not None
        else questionnaire.default_population
      )
      for questionnaire in questionnaires
    }
    resolved_total_population = max(population_by_questionnaire.values())

    started_at = now_utc()
    execution_results: list[ModelExecutionResult] = []
    collected_errors: list[RunnerError] = []

    population_semaphore = asyncio.Semaphore(parallel_sessions)
    provider_semaphore = asyncio.Semaphore(self._max_concurrency)
    tasks = [
      asyncio.create_task(
        self._execute_questionnaire_bounded(
          model,
          questionnaire,
          population_index,
          population_semaphore,
          provider_semaphore,
        )
      )
      for model in models
      for questionnaire in questionnaires
      for population_index in range(population_by_questionnaire[questionnaire.id])
    ]
    raw_results = await asyncio.gather(*tasks)
    for result, errors in raw_results:
      execution_results.append(result)
      collected_errors.extend(errors)

    completed_at = now_utc()

    evaluator = BenchmarkEvaluator(
      questionnaires=questionnaires,
      models=models,
      llm_config=llm_config,
      started_at=started_at,
      completed_at=completed_at,
      total_population=resolved_total_population,
      total_population_by_questionnaire=population_by_questionnaire,
      parallel_sessions=parallel_sessions,
    )
    return evaluator.evaluate(execution_results, collected_errors)

  def run_sync(
    self,
    questionnaires: Sequence[Questionnaire],
    models: Sequence[str],
    *,
    llm_config: str | None = None,
    total_population: int | None = None,
    parallel_sessions: int = 1,
  ) -> BenchmarkResult:
    """Synchronous wrapper executing the benchmark in a new event loop."""

    return asyncio.run(
      self.run(
        questionnaires,
        models,
        llm_config=llm_config,
        total_population=total_population,
        parallel_sessions=parallel_sessions,
      )
    )

  async def _execute_model(
    self,
    plan: ModelExecutionPlan,
    semaphore: asyncio.Semaphore,
  ) -> ModelRunOutcome:
    async with semaphore:
      results: list[ModelExecutionResult] = []
      errors: list[RunnerError] = []

      for questionnaire in plan.questionnaires:
        result, questionnaire_errors = await self._execute_questionnaire(
          plan.model,
          questionnaire,
        )
        results.append(result)
        errors.extend(questionnaire_errors)

      return ModelRunOutcome(
        model=plan.model,
        results=results,
        errors=errors,
      )

  async def _execute_questionnaire_bounded(
    self,
    model: str,
    questionnaire: Questionnaire,
    population_index: int,
    semaphore: asyncio.Semaphore,
    provider_semaphore: asyncio.Semaphore,
  ) -> tuple[ModelExecutionResult, list[RunnerError]]:
    async with semaphore:
      return await self._execute_questionnaire(
        model,
        questionnaire,
        population_index,
        provider_semaphore,
      )

  async def _execute_questionnaire(
    self,
    model: str,
    questionnaire: Questionnaire,
    population_index: int = 0,
    provider_semaphore: asyncio.Semaphore | None = None,
  ) -> tuple[ModelExecutionResult, list[RunnerError]]:
    questionnaire_errors: list[RunnerError] = []
    started_at = now_utc()
    section_results = await asyncio.gather(
      *[
        asyncio.create_task(
          self._execute_section(
            model,
            questionnaire,
            section,
            provider_semaphore,
          )
        )
        for section in questionnaire.sections
      ]
    )
    question_traces: list[QuestionRunTrace] = []
    section_transcripts: dict[str, list[ConversationTurn]] = {}
    abort_questionnaire = False
    for section, traces, transcript, errors, aborted in section_results:
      section_transcripts[section.name] = transcript
      question_traces.extend(traces)
      questionnaire_errors.extend(errors)
      abort_questionnaire = abort_questionnaire or aborted

    completed_at = None if abort_questionnaire else now_utc()
    result = ModelExecutionResult(
      model=model,
      questionnaire_id=questionnaire.id,
      population_index=population_index,
      section_transcripts=section_transcripts,
      question_traces=question_traces,
      started_at=started_at,
      completed_at=completed_at,
      errors=list(questionnaire_errors),
    )
    return result, questionnaire_errors

  async def _execute_section(
    self,
    model: str,
    questionnaire: Questionnaire,
    section: Section,
    provider_semaphore: asyncio.Semaphore | None,
  ) -> tuple[
    Section,
    list[QuestionRunTrace],
    list[ConversationTurn],
    list[RunnerError],
    bool,
  ]:
    section_errors: list[RunnerError] = []
    question_traces: list[QuestionRunTrace] = []
    prior_answers: list[QuestionAnswer] = []
    abort_section = False

    try:
      conversation = self._conversation_factory.create(
        model,
        system_prompt=questionnaire.system_prompt,
      )
    except ConfigurationError as exc:
      error = RunnerError(
        model=model,
        stage="configuration",
        message=str(exc),
        details={
          "questionnaire_id": questionnaire.id,
          "section_name": section.name,
        },
      )
      return section, [], [], [error], True

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
      trace = QuestionRunTrace(
        section_name=section.name,
        question_id=question.id,
        prompt=prompt,
        response=None,
        attempts=0,
        latency_ms=None,
      )

      try:
        response, latency_ms = await self._ask_question(
          conversation,
          prompt,
          provider_semaphore,
        )
      except ValidationFailedError as exc:
        error = RunnerError(
          model=model,
          stage="validation",
          message=str(exc),
          details={"question_id": question.id, "section_name": section.name},
        )
        trace.attempts += 1
        trace.errors.append(error)
        section_errors.append(error)
      except ConversationArchivedError as exc:
        error = RunnerError(
          model=model,
          stage="conversation",
          message=str(exc),
          details={"question_id": question.id, "section_name": section.name},
        )
        trace.errors.append(error)
        section_errors.append(error)
        abort_section = True
      except LLMConnectorError as exc:
        error = RunnerError(
          model=model,
          stage="network",
          message=str(exc),
          details={"question_id": question.id, "section_name": section.name},
        )
        trace.errors.append(error)
        section_errors.append(error)
        abort_section = True
      except Exception as exc:  # pragma: no cover - defensive
        error = RunnerError(
          model=model,
          stage="unknown",
          message=str(exc),
          details={"question_id": question.id, "section_name": section.name},
        )
        trace.errors.append(error)
        section_errors.append(error)
        abort_section = True
      else:
        trace.response = response
        trace.attempts += 1
        trace.latency_ms = latency_ms
        prior_answers.append(
          QuestionAnswer(
            question_id=question.id,
            prompt=prompt,
            response_text=response.text,
          )
        )

      question_traces.append(trace)
      if abort_section:
        break

    try:
      archive = conversation.archive()
      transcript = archive.turns
    except ConversationArchivedError as exc:
      error = RunnerError(
        model=model,
        stage="conversation",
        message=str(exc),
        details={
          "questionnaire_id": questionnaire.id,
          "section_name": section.name,
        },
      )
      section_errors.append(error)
      transcript = []

    return section, question_traces, transcript, section_errors, abort_section

  async def _ask_question(
    self,
    conversation: LLMConversation,
    prompt: str,
    provider_semaphore: asyncio.Semaphore | None = None,
  ):
    """Call the underlying conversation in a worker thread."""

    start_time = time.perf_counter()
    if provider_semaphore is None:
      response = await asyncio.to_thread(conversation.ask, prompt)
    else:
      async with provider_semaphore:
        response = await asyncio.to_thread(conversation.ask, prompt)
    latency_seconds = time.perf_counter() - start_time
    latency_ms = int(latency_seconds * 1000)
    return response, latency_ms


__all__ = ["BenchmarkRunner", "RunnerConfigError"]
