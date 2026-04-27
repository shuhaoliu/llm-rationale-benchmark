"""Evaluation logic translating execution artefacts into benchmark results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rationale_benchmark.llm.conversation import ConversationTurn, LLMResponse
from rationale_benchmark.questionnaire.errors import (
  AnswerValidationError,
  QuestionnaireConfigError,
)
from rationale_benchmark.questionnaire.models import (
  Question,
  Questionnaire,
  QuestionScore,
)
from rationale_benchmark.questionnaire.scoring import score_question, validate_answer
from rationale_benchmark.runner.types import (
  BenchmarkInfo,
  BenchmarkResult,
  BenchmarkSummary,
  ModelBenchmarkResult,
  ModelExecutionResult,
  PopulationResult,
  QuestionnaireScore,
  QuestionResult,
  QuestionRunTrace,
  RunnerError,
  SectionScore,
)


class BenchmarkEvaluator:
  """Evaluate model execution results into aggregate benchmark outputs."""

  def __init__(
    self,
    *,
    questionnaires: Sequence[Questionnaire],
    models: Sequence[str],
    llm_config: str | None,
    started_at,
    completed_at,
    total_population: int = 1,
    total_population_by_questionnaire: dict[str, int] | None = None,
    parallel_sessions: int = 1,
  ) -> None:
    self._questionnaires = {q.id: q for q in questionnaires}
    self._questionnaire_ids = tuple(q.id for q in questionnaires)
    self._models = tuple(models)
    self._llm_config = llm_config
    self._started_at = started_at
    self._completed_at = completed_at
    self._total_population = total_population
    self._total_population_by_questionnaire = (
      dict(total_population_by_questionnaire)
      if total_population_by_questionnaire is not None
      else {q.id: total_population for q in questionnaires}
    )
    self._parallel_sessions = parallel_sessions

  def evaluate(
    self,
    execution_results: Sequence[ModelExecutionResult],
    execution_errors: Sequence[RunnerError],
  ) -> BenchmarkResult:
    grouped = defaultdict(list)
    for result in execution_results:
      grouped[result.model].append(result)

    question_counts = {
      qid: sum(len(section.questions) for section in questionnaire.sections)
      for qid, questionnaire in self._questionnaires.items()
    }

    questionnaire_totals: dict[str, list[tuple[int, int]]] = {
      qid: [] for qid in self._questionnaires
    }
    model_totals: dict[str, list[tuple[int, int]]] = {model: [] for model in self._models}
    cost_estimates: dict[str, float] = dict.fromkeys(self._models, 0.0)

    model_results: list[ModelBenchmarkResult] = []
    scoring_errors: list[RunnerError] = []

    for model in self._models:
      model_execution_results = grouped.get(model, [])
      model_questionnaire_scores: list[QuestionnaireScore] = []
      model_questions: list[QuestionResult] = []
      model_section_transcripts: dict[str, list[ConversationTurn]] = {}
      model_errors: list[RunnerError] = []
      model_population_index = 0

      for execution_result in model_execution_results:
        questionnaire = self._questionnaires.get(execution_result.questionnaire_id)
        if questionnaire is None:
          continue
        model_population_index = execution_result.population_index

        trace_lookup = {
          trace.question_id: trace for trace in execution_result.question_traces
        }

        section_scores: list[SectionScore] = []

        for section in questionnaire.sections:
          question_scores = []
          for question in section.questions:
            trace = trace_lookup.get(question.id)
            result, score, errors = self._score_question(
              questionnaire.id,
              section.name,
              model,
              execution_result.population_index,
              question,
              trace,
            )
            model_questions.append(result)
            question_scores.append(score)
            model_errors.extend(errors)
            scoring_errors.extend(
              error for error in errors if error.stage == "scoring"
            )
            if result.metadata.get("cost") is not None:
              try:
                cost_estimates[model] += float(result.metadata["cost"])
              except (TypeError, ValueError):
                pass
            if result.metadata.get("cost_estimate") is not None:
              try:
                cost_estimates[model] += float(result.metadata["cost_estimate"])
              except (TypeError, ValueError):
                pass
          section_scores.append(
            SectionScore(section_name=section.name, questions=question_scores)
          )

        questionnaire_score = QuestionnaireScore(
          questionnaire_id=questionnaire.id,
          sections=section_scores,
        )
        model_questionnaire_scores.append(questionnaire_score)
        questionnaire_totals[questionnaire.id].append(
          (questionnaire_score.awarded, questionnaire_score.total)
        )
        model_totals.setdefault(model, []).append(
          (questionnaire_score.awarded, questionnaire_score.total)
        )
        for section_name, transcript in execution_result.section_transcripts.items():
          key = section_name
          if key in model_section_transcripts:
            key = (
              f"{execution_result.questionnaire_id}:"
              f"{execution_result.population_index}:"
              f"{section_name}"
            )
          model_section_transcripts[key] = transcript
        model_errors.extend(execution_result.errors)

      model_results.append(
        ModelBenchmarkResult(
          model=model,
          population_index=model_population_index,
          questionnaire_scores=model_questionnaire_scores,
          questions=model_questions,
          section_transcripts=model_section_transcripts,
          errors=model_errors,
          cost_estimate=cost_estimates.get(model, 0.0),
        )
      )

    questionnaires_run = len(self._questionnaires)
    total_questions = sum(question_counts.values())
    models_tested = len(self._models)

    average_scores_by_questionnaire = {
      questionnaire_id: (
        sum(awarded for awarded, _ in entries) / max(1, sum(total for _, total in entries))
        if entries
        else 0.0
      )
      for questionnaire_id, entries in questionnaire_totals.items()
    }

    average_scores_by_model = {
      model: (
        sum(awarded for awarded, _ in entries) / max(1, sum(total for _, total in entries))
        if entries
        else 0.0
      )
      for model, entries in model_totals.items()
    }

    summary = BenchmarkSummary(
      questionnaires_run=questionnaires_run,
      total_population=sum(self._total_population_by_questionnaire.values()),
      total_questions=total_questions,
      models_tested=models_tested,
      average_scores_by_questionnaire=average_scores_by_questionnaire,
      average_scores_by_model=average_scores_by_model,
      cost_estimates=cost_estimates,
    )

    info = BenchmarkInfo(
      questionnaires=self._questionnaire_ids,
      models_tested=self._models,
      llm_config=self._llm_config,
      started_at=self._started_at,
      completed_at=self._completed_at,
      total_population=self._total_population,
      total_population_by_questionnaire=self._total_population_by_questionnaire,
      parallel_sessions=self._parallel_sessions,
    )

    errors = list(execution_errors) + scoring_errors

    population_results: list[PopulationResult] = []
    if self._total_population > 1:
      for model_result in model_results:
        sessions_by_questionnaire: dict[str, list[QuestionnaireScore]] = defaultdict(list)
        for qs in model_result.questionnaire_scores:
          sessions_by_questionnaire[qs.questionnaire_id].append(qs)
        for qid, sessions in sessions_by_questionnaire.items():
          population_results.append(
            PopulationResult(
              questionnaire_id=qid,
              model=model_result.model,
              total_population=self._total_population_by_questionnaire.get(
                qid,
                self._total_population,
              ),
              parallel_sessions=self._parallel_sessions,
              sessions=sessions,
            )
          )

    return BenchmarkResult(
      info=info,
      model_results=model_results,
      summary=summary,
      errors=errors,
      population_results=population_results,
    )

  def _score_question(
    self,
    questionnaire_id: str,
    section_name: str,
    model: str,
    population_index: int,
    question: Question,
    trace: QuestionRunTrace | None,
  ) -> tuple[QuestionResult, QuestionScore, list[RunnerError]]:
    response = trace.response if trace and trace.response else None
    response_text = response.text if response else ""
    reasoning = self._extract_reasoning(response)
    latency_ms = trace.latency_ms if trace and trace.latency_ms is not None else 0
    metadata = dict(response.metadata) if response else {}

    errors: list[RunnerError] = []
    score_errors: list[RunnerError] = []
    if trace and trace.errors:
      errors.extend(trace.errors)

    raw_answer = self._extract_answer(response)

    if raw_answer is None:
      question_score = QuestionScore(
        question_id=question.id,
        awarded=0,
        total=question.scoring.total,
      )
      score_errors.append(
        RunnerError(
          model=model,
          stage="scoring",
          message="Missing answer",
          details={"question_id": question.id},
        )
      )
    else:
      try:
        answer_token = validate_answer(question, raw_answer)
        question_score = score_question(question, answer_token)
      except AnswerValidationError as exc:
        question_score = QuestionScore(
          question_id=question.id,
          awarded=0,
          total=question.scoring.total,
        )
        score_errors.append(
          RunnerError(
            model=model,
            stage="scoring",
            message=str(exc),
            details={"question_id": question.id},
          )
        )
      except QuestionnaireConfigError as exc:
        question_score = QuestionScore(
          question_id=question.id,
          awarded=0,
          total=question.scoring.total,
        )
        score_errors.append(
          RunnerError(
            model=model,
            stage="scoring",
            message=str(exc),
            details={"question_id": question.id},
          )
        )

    errors.extend(score_errors)

    question_result = QuestionResult(
      questionnaire_id=questionnaire_id,
      section_name=section_name,
      model=model,
      population_index=population_index,
      question_id=question.id,
      response_text=response_text,
      reasoning=reasoning,
      score=question_score,
      latency_ms=latency_ms,
      metadata=metadata,
      errors=errors,
    )

    return question_result, question_score, errors

  def _extract_answer(
    self,
    response: LLMResponse | None,
  ):
    if response is None:
      return None

    parsed = response.parsed
    if isinstance(parsed, dict):
      for key in ("answer", "value", "selection"):
        if key in parsed:
          return parsed[key]
    return response.text

  def _extract_reasoning(self, response: LLMResponse | None):
    if response is None:
      return None

    parsed = response.parsed
    if isinstance(parsed, dict):
      for key in ("reasoning", "rationale", "explanation"):
        if key in parsed:
          return parsed[key]
    return None


__all__ = ["BenchmarkEvaluator"]
