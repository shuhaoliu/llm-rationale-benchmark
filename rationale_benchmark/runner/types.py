"""Shared data structures for the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional

from rationale_benchmark.llm.conversation import ConversationTurn, LLMResponse
from rationale_benchmark.questionnaire.models import Question, QuestionScore


@dataclass(frozen=True)
class RunnerError:
  """Normalized error surfaced during execution or evaluation."""

  model: str
  stage: str
  message: str
  details: dict[str, Any] = field(default_factory=dict)
  retry_count: Optional[int] = None


@dataclass(frozen=True)
class PromptContext:
  """Context used when rendering prompts for questionnaire questions."""

  questionnaire_id: str
  section_name: str
  section_instructions: Optional[str]
  question: Question
  system_prompt: Optional[str]


@dataclass
class QuestionRunTrace:
  """Execution trace for a single questionnaire question."""

  question_id: str
  prompt: str
  response: Optional[LLMResponse]
  attempts: int
  latency_ms: Optional[int]
  errors: list[RunnerError] = field(default_factory=list)


@dataclass
class ModelExecutionResult:
  """Execution artefacts for a model/questionnaire pairing."""

  model: str
  questionnaire_id: str
  transcript: list[ConversationTurn]
  question_traces: list[QuestionRunTrace]
  started_at: datetime
  completed_at: Optional[datetime]
  errors: list[RunnerError] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkInfo:
  """Metadata describing the executed benchmark run."""

  questionnaires: tuple[str, ...]
  models_tested: tuple[str, ...]
  llm_config: Optional[str]
  started_at: datetime
  completed_at: datetime


@dataclass
class QuestionResult:
  """Evaluation result for an answered question."""

  questionnaire_id: str
  model: str
  question_id: str
  response_text: str
  reasoning: Optional[str]
  score: QuestionScore
  latency_ms: int
  metadata: dict[str, Any] = field(default_factory=dict)
  errors: list[RunnerError] = field(default_factory=list)


@dataclass
class SectionScore:
  """Aggregated scores for a questionnaire section."""

  section_name: str
  questions: list[QuestionScore]

  @property
  def awarded(self) -> int:
    return sum(score.awarded for score in self.questions)

  @property
  def total(self) -> int:
    return sum(score.total for score in self.questions)


@dataclass
class QuestionnaireScore:
  """Aggregated scores for a questionnaire."""

  questionnaire_id: str
  sections: list[SectionScore]

  @property
  def awarded(self) -> int:
    return sum(section.awarded for section in self.sections)

  @property
  def total(self) -> int:
    return sum(section.total for section in self.sections)


@dataclass
class ModelBenchmarkResult:
  """Evaluation outputs for a specific model."""

  model: str
  questionnaire_scores: list[QuestionnaireScore]
  questions: list[QuestionResult]
  transcript: list[ConversationTurn]
  errors: list[RunnerError] = field(default_factory=list)
  cost_estimate: float = 0.0


@dataclass
class BenchmarkSummary:
  """Cross-model summary statistics."""

  questionnaires_run: int
  total_questions: int
  models_tested: int
  average_scores_by_questionnaire: dict[str, float]
  average_scores_by_model: dict[str, float]
  cost_estimates: dict[str, float]


@dataclass
class BenchmarkResult:
  """Top-level result returned by the runner."""

  info: BenchmarkInfo
  model_results: list[ModelBenchmarkResult]
  summary: BenchmarkSummary
  errors: list[RunnerError] = field(default_factory=list)


def now_utc() -> datetime:
  """Return the current UTC timestamp."""

  return datetime.now(UTC)


__all__ = [
  "BenchmarkInfo",
  "BenchmarkResult",
  "BenchmarkSummary",
  "ModelBenchmarkResult",
  "ModelExecutionResult",
  "PromptContext",
  "QuestionResult",
  "QuestionRunTrace",
  "QuestionnaireScore",
  "RunnerError",
  "SectionScore",
  "now_utc",
]
