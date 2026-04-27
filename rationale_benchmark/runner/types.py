"""Shared data structures for the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rationale_benchmark.llm.conversation import ConversationTurn, LLMResponse
from rationale_benchmark.questionnaire.models import Question, QuestionScore


@dataclass(frozen=True)
class RunnerError:
  """Normalized error surfaced during execution or evaluation."""

  model: str
  stage: str
  message: str
  details: dict[str, Any] = field(default_factory=dict)
  retry_count: int | None = None


@dataclass(frozen=True)
class PromptContext:
  """Context used when rendering prompts for questionnaire questions."""

  questionnaire_id: str
  section_name: str
  section_instructions: str | None
  question: Question
  system_prompt: str | None
  prior_answers: list[QuestionAnswer] = field(default_factory=list)


@dataclass(frozen=True)
class QuestionAnswer:
  """Question-answer pair available as in-section context."""

  question_id: str
  prompt: str
  response_text: str


@dataclass
class QuestionRunTrace:
  """Execution trace for a single questionnaire question."""

  section_name: str
  question_id: str
  prompt: str
  response: LLMResponse | None
  attempts: int
  latency_ms: int | None
  errors: list[RunnerError] = field(default_factory=list)


@dataclass
class ModelExecutionResult:
  """Execution artefacts for a model/questionnaire pairing."""

  model: str
  questionnaire_id: str
  population_index: int
  section_transcripts: dict[str, list[ConversationTurn]]
  question_traces: list[QuestionRunTrace]
  started_at: datetime
  completed_at: datetime | None
  errors: list[RunnerError] = field(default_factory=list)

  @property
  def transcript(self) -> list[ConversationTurn]:
    """Return flattened section transcripts for backward-compatible consumers."""

    turns: list[ConversationTurn] = []
    for transcript in self.section_transcripts.values():
      turns.extend(transcript)
    return turns


@dataclass(frozen=True)
class BenchmarkInfo:
  """Metadata describing the executed benchmark run."""

  questionnaires: tuple[str, ...]
  models_tested: tuple[str, ...]
  llm_config: str | None
  started_at: datetime
  completed_at: datetime
  total_population: int = 1
  total_population_by_questionnaire: dict[str, int] = field(default_factory=dict)
  parallel_sessions: int = 1


@dataclass
class QuestionResult:
  """Evaluation result for an answered question."""

  questionnaire_id: str
  section_name: str
  model: str
  population_index: int
  question_id: str
  response_text: str
  reasoning: str | None
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
  population_index: int
  questionnaire_scores: list[QuestionnaireScore]
  questions: list[QuestionResult]
  section_transcripts: dict[str, list[ConversationTurn]]
  errors: list[RunnerError] = field(default_factory=list)
  cost_estimate: float = 0.0

  @property
  def transcript(self) -> list[ConversationTurn]:
    """Return flattened section transcripts for backward-compatible consumers."""

    turns: list[ConversationTurn] = []
    for transcript in self.section_transcripts.values():
      turns.extend(transcript)
    return turns


@dataclass
class BenchmarkSummary:
  """Cross-model summary statistics."""

  questionnaires_run: int
  total_population: int
  total_questions: int
  models_tested: int
  average_scores_by_questionnaire: dict[str, float]
  average_scores_by_model: dict[str, float]
  cost_estimates: dict[str, float]


@dataclass
class PopulationResult:
  """Distribution of answers for a questionnaire dispatched across many independent sessions."""

  questionnaire_id: str
  model: str
  total_population: int
  parallel_sessions: int
  sessions: list[QuestionnaireScore]


@dataclass
class BenchmarkResult:
  """Top-level result returned by the runner."""

  info: BenchmarkInfo
  model_results: list[ModelBenchmarkResult]
  summary: BenchmarkSummary
  errors: list[RunnerError] = field(default_factory=list)
  population_results: list[PopulationResult] = field(default_factory=list)


def now_utc() -> datetime:
  """Return the current UTC timestamp."""

  return datetime.now(UTC)


__all__ = [
  "BenchmarkInfo",
  "BenchmarkResult",
  "BenchmarkSummary",
  "ModelBenchmarkResult",
  "ModelExecutionResult",
  "PopulationResult",
  "PromptContext",
  "QuestionAnswer",
  "QuestionResult",
  "QuestionRunTrace",
  "QuestionnaireScore",
  "RunnerError",
  "SectionScore",
  "now_utc",
]
