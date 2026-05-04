"""Shared data structures for the questionnaire runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rationale_benchmark.llm.conversation import LLMResponse
from rationale_benchmark.questionnaire.models import Question, Questionnaire


@dataclass(frozen=True)
class RunnerError:
  """Normalized error surfaced during runner execution."""

  llm_id: str
  stage: str
  message: str
  questionnaire_id: str | None = None
  population_index: int | None = None
  section_id: str | None = None
  question_id: str | None = None
  details: dict[str, Any] = field(default_factory=dict)
  retry_count: int | None = None

  def to_json_dict(self) -> dict[str, Any]:
    """Return a JSON-serialisable mapping."""

    return {
      "llm_id": self.llm_id,
      "stage": self.stage,
      "message": self.message,
      "questionnaire_id": self.questionnaire_id,
      "population_index": self.population_index,
      "section_id": self.section_id,
      "question_id": self.question_id,
      "details": self.details,
      "retry_count": self.retry_count,
    }


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


@dataclass(frozen=True)
class RunnerExecutionPlan:
  """Immutable execution plan for one LLM/population administration."""

  questionnaire: Questionnaire
  questionnaire_path: Path | None
  llm_id: str
  population_index: int
  output_dir: Path | None


@dataclass
class QuestionRunTrace:
  """Execution trace for a single questionnaire question."""

  section_id: str
  question_id: str
  canonical_response: str | None
  attempts: list[QuestionAttemptTrace]
  errors: list[RunnerError] = field(default_factory=list)


@dataclass
class QuestionAttemptTrace:
  """Raw provider attempt and validation result for one question."""

  attempt: int
  raw_response: LLMResponse | None
  canonical_response: str | None
  validation_error: str | None
  provider_metadata: dict[str, Any] = field(default_factory=dict)

  def to_json_dict(self) -> dict[str, Any]:
    """Return a JSON-serialisable mapping."""

    return {
      "attempt": self.attempt,
      "raw_response": (
        self.raw_response.text if self.raw_response is not None else None
      ),
      "canonical_response": self.canonical_response,
      "validation_error": self.validation_error,
      "provider_metadata": dict(self.provider_metadata),
    }


@dataclass
class ResponseRecord:
  """Canonical response record for one questionnaire administration."""

  questionnaire_name: str
  questionnaire_path: Path | None
  llm_id: str
  population_index: int
  query_time: datetime
  response: dict[str, Any]
  errors: list[RunnerError] = field(default_factory=list)

  def to_json_dict(self) -> dict[str, Any]:
    """Return the JSONL payload for this record."""

    return {
      "questionnaire": {
        "name": self.questionnaire_name,
        "path": str(self.questionnaire_path) if self.questionnaire_path else None,
      },
      "llm_id": self.llm_id,
      "population_index": self.population_index,
      "query_time": self.query_time.isoformat(),
      "response": self.response,
      "errors": [error.to_json_dict() for error in self.errors],
    }


@dataclass
class MetadataRecord:
  """Provider metadata record aligned with a response record."""

  questionnaire_name: str
  questionnaire_path: Path | None
  llm_id: str
  population_index: int
  query_time: datetime
  metadata: dict[str, Any]
  errors: list[RunnerError] = field(default_factory=list)

  def to_json_dict(self) -> dict[str, Any]:
    """Return the JSONL payload for this record."""

    return {
      "questionnaire": {
        "name": self.questionnaire_name,
        "path": str(self.questionnaire_path) if self.questionnaire_path else None,
      },
      "llm_id": self.llm_id,
      "population_index": self.population_index,
      "query_time": self.query_time.isoformat(),
      "metadata": self.metadata,
      "errors": [error.to_json_dict() for error in self.errors],
    }


@dataclass
class OutputRecordPair:
  """Line-aligned response and metadata records."""

  response_record: ResponseRecord
  metadata_record: MetadataRecord


@dataclass
class RawRunResult:
  """Result returned by a runner execution."""

  records: list[ResponseRecord]
  metadata_records: list[MetadataRecord] = field(default_factory=list)
  output_dir: Path | None = None
  errors: list[RunnerError] = field(default_factory=list)


def now_utc() -> datetime:
  """Return the current UTC timestamp."""

  return datetime.now(timezone.utc)


__all__ = [
  "MetadataRecord",
  "OutputRecordPair",
  "PromptContext",
  "QuestionAnswer",
  "QuestionAttemptTrace",
  "QuestionRunTrace",
  "RawRunResult",
  "ResponseRecord",
  "RunnerError",
  "RunnerExecutionPlan",
  "now_utc",
]
