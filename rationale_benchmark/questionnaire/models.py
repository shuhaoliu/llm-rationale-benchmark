from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuestionType(str, Enum):
  """Supported questionnaire question types."""

  RATING_5 = "rating-5"
  RATING_7 = "rating-7"
  RATING_11 = "rating-11"
  CHOICE = "choice"

  @property
  def rating_scale(self) -> int | None:
    if self is QuestionType.CHOICE:
      return None
    return int(self.value.split("-")[1])


@dataclass(frozen=True)
class ScoringRule:
  total: int
  weights: dict[str, int]


@dataclass(frozen=True)
class Question:
  id: str
  type: QuestionType
  prompt: str
  options: dict[str, str] | None
  scoring: ScoringRule


@dataclass(frozen=True)
class Section:
  name: str
  instructions: str | None
  questions: list[Question]


@dataclass(frozen=True)
class Questionnaire:
  id: str
  name: str
  description: str | None
  version: int | None
  metadata: dict[str, str]
  sections: list[Section]


@dataclass(frozen=True)
class QuestionScore:
  question_id: str
  awarded: int
  total: int
