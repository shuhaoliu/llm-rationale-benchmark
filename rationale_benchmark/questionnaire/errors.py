from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionnaireConfigError(Exception):
  """Raised when questionnaire configuration fails validation."""

  message: str
  file_path: str | None = None
  location: str | None = None
  line_number: int | None = None

  def __str__(self) -> str:
    parts = [self.message]
    if self.location:
      parts.append(f"at {self.location}")
    if self.file_path:
      parts.append(f"in {self.file_path}")
    if self.line_number is not None:
      parts.append(f"line {self.line_number}")
    return " ".join(parts)


@dataclass(frozen=True)
class AnswerValidationError(Exception):
  """Raised when an answer fails validation against a question."""

  message: str
  question_id: str

  def __str__(self) -> str:
    return f"{self.message} for question {self.question_id}"
