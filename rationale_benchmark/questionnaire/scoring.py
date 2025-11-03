from __future__ import annotations

from typing import Any

from .errors import AnswerValidationError, QuestionnaireConfigError
from .models import Question, QuestionScore, QuestionType


def validate_answer(question: Question, answer: Any) -> str:
  """Validate an answer against the question type and return a canonical token."""
  if question.type is QuestionType.CHOICE:
    if not isinstance(answer, str):
      raise AnswerValidationError(
        "Expected string answer matching declared options", question.id
      )
    if question.options is None or answer not in question.options:
      allowed = sorted(question.options or {})
      raise AnswerValidationError(
        f"Answer '{answer}' not allowed. Expected one of {allowed}",
        question.id,
      )
    return answer

  if not isinstance(answer, int):
    raise AnswerValidationError(
      "Expected integer rating within allowed range", question.id
    )
  scale = question.type.rating_scale
  if scale is None:
    raise QuestionnaireConfigError(
      f"Unsupported rating scale for question {question.id}"
    )
  if answer < 1 or answer > scale:
    raise AnswerValidationError(
      f"Rating {answer} outside allowed range 1-{scale}", question.id
    )
  return str(answer)


def score_question(question: Question, answer_token: str) -> QuestionScore:
  """Compute awarded points for a question given a validated answer token."""
  try:
    awarded = question.scoring.weights[answer_token]
  except KeyError as exc:
    raise QuestionnaireConfigError(
      f"Missing scoring weight for answer '{answer_token}' "
      f"on question {question.id}"
    ) from exc
  return QuestionScore(
    question_id=question.id,
    awarded=awarded,
    total=question.scoring.total,
  )
