from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import Question, QuestionType


def resolve_output_schema(question: Question) -> dict[str, Any]:
  """Return the effective full JSON schema for a question."""

  if question.output_schema:
    return deepcopy(question.output_schema)

  return _default_output_schema(
    question.id,
    question.type,
    question.options,
  )


def build_full_output_schema(
  question_id: str,
  raw_schema: dict[str, Any],
) -> dict[str, Any]:
  """Wrap a simplified questionnaire schema in a full JSON schema object."""

  normalized = deepcopy(raw_schema)
  normalized["type"] = "object"
  normalized["additionalProperties"] = False
  normalized["title"] = str(normalized.get("title") or question_id)
  return normalized


def _default_output_schema(
  question_id: str,
  question_type: QuestionType,
  options: dict[str, str] | None,
) -> dict[str, Any]:
  if question_type is QuestionType.CHOICE:
    answer_schema: dict[str, Any] = {
      "type": "string",
      "enum": list((options or {}).keys()),
    }
  else:
    scale = question_type.rating_scale
    if scale is None:  # pragma: no cover - guard rail
      raise ValueError(
        f"Question type '{question_type.value}' does not define a rating scale"
      )
    answer_schema = {
      "type": "integer",
      "minimum": 1,
      "maximum": scale,
    }

  return {
    "type": "object",
    "additionalProperties": False,
    "properties": {"answer": answer_schema},
    "required": ["answer"],
    "title": question_id,
  }


__all__ = ["build_full_output_schema", "resolve_output_schema"]
