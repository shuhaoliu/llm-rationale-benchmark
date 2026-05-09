from __future__ import annotations

from rationale_benchmark.questionnaire.models import (
  Question,
  QuestionType,
  ScoringRule,
)
from rationale_benchmark.runner.prompts import build_prompt
from rationale_benchmark.runner.types import PromptContext


def test_build_prompt_includes_normalized_output_schema() -> None:
  question = Question(
    id="rating_01",
    type=QuestionType.RATING_5,
    prompt="Rate the statement from 1 to 5.",
    output_schema={
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "answer": {
          "type": "integer",
          "minimum": 1,
          "maximum": 5,
        }
      },
      "required": ["answer"],
      "title": "rating_01",
    },
    options=None,
    scoring=ScoringRule(
      total=5,
      weights={str(value): value for value in range(1, 6)},
    ),
  )
  context = PromptContext(
    questionnaire_id="sample",
    section_name="General",
    section_instructions="Answer carefully.",
    question=question,
    system_prompt="System prompt",
    prior_answers=[],
  )

  prompt = build_prompt(context)

  assert "Rate the statement from 1 to 5." in prompt
  assert "Respond with JSON that matches this schema exactly" in prompt
  assert '"title": "rating_01"' in prompt
  assert '"additionalProperties": false' in prompt

