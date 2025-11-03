from __future__ import annotations

import pytest

from rationale_benchmark.questionnaire import (
  AnswerValidationError,
  Question,
  QuestionScore,
  QuestionType,
  ScoringRule,
  score_question,
  validate_answer,
)


def _rating_question() -> Question:
  return Question(
    id="q1",
    type=QuestionType.RATING_5,
    prompt="Prompt",
    options=None,
    scoring=ScoringRule(
      total=5,
      weights={
        "1": 0,
        "2": 1,
        "3": 3,
        "4": 4,
        "5": 5,
      },
    ),
  )


def _choice_question() -> Question:
  return Question(
    id="q2",
    type=QuestionType.CHOICE,
    prompt="Prompt",
    options={
      "low": "Low",
      "high": "High",
    },
    scoring=ScoringRule(
      total=3,
      weights={
        "low": 0,
        "high": 3,
      },
    ),
  )


def test_validate_answer_rating_returns_token() -> None:
  question = _rating_question()
  token = validate_answer(question, 3)
  assert token == "3"


def test_validate_answer_choice_returns_key() -> None:
  question = _choice_question()
  token = validate_answer(question, "high")
  assert token == "high"


def test_validate_answer_rating_out_of_range_raises() -> None:
  question = _rating_question()
  with pytest.raises(AnswerValidationError):
    validate_answer(question, 10)


def test_validate_answer_choice_unknown_key_raises() -> None:
  question = _choice_question()
  with pytest.raises(AnswerValidationError):
    validate_answer(question, "medium")


def test_score_question_returns_expected_points() -> None:
  question = _choice_question()
  score = score_question(question, "high")
  assert isinstance(score, QuestionScore)
  assert score.question_id == "q2"
  assert score.awarded == 3
  assert score.total == 3
