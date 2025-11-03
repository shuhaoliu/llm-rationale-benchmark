from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from rationale_benchmark.questionnaire import (
  QuestionnaireConfigError,
  list_questionnaires,
  load_questionnaire,
)


@pytest.fixture()
def questionnaire_dir(tmp_path: Path) -> Path:
  root = tmp_path / "questionnaires"
  root.mkdir()
  return root


def _write_yaml(root: Path, name: str, content: str) -> Path:
  path = root / f"{name}.yaml"
  path.write_text(dedent(content), encoding="utf-8")
  return path


def test_load_questionnaire_success(questionnaire_dir: Path) -> None:
  _write_yaml(
    questionnaire_dir,
    "burnout",
    """
    questionnaire:
      id: "burnout"
      name: "Burnout Survey"
      description: "A test questionnaire"
      version: 1
      metadata:
        author: "Psych Lab"
      sections:
        - name: "Workload"
          instructions: "Rate how often..."
          questions:
            - id: "workload_01"
              type: "rating-5"
              prompt: "I feel overwhelmed by tasks."
              scoring:
                total: 5
                weights: [0, 1, 3, 4, 5]
            - id: "workload_02"
              type: "choice"
              prompt: "Which statement best reflects your current workload?"
              options:
                low: "Manageable"
                high: "Overwhelming"
              scoring:
                total: 3
                weights:
                  low: 0
                  high: 3
    """,
  )
  questionnaire = load_questionnaire("burnout", questionnaire_dir.parent)
  assert questionnaire.id == "burnout"
  assert questionnaire.metadata == {"author": "Psych Lab"}
  assert len(questionnaire.sections) == 1
  section = questionnaire.sections[0]
  assert section.name == "Workload"
  assert len(section.questions) == 2
  rating = section.questions[0]
  assert rating.scoring.weights == {
    "1": 0,
    "2": 1,
    "3": 3,
    "4": 4,
    "5": 5,
  }
  choice = section.questions[1]
  assert choice.options == {"low": "Manageable", "high": "Overwhelming"}


def test_duplicate_question_id_raises(questionnaire_dir: Path) -> None:
  _write_yaml(
    questionnaire_dir,
    "dup-question",
    """
    questionnaire:
      id: "dup-question"
      name: "Dup"
      sections:
        - name: "Workload"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "One"
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
            - id: "q1"
              type: "choice"
              prompt: "Two"
              options:
                a: "A"
                b: "B"
              scoring:
                total: 2
                weights:
                  a: 0
                  b: 2
    """,
  )
  with pytest.raises(QuestionnaireConfigError) as error:
    load_questionnaire("dup-question", questionnaire_dir.parent)
  assert "Duplicate question id" in error.value.message


def test_choice_weights_mismatch_raises(questionnaire_dir: Path) -> None:
  _write_yaml(
    questionnaire_dir,
    "bad-choice",
    """
    questionnaire:
      id: "bad-choice"
      name: "Bad"
      sections:
        - name: "Workload"
          questions:
            - id: "q1"
              type: "choice"
              prompt: "Two"
              options:
                a: "A"
                b: "B"
              scoring:
                total: 2
                weights:
                  a: 0
    """,
  )
  with pytest.raises(QuestionnaireConfigError) as error:
    load_questionnaire("bad-choice", questionnaire_dir.parent)
  assert "match declared option keys" in error.value.message


def test_list_questionnaires_returns_file_stems(questionnaire_dir: Path) -> None:
  _write_yaml(
    questionnaire_dir,
    "first",
    """
    questionnaire:
      id: "first"
      name: "First"
      sections:
        - name: "Only"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "Prompt"
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
    """,
  )
  _write_yaml(
    questionnaire_dir,
    "second",
    """
    questionnaire:
      id: "second"
      name: "Second"
      sections:
        - name: "Only"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "Prompt"
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
    """,
  )
  names = list_questionnaires(questionnaire_dir.parent)
  assert names == ["first", "second"]
