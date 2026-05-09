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
      system_prompt: "You are a careful assistant helping with burnout surveys."
      metadata:
        default_population: 5
        author: "Psych Lab"
      sections:
        - name: "Workload"
          human:
            average: 6.4
            population: 128
          instructions: "Rate how often..."
          questions:
            - id: "workload_01"
              type: "rating-5"
              prompt: "I feel overwhelmed by tasks."
              output_schema:
                properties:
                  answer:
                    type: "integer"
                    minimum: 1
                    maximum: 5
                required: ["answer"]
              scoring:
                total: 5
                weights: [0, 1, 3, 4, 5]
            - id: "workload_02"
              type: "choice"
              prompt: "Which statement best reflects your current workload?"
              output_schema:
                properties:
                  answer:
                    type: "string"
                    enum: ["low", "high"]
                required: ["answer"]
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
  assert questionnaire.default_population == 5
  assert questionnaire.metadata == {
    "default_population": 5,
    "author": "Psych Lab",
  }
  assert (
    questionnaire.system_prompt
    == "You are a careful assistant helping with burnout surveys."
  )
  assert len(questionnaire.sections) == 1
  section = questionnaire.sections[0]
  assert section.name == "Workload"
  assert section.human is not None
  assert section.human.average == 6.4
  assert section.human.population == 128
  assert len(section.questions) == 2
  rating = section.questions[0]
  assert rating.scoring.weights == {
    "1": 0,
    "2": 1,
    "3": 3,
    "4": 4,
    "5": 5,
  }
  assert rating.output_schema == {
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
    "title": "workload_01",
  }
  choice = section.questions[1]
  assert choice.options == {"low": "Manageable", "high": "Overwhelming"}
  assert choice.output_schema == {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "answer": {
        "type": "string",
        "enum": ["low", "high"],
      }
    },
    "required": ["answer"],
    "title": "workload_02",
  }


def test_missing_output_schema_raises(questionnaire_dir: Path) -> None:
  _write_yaml(
    questionnaire_dir,
    "missing-output-schema",
    """
    questionnaire:
      id: "missing-output-schema"
      name: "Missing Output Schema"
      system_prompt: "You answer using short sentences."
      metadata:
        default_population: 1
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

  with pytest.raises(QuestionnaireConfigError) as error:
    load_questionnaire("missing-output-schema", questionnaire_dir.parent)
  assert error.value.location == "questionnaire.sections.0.questions.0.output_schema"


def test_missing_system_prompt_raises(questionnaire_dir: Path) -> None:
  _write_yaml(
    questionnaire_dir,
    "no-system",
    """
    questionnaire:
      id: "no-system"
      name: "No System Prompt"
      metadata:
        default_population: 1
      sections:
        - name: "Only"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "Prompt"
              output_schema:
                properties:
                  answer:
                    type: "integer"
                    minimum: 1
                    maximum: 5
                required: ["answer"]
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
    """,
  )
  with pytest.raises(QuestionnaireConfigError) as error:
    load_questionnaire("no-system", questionnaire_dir.parent)
  assert error.value.location == "questionnaire.system_prompt"


def test_missing_default_population_raises(questionnaire_dir: Path) -> None:
  _write_yaml(
    questionnaire_dir,
    "no-population",
    """
    questionnaire:
      id: "no-population"
      name: "No Population"
      system_prompt: "You answer using short sentences."
      metadata:
        author: "Psych Lab"
      sections:
        - name: "Only"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "Prompt"
              output_schema:
                properties:
                  answer:
                    type: "integer"
                    minimum: 1
                    maximum: 5
                required: ["answer"]
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
    """,
  )
  with pytest.raises(QuestionnaireConfigError) as error:
    load_questionnaire("no-population", questionnaire_dir.parent)
  assert error.value.location == "questionnaire.metadata.default_population"


def test_non_positive_default_population_raises(questionnaire_dir: Path) -> None:
  _write_yaml(
    questionnaire_dir,
    "bad-population",
    """
    questionnaire:
      id: "bad-population"
      name: "Bad Population"
      system_prompt: "You answer using short sentences."
      metadata:
        default_population: 0
      sections:
        - name: "Only"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "Prompt"
              output_schema:
                properties:
                  answer:
                    type: "integer"
                    minimum: 1
                    maximum: 5
                required: ["answer"]
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
    """,
  )
  with pytest.raises(QuestionnaireConfigError) as error:
    load_questionnaire("bad-population", questionnaire_dir.parent)
  assert error.value.location == "questionnaire.metadata.default_population"


def test_non_positive_human_population_raises(questionnaire_dir: Path) -> None:
  _write_yaml(
    questionnaire_dir,
    "bad-human-population",
    """
    questionnaire:
      id: "bad-human-population"
      name: "Bad Human Population"
      system_prompt: "You answer using short sentences."
      metadata:
        default_population: 1
      sections:
        - name: "Only"
          human:
            average: 3.5
            population: 0
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "Prompt"
              output_schema:
                properties:
                  answer:
                    type: "integer"
                    minimum: 1
                    maximum: 5
                required: ["answer"]
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
    """,
  )
  with pytest.raises(QuestionnaireConfigError) as error:
    load_questionnaire("bad-human-population", questionnaire_dir.parent)
  assert error.value.location == "questionnaire.sections[0].human.population"


def test_duplicate_question_id_raises(questionnaire_dir: Path) -> None:
  _write_yaml(
    questionnaire_dir,
    "dup-question",
    """
    questionnaire:
      id: "dup-question"
      name: "Dup"
      system_prompt: "You default to structured survey guidance."
      metadata:
        default_population: 1
      sections:
        - name: "Workload"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "One"
              output_schema:
                properties:
                  answer:
                    type: "integer"
                    minimum: 1
                    maximum: 5
                required: ["answer"]
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
            - id: "q1"
              type: "choice"
              prompt: "Two"
              output_schema:
                properties:
                  answer:
                    type: "string"
                    enum: ["a", "b"]
                required: ["answer"]
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
      system_prompt: "Provide structured answers."
      metadata:
        default_population: 1
      sections:
        - name: "Workload"
          questions:
            - id: "q1"
              type: "choice"
              prompt: "Two"
              output_schema:
                properties:
                  answer:
                    type: "string"
                    enum: ["a", "b"]
                required: ["answer"]
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
      system_prompt: "Stay neutral."
      metadata:
        default_population: 1
      sections:
        - name: "Only"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "Prompt"
              output_schema:
                properties:
                  answer:
                    type: "integer"
                    minimum: 1
                    maximum: 5
                required: ["answer"]
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
      system_prompt: "Stay neutral."
      metadata:
        default_population: 1
      sections:
        - name: "Only"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "Prompt"
              output_schema:
                properties:
                  answer:
                    type: "integer"
                    minimum: 1
                    maximum: 5
                required: ["answer"]
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
    """,
  )
  names = list_questionnaires(questionnaire_dir.parent)
  assert names == ["first", "second"]
