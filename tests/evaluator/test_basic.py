from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rationale_benchmark.evaluator.basic import EvaluatorError, evaluate_basic

FIXTURE_CONFIG_DIR = Path(__file__).parent / "fixtures" / "config"


def write_runner_output(output_dir: Path, records: list[dict[str, Any]]) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "responses.jsonl").write_text(
    "\n".join(json.dumps(record) for record in records) + "\n",
    encoding="utf-8",
  )


def record(
  *,
  llm_id: str = "openai/gpt-4",
  population_index: int = 0,
  questionnaire_name: str = "evaluator-fixture",
  sections: list[dict[str, Any]] | None = None,
  errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
  return {
    "questionnaire": {
      "name": questionnaire_name,
      "path": "ignored/provenance.yaml",
    },
    "llm_id": llm_id,
    "population_index": population_index,
    "query_time": "2026-05-04T00:00:00+00:00",
    "response": {
      "sections": sections
      if sections is not None
      else [
        {
          "id": "Risk",
          "questions": [
            {
              "id": "risk-rating",
              "response": "5",
              "errors": [],
            },
            {
              "id": "risk-choice",
              "response": "high",
              "errors": [],
            },
          ],
        },
        {
          "id": "Trust",
          "questions": [
            {
              "id": "trust-rating",
              "response": "1",
              "errors": [],
            },
          ],
        },
      ],
    },
    "errors": errors or [],
  }


def test_evaluate_basic_scores_groups_and_compares_human_baselines(
  tmp_path: Path,
) -> None:
  output_dir = tmp_path / "fixture-output"
  write_runner_output(
    output_dir,
    [
      record(llm_id="openai/gpt-4", population_index=0),
      record(
        llm_id="openai/gpt-4",
        population_index=1,
        sections=[
          {
            "id": "Risk",
            "questions": [
              {
                "id": "risk-rating",
                "response": "3",
                "errors": [],
              },
              {
                "id": "risk-choice",
                "response": "low",
                "errors": [],
              },
            ],
          },
          {
            "id": "Trust",
            "questions": [
              {
                "id": "trust-rating",
                "response": "3",
                "errors": [],
              },
            ],
          },
        ],
      ),
      record(llm_id="anthropic/claude", population_index=0),
    ],
  )

  result = evaluate_basic(output_dir, config_dir=FIXTURE_CONFIG_DIR)

  assert result.questionnaire_id == "evaluator-fixture"
  assert result.output_dir == output_dir
  assert result.model_section_means[("openai/gpt-4", "Risk")].mean == 5.5
  assert result.model_section_means[("openai/gpt-4", "Risk")].record_count == 2
  assert result.model_section_means[("anthropic/claude", "Risk")].mean == 8.0
  comparison = result.human_comparisons[("openai/gpt-4", "Risk")]
  assert comparison.human_average == 4.0
  assert comparison.delta == 1.5
  assert comparison.absolute_delta == 1.5


def test_evaluate_basic_excludes_question_level_runner_errors(
  tmp_path: Path,
) -> None:
  output_dir = tmp_path / "with-errors"
  write_runner_output(
    output_dir,
    [
      record(
        sections=[
          {
            "id": "Risk",
            "questions": [
              {
                "id": "risk-rating",
                "response": None,
                "errors": [{"question_id": "risk-rating", "message": "failed"}],
              },
              {
                "id": "risk-choice",
                "response": "high",
                "errors": [],
              },
            ],
          }
        ],
      )
    ],
  )

  result = evaluate_basic(output_dir, config_dir=FIXTURE_CONFIG_DIR)

  assert result.model_section_means[("openai/gpt-4", "Risk")].mean == 3.0


def test_evaluate_basic_accepts_choice_option_labels(tmp_path: Path) -> None:
  output_dir = tmp_path / "plan-labels"
  write_runner_output(
    output_dir,
    [
      record(
        questionnaire_name="plan-choice-fixture",
        sections=[
          {
            "id": "Plans",
            "questions": [
              {
                "id": "plan-choice",
                "response": "I choose Plan B.",
                "errors": [],
              }
            ],
          }
        ],
      )
    ],
  )

  result = evaluate_basic(output_dir, config_dir=FIXTURE_CONFIG_DIR)

  assert result.model_section_means[("openai/gpt-4", "Plans")].mean == 1.0


def test_evaluate_basic_writes_question_analysis_json(tmp_path: Path) -> None:
  output_dir = tmp_path / "question-analysis"
  write_runner_output(
    output_dir,
    [
      record(llm_id="openai/gpt-4", population_index=0),
      record(
        llm_id="openai/gpt-4",
        population_index=1,
        sections=[
          {
            "id": "Risk",
            "questions": [
              {
                "id": "risk-rating",
                "response": "3",
                "errors": [],
              },
              {
                "id": "risk-choice",
                "response": "low",
                "errors": [],
              },
            ],
          },
          {
            "id": "Trust",
            "questions": [
              {
                "id": "trust-rating",
                "response": "3",
                "errors": [],
              },
            ],
          },
        ],
      ),
      record(llm_id="anthropic/claude", population_index=0),
    ],
  )

  result = evaluate_basic(output_dir, config_dir=FIXTURE_CONFIG_DIR)
  payload = json.loads(result.question_analysis_json.read_text(encoding="utf-8"))

  assert result.question_analysis_json == output_dir / "question-analysis.json"
  assert payload == [
    {
      "questionnaire_id": "evaluator-fixture",
      "section_name": "Risk",
      "question_id": "risk-rating",
      "question_prompt": "Rate the risk.",
      "population": 3,
      "responses": [
        {"option": "1", "count": 0, "percentage": 0.0, "delta": None},
        {"option": "2", "count": 0, "percentage": 0.0, "delta": None},
        {
          "option": "3",
          "count": 1,
          "percentage": pytest.approx(1 / 3),
          "delta": None,
        },
        {"option": "4", "count": 0, "percentage": 0.0, "delta": None},
        {
          "option": "5",
          "count": 2,
          "percentage": pytest.approx(2 / 3),
          "delta": None,
        },
      ],
    },
    {
      "questionnaire_id": "evaluator-fixture",
      "section_name": "Risk",
      "question_id": "risk-choice",
      "question_prompt": "Choose a risk option.",
      "population": 3,
      "responses": [
        {
          "option": "low",
          "count": 1,
          "percentage": pytest.approx(1 / 3),
          "delta": None,
        },
        {
          "option": "high",
          "count": 2,
          "percentage": pytest.approx(2 / 3),
          "delta": None,
        },
      ],
    },
    {
      "questionnaire_id": "evaluator-fixture",
      "section_name": "Trust",
      "question_id": "trust-rating",
      "question_prompt": "Rate trust.",
      "population": 3,
      "responses": [
        {
          "option": "1",
          "count": 2,
          "percentage": pytest.approx(2 / 3),
          "delta": None,
        },
        {"option": "2", "count": 0, "percentage": 0.0, "delta": None},
        {
          "option": "3",
          "count": 1,
          "percentage": pytest.approx(1 / 3),
          "delta": None,
        },
        {"option": "4", "count": 0, "percentage": 0.0, "delta": None},
        {"option": "5", "count": 0, "percentage": 0.0, "delta": None},
      ],
    },
  ]


def test_evaluate_basic_writes_non_empty_pdf_charts(tmp_path: Path) -> None:
  output_dir = tmp_path / "chart-output"
  write_runner_output(output_dir, [record()])

  result = evaluate_basic(output_dir, config_dir=FIXTURE_CONFIG_DIR)

  assert result.section_scores_pdf == output_dir / "section-scores.pdf"
  assert result.section_delta_pdf == output_dir / "section-delta.pdf"
  assert result.section_scores_pdf.stat().st_size > 100
  assert result.section_delta_pdf.stat().st_size > 100


def test_evaluate_basic_adds_score_ticks_and_bar_values(tmp_path: Path) -> None:
  output_dir = tmp_path / "value-labels"
  write_runner_output(output_dir, [record()])

  result = evaluate_basic(output_dir, config_dir=FIXTURE_CONFIG_DIR)
  score_pdf = result.section_scores_pdf.read_bytes()
  delta_pdf = result.section_delta_pdf.read_bytes()

  for tick_label in ["0", "0.25", "0.5", "0.75", "1"]:
    assert tick_label.encode("latin-1") in score_pdf
  assert b"8.00" in score_pdf
  assert b"4.00" in score_pdf
  assert b"+4.00" in delta_pdf
  assert b"-1.00" in delta_pdf


def test_evaluate_basic_preserves_long_section_labels_in_charts(
  tmp_path: Path,
) -> None:
  section_name = "Group size sensitivity for human lives-P-6000life"
  output_dir = tmp_path / "long-labels"
  write_runner_output(
    output_dir,
    [
      record(
        questionnaire_name="long-section-fixture",
        sections=[
          {
            "id": section_name,
            "questions": [
              {
                "id": "long-choice",
                "response": "Plan B",
                "errors": [],
              }
            ],
          }
        ],
      )
    ],
  )

  result = evaluate_basic(output_dir, config_dir=FIXTURE_CONFIG_DIR)

  assert section_name.encode("latin-1") in result.section_scores_pdf.read_bytes()
  assert section_name.encode("latin-1") in result.section_delta_pdf.read_bytes()


def test_evaluate_basic_fails_on_mixed_questionnaires(tmp_path: Path) -> None:
  output_dir = tmp_path / "mixed"
  write_runner_output(
    output_dir,
    [
      record(questionnaire_name="evaluator-fixture"),
      record(questionnaire_name="other-fixture"),
    ],
  )

  with pytest.raises(EvaluatorError, match="multiple questionnaires"):
    evaluate_basic(output_dir, config_dir=FIXTURE_CONFIG_DIR)


def test_evaluate_basic_fails_on_unknown_section(tmp_path: Path) -> None:
  output_dir = tmp_path / "unknown-section"
  write_runner_output(
    output_dir,
    [record(sections=[{"id": "Unknown", "questions": []}])],
  )

  with pytest.raises(EvaluatorError, match="Unknown section"):
    evaluate_basic(output_dir, config_dir=FIXTURE_CONFIG_DIR)


def test_evaluate_basic_fails_on_unknown_question(tmp_path: Path) -> None:
  output_dir = tmp_path / "unknown-question"
  write_runner_output(
    output_dir,
    [
      record(
        sections=[
          {
            "id": "Risk",
            "questions": [
              {
                "id": "missing",
                "response": "5",
                "errors": [],
              }
            ],
          }
        ]
      )
    ],
  )

  with pytest.raises(EvaluatorError, match="Unknown question"):
    evaluate_basic(output_dir, config_dir=FIXTURE_CONFIG_DIR)
