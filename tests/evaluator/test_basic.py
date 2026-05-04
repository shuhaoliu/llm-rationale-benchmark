from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rationale_benchmark.evaluator.basic import EvaluatorError, evaluate_basic

FIXTURE_CONFIG_DIR = Path(__file__).parent / "fixtures" / "config"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
  path.write_text(
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
              "response": {
                "raw": json.dumps({"answer": 5}),
                "parsed": {"answer": 5},
              },
              "errors": [],
            },
            {
              "id": "risk-choice",
              "response": {
                "raw": json.dumps({"answer": "high"}),
                "parsed": {"answer": "high"},
              },
              "errors": [],
            },
          ],
        },
        {
          "id": "Trust",
          "questions": [
            {
              "id": "trust-rating",
              "response": {
                "raw": json.dumps({"answer": 1}),
                "parsed": {"answer": 1},
              },
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
  output_path = tmp_path / "fixture-output.jsonl"
  write_jsonl(
    output_path,
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
                "response": {"raw": json.dumps({"answer": 3}), "parsed": {"answer": 3}},
                "errors": [],
              },
              {
                "id": "risk-choice",
                "response": {
                  "raw": json.dumps({"answer": "low"}),
                  "parsed": {"answer": "low"},
                },
                "errors": [],
              },
            ],
          },
          {
            "id": "Trust",
            "questions": [
              {
                "id": "trust-rating",
                "response": {"raw": json.dumps({"answer": 3}), "parsed": {"answer": 3}},
                "errors": [],
              },
            ],
          },
        ],
      ),
      record(llm_id="anthropic/claude", population_index=0),
    ],
  )

  result = evaluate_basic(output_path, config_dir=FIXTURE_CONFIG_DIR)

  assert result.questionnaire_id == "evaluator-fixture"
  assert result.output_dir == tmp_path / "fixture-output"
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
  output_path = tmp_path / "with-errors.jsonl"
  write_jsonl(
    output_path,
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
                "response": {
                  "raw": json.dumps({"answer": "high"}),
                  "parsed": {"answer": "high"},
                },
                "errors": [],
              },
            ],
          }
        ],
      )
    ],
  )

  result = evaluate_basic(output_path, config_dir=FIXTURE_CONFIG_DIR)

  assert result.model_section_means[("openai/gpt-4", "Risk")].mean == 3.0


def test_evaluate_basic_accepts_choice_option_labels(tmp_path: Path) -> None:
  output_path = tmp_path / "plan-labels.jsonl"
  write_jsonl(
    output_path,
    [
      record(
        questionnaire_name="plan-choice-fixture",
        sections=[
          {
            "id": "Plans",
            "questions": [
              {
                "id": "plan-choice",
                "response": {"raw": "I choose Plan B.", "parsed": "I choose Plan B."},
                "errors": [],
              }
            ],
          }
        ],
      )
    ],
  )

  result = evaluate_basic(output_path, config_dir=FIXTURE_CONFIG_DIR)

  assert result.model_section_means[("openai/gpt-4", "Plans")].mean == 1.0


def test_evaluate_basic_writes_non_empty_pdf_charts(tmp_path: Path) -> None:
  output_path = tmp_path / "chart-output.jsonl"
  write_jsonl(output_path, [record()])

  result = evaluate_basic(output_path, config_dir=FIXTURE_CONFIG_DIR)

  assert result.section_scores_pdf == tmp_path / "chart-output" / "section-scores.pdf"
  assert result.section_delta_pdf == tmp_path / "chart-output" / "section-delta.pdf"
  assert result.section_scores_pdf.stat().st_size > 100
  assert result.section_delta_pdf.stat().st_size > 100


def test_evaluate_basic_adds_score_ticks_and_bar_values(tmp_path: Path) -> None:
  output_path = tmp_path / "value-labels.jsonl"
  write_jsonl(output_path, [record()])

  result = evaluate_basic(output_path, config_dir=FIXTURE_CONFIG_DIR)
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
  output_path = tmp_path / "long-labels.jsonl"
  write_jsonl(
    output_path,
    [
      record(
        questionnaire_name="long-section-fixture",
        sections=[
          {
            "id": section_name,
            "questions": [
              {
                "id": "long-choice",
                "response": {"raw": "Plan B", "parsed": "Plan B"},
                "errors": [],
              }
            ],
          }
        ],
      )
    ],
  )

  result = evaluate_basic(output_path, config_dir=FIXTURE_CONFIG_DIR)

  assert section_name.encode("latin-1") in result.section_scores_pdf.read_bytes()
  assert section_name.encode("latin-1") in result.section_delta_pdf.read_bytes()


def test_evaluate_basic_fails_on_mixed_questionnaires(tmp_path: Path) -> None:
  output_path = tmp_path / "mixed.jsonl"
  write_jsonl(
    output_path,
    [
      record(questionnaire_name="evaluator-fixture"),
      record(questionnaire_name="other-fixture"),
    ],
  )

  with pytest.raises(EvaluatorError, match="multiple questionnaires"):
    evaluate_basic(output_path, config_dir=FIXTURE_CONFIG_DIR)


def test_evaluate_basic_fails_on_unknown_section(tmp_path: Path) -> None:
  output_path = tmp_path / "unknown-section.jsonl"
  write_jsonl(output_path, [record(sections=[{"id": "Unknown", "questions": []}])])

  with pytest.raises(EvaluatorError, match="Unknown section"):
    evaluate_basic(output_path, config_dir=FIXTURE_CONFIG_DIR)


def test_evaluate_basic_fails_on_unknown_question(tmp_path: Path) -> None:
  output_path = tmp_path / "unknown-question.jsonl"
  write_jsonl(
    output_path,
    [
      record(
        sections=[
          {
            "id": "Risk",
            "questions": [
              {
                "id": "missing",
                "response": {"raw": json.dumps({"answer": 5}), "parsed": {"answer": 5}},
                "errors": [],
              }
            ],
          }
        ]
      )
    ],
  )

  with pytest.raises(EvaluatorError, match="Unknown question"):
    evaluate_basic(output_path, config_dir=FIXTURE_CONFIG_DIR)
