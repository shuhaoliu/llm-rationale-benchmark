from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from rationale_benchmark.questionnaire.validate_cli import cli


def _write_yaml(root: Path, name: str, content: str) -> Path:
  path = root / f"{name}.yaml"
  path.write_text(dedent(content), encoding="utf-8")
  return path


def test_cli_validates_file(tmp_path: Path) -> None:
  dir_path = tmp_path / "questionnaires"
  dir_path.mkdir()
  file_path = _write_yaml(
    dir_path,
    "sample",
    """
    questionnaire:
      id: "sample"
      name: "Sample"
      sections:
        - name: "One"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "Prompt"
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
    """,
  )
  runner = CliRunner()
  result = runner.invoke(cli, [str(file_path)])
  assert result.exit_code == 0
  assert "OK" in result.output


def test_cli_supports_json_output(tmp_path: Path) -> None:
  dir_path = tmp_path / "questionnaires"
  dir_path.mkdir()
  file_path = _write_yaml(
    dir_path,
    "json-sample",
    """
    questionnaire:
      id: "json-sample"
      name: "Sample"
      sections:
        - name: "One"
          questions:
            - id: "q1"
              type: "rating-5"
              prompt: "Prompt"
              scoring:
                total: 5
                weights: [0, 1, 2, 3, 4]
    """,
  )
  runner = CliRunner()
  result = runner.invoke(cli, [str(file_path), "--json"])
  assert result.exit_code == 0
  payload = json.loads(result.output)
  assert payload["status"] == "ok"
  assert payload["results"][0]["questionnaire_id"] == "json-sample"


def test_cli_reports_validation_errors(tmp_path: Path) -> None:
  dir_path = tmp_path / "questionnaires"
  dir_path.mkdir()
  file_path = _write_yaml(
    dir_path,
    "invalid",
    """
    questionnaire:
      id: "invalid"
      name: "Broken"
      sections:
        - name: "One"
          questions:
            - id: "q1"
              type: "choice"
              prompt: "Prompt"
              scoring:
                total: 2
                weights:
                  a: 0
    """,
  )
  runner = CliRunner()
  result = runner.invoke(cli, [str(file_path)])
  assert result.exit_code == 1
  assert "choice" in result.output.lower()
