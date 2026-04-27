from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner

from rationale_benchmark.cli import main
from rationale_benchmark.llm.config.connector_models import (
  LLMConnectorConfig,
  ResponseFormat,
)
from rationale_benchmark.llm.provider_client import (
  BaseProviderClient,
  ProviderResponse,
)


class DummyProvider(BaseProviderClient):
  """Provider returning deterministic JSON payloads for testing."""

  def _generate(
    self,
    messages,
    *,
    response_format: ResponseFormat,
  ) -> ProviderResponse:
    return ProviderResponse(
      content='{"answer": 3, "reasoning": "Test rationale"}',
      raw={"messages": messages},
    )


class DummyRegistry:
  """Registry that always returns ``DummyProvider`` instances."""

  def __init__(self, *args, **kwargs):
    pass

  def create(self, config: LLMConnectorConfig) -> BaseProviderClient:
    return DummyProvider(config)


@pytest.fixture()
def runner() -> CliRunner:
  return CliRunner()


def _extract_json_payload(text: str) -> dict[str, object]:
  lines = text.splitlines()
  captured: list[str] = []
  inside = False
  depth = 0

  for line in lines:
    stripped = line.strip()
    if not inside:
      if stripped == "{":
        inside = True
        depth = 1
        captured.append(line)
      continue
    depth += stripped.count("{")
    depth -= stripped.count("}")
    captured.append(line)
    if depth == 0:
      break

  payload = "\n".join(captured)
  if not payload:
    raise AssertionError("JSON payload not found in CLI output")
  return json.loads(payload)


def _write_questionnaire(root: Path, name: str) -> None:
  questionnaires = root / "questionnaires"
  questionnaires.mkdir(parents=True, exist_ok=True)
  path = questionnaires / f"{name}.yaml"
  path.write_text(
    dedent(
      """
      questionnaire:
        id: "sample"
        name: "Sample Questionnaire"
        description: "Test questionnaire"
        system_prompt: "Answer carefully."
        metadata:
          default_population: 3
        sections:
          - name: "Section"
            instructions: "Rate the statements."
            questions:
              - id: "q1"
                type: "rating-5"
                prompt: "Select a rating."
                scoring:
                  total: 5
                  weights: [0, 1, 2, 3, 5]
      """
    ),
    encoding="utf-8",
  )


def _write_llm_config(root: Path) -> None:
  llms = root / "llms"
  llms.mkdir(parents=True, exist_ok=True)
  path = llms / "default-llms.yaml"
  path.write_text(
    dedent(
      """
      defaults:
        timeout: 10
        max_retries: 1
      providers:
        openai:
          api_key: "test"
          models:
            - "stub-model"
      """
    ),
    encoding="utf-8",
  )


def test_list_questionnaires_outputs_available_names(
  tmp_path: Path,
  runner: CliRunner,
) -> None:
  _write_questionnaire(tmp_path, "sample")
  result = runner.invoke(
    main,
    [
      "--config-dir",
      str(tmp_path),
      "--list-questionnaires",
    ],
  )
  assert result.exit_code == 0
  assert "sample" in result.output


def test_run_benchmark_emits_json_and_summary(
  tmp_path: Path,
  runner: CliRunner,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _write_questionnaire(tmp_path, "sample")
  _write_llm_config(tmp_path)
  monkeypatch.setattr("rationale_benchmark.cli.ProviderRegistry", DummyRegistry)
  result = runner.invoke(
    main,
    [
      "--config-dir",
      str(tmp_path),
      "--questionnaire",
      "sample",
      "--max-concurrency",
      "1",
    ],
  )
  assert result.exit_code == 0
  payload = _extract_json_payload(result.output)
  assert payload["info"]["questionnaires"] == ["sample"]
  assert payload["info"]["total_population"] == 3
  assert payload["info"]["total_population_by_questionnaire"] == {"sample": 3}
  assert payload["summary"]["total_questions"] == 1
  assert payload["summary"]["models_tested"] == 1
  assert "Benchmark Summary" in result.output


def test_total_population_option_overrides_questionnaire_default(
  tmp_path: Path,
  runner: CliRunner,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _write_questionnaire(tmp_path, "sample")
  _write_llm_config(tmp_path)
  monkeypatch.setattr("rationale_benchmark.cli.ProviderRegistry", DummyRegistry)
  result = runner.invoke(
    main,
    [
      "--config-dir",
      str(tmp_path),
      "--questionnaire",
      "sample",
      "--total-population",
      "2",
      "--parallel-sessions",
      "1",
      "--max-concurrency",
      "1",
    ],
  )
  assert result.exit_code == 0
  payload = _extract_json_payload(result.output)
  assert payload["info"]["total_population"] == 2
  assert payload["info"]["total_population_by_questionnaire"] == {"sample": 2}


def test_missing_questionnaire_returns_exit_code_two(
  tmp_path: Path,
  runner: CliRunner,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _write_llm_config(tmp_path)
  monkeypatch.setattr("rationale_benchmark.cli.ProviderRegistry", DummyRegistry)
  result = runner.invoke(
    main,
    [
      "--config-dir",
      str(tmp_path),
      "--questionnaire",
      "missing",
    ],
  )
  assert result.exit_code == 2
  assert "Questionnaire" in result.output
