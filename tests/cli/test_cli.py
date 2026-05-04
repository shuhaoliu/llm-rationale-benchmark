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


def _jsonl_records(text: str) -> list[dict[str, object]]:
  records = []
  for line in text.splitlines():
    if not line.strip().startswith("{"):
      continue
    payload = json.loads(line)
    if "llm_id" in payload and "response" in payload:
      records.append(payload)
  return records


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


def test_run_benchmark_emits_jsonl_and_summary(
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
  records = _jsonl_records(result.output)
  assert len(records) == 3
  assert {record["population_index"] for record in records} == {0, 1, 2}
  assert all(record["questionnaire"]["name"] == "sample" for record in records)
  assert all(record["llm_id"] == "openai/stub-model" for record in records)
  assert records[0]["response"]["sections"][0]["questions"][0]["id"] == "q1"
  assert "Run Summary" in result.output


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
      "--max-concurrency",
      "1",
    ],
  )
  assert result.exit_code == 0
  records = _jsonl_records(result.output)
  assert len(records) == 2
  assert {record["population_index"] for record in records} == {0, 1}


def test_output_option_writes_jsonl_file(
  tmp_path: Path,
  runner: CliRunner,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _write_questionnaire(tmp_path, "sample")
  _write_llm_config(tmp_path)
  output_path = tmp_path / "responses.jsonl"
  monkeypatch.setattr("rationale_benchmark.cli.ProviderRegistry", DummyRegistry)
  result = runner.invoke(
    main,
    [
      "--config-dir",
      str(tmp_path),
      "--questionnaire",
      "sample",
      "--total-population",
      "1",
      "--output",
      str(output_path),
    ],
  )
  assert result.exit_code == 0
  assert _jsonl_records(result.output) == []
  records = _jsonl_records(output_path.read_text(encoding="utf-8"))
  assert len(records) == 1
  assert records[0]["questionnaire"]["name"] == "sample"


def test_explicit_llm_config_does_not_merge_default_models(
  tmp_path: Path,
  runner: CliRunner,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _write_questionnaire(tmp_path, "sample")
  _write_llm_config(tmp_path)
  llm_path = tmp_path / "llms" / "aliyun.yaml"
  llm_path.write_text(
    dedent(
      """
      defaults:
        response_format: text
      providers:
        aliyun_openai_compatible:
          api_key: "test"
          base_url: "https://example.test/v1"
          models:
            - "qwen-test"
      """
    ),
    encoding="utf-8",
  )
  monkeypatch.setattr("rationale_benchmark.cli.ProviderRegistry", DummyRegistry)

  result = runner.invoke(
    main,
    [
      "--config-dir",
      str(tmp_path),
      "--questionnaire",
      "sample",
      "--llm-config",
      "aliyun",
      "--total-population",
      "1",
    ],
  )

  assert result.exit_code == 0
  records = _jsonl_records(result.output)
  assert [record["llm_id"] for record in records] == [
    "aliyun_openai_compatible/qwen-test"
  ]


def test_parallel_sessions_option_is_not_supported(
  tmp_path: Path,
  runner: CliRunner,
) -> None:
  _write_questionnaire(tmp_path, "sample")
  _write_llm_config(tmp_path)
  result = runner.invoke(
    main,
    [
      "--config-dir",
      str(tmp_path),
      "--questionnaire",
      "sample",
      "--parallel-sessions",
      "1",
    ],
  )
  assert result.exit_code == 2
  assert "No such option" in result.output


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
