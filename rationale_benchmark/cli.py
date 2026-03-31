"""Command-line interface for the rationale benchmark tool."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import click
import structlog

from rationale_benchmark.llm import (
  ConnectorConfigLoader,
  LLMConnectorConfig,
  LLMConversation,
  ProviderRegistry,
)
from rationale_benchmark.llm.exceptions import ConfigurationError
from rationale_benchmark.llm.provider_client import BaseProviderClient
from rationale_benchmark.questionnaire import (
  Questionnaire,
  QuestionnaireConfigError,
  list_questionnaires,
  load_multiple,
)
from rationale_benchmark.runner import (
  BenchmarkResult,
  BenchmarkRunner,
  RunnerConfigError,
)


DEFAULT_CONFIG_DIR = Path("./config")
DEFAULT_LLM_CONFIG = "default-llms"
DEFAULT_MAX_CONCURRENCY = 4


class CliConfigurationError(click.ClickException):
  """Raised for configuration or selection errors exposed to the user."""

  exit_code = 2


class ConversationFactory:
  """Adapter creating :class:`LLMConversation` instances for the runner."""

  def __init__(
    self,
    configs: Mapping[str, LLMConnectorConfig],
    registry: ProviderRegistry | None = None,
  ) -> None:
    self._configs = dict(configs)
    self._registry = registry or ProviderRegistry()
    self._client_cache: dict[str, BaseProviderClient] = {}

  def create(
    self,
    model: str,
    *,
    system_prompt: str | None = None,
  ) -> LLMConversation:
    config = self._configs.get(model)
    if config is None:
      available = ", ".join(sorted(self._configs))
      raise ConfigurationError(
        f"Model '{model}' not configured. Available models: {available}"
      )
    client = self._client_for(config)
    prompt = system_prompt if system_prompt is not None else config.system_prompt
    return LLMConversation(
      config=config,
      provider_client=client,
      system_prompt=prompt,
    )

  def _client_for(self, config: LLMConnectorConfig) -> BaseProviderClient:
    cache_key = config.cache_key()
    if cache_key not in self._client_cache:
      self._client_cache[cache_key] = self._registry.create(config)
    return self._client_cache[cache_key]


def configure_logging(verbose: bool) -> None:
  """Configure structlog to emit JSON logs."""
  level = logging.DEBUG if verbose else logging.INFO
  structlog.configure(
    processors=[
      structlog.contextvars.merge_contextvars,
      structlog.processors.add_log_level,
      structlog.processors.TimeStamper(fmt="iso"),
      structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(level),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=True,
  )
  logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


def parse_questionnaire_selection(
  questionnaire: str | None,
  questionnaires: str | None,
  *,
  discovered: Iterable[str],
) -> list[str]:
  """Return the list of questionnaire IDs to execute."""
  if questionnaire and questionnaires:
    raise click.BadOptionUsage(
      option_name="--questionnaire",
      message="Cannot combine --questionnaire with --questionnaires.",
    )

  available = sorted(set(discovered))
  if questionnaire:
    return [questionnaire]

  if questionnaires:
    selected = []
    for item in questionnaires.split(","):
      stripped = item.strip()
      if not stripped:
        continue
      if stripped not in available:
        raise CliConfigurationError(
          f"Questionnaire '{stripped}' not found. "
          f"Available: {', '.join(available) or 'none'}."
        )
      selected.append(stripped)
    return selected

  return available


def list_llm_configurations(config_dir: Path) -> list[str]:
  """Return LLM configuration file stems located in ``config_dir/llms``."""
  root = config_dir / "llms"
  if not root.exists():
    return []
  names: list[str] = []
  for path in sorted(root.glob("*.yaml")):
    names.append(path.stem)
  return names


def load_llm_configurations(
  config_dir: Path,
  llm_config: str,
) -> dict[str, LLMConnectorConfig]:
  """Load and validate LLM configurations."""
  llm_dir = config_dir / "llms"
  if not llm_dir.is_dir():
    raise CliConfigurationError(f"LLM configuration directory missing: {llm_dir}")
  config_path = llm_dir / f"{llm_config}.yaml"
  loader = ConnectorConfigLoader()
  return loader.load(config_path)


def select_models(
  configs: Mapping[str, LLMConnectorConfig],
  requested: str | None,
) -> list[str]:
  """Return ordered model selectors for execution."""
  available = list(configs.keys())
  if not available:
    raise CliConfigurationError("No models defined in the selected LLM configuration.")

  if not requested:
    return sorted(available)

  selected: list[str] = []
  seen: set[str] = set()
  for item in requested.split(","):
    stripped = item.strip()
    if not stripped:
      continue
    if stripped not in configs:
      raise CliConfigurationError(
        f"Model '{stripped}' not found. "
        f"Available: {', '.join(sorted(available))}."
      )
    if stripped not in seen:
      selected.append(stripped)
      seen.add(stripped)
  return selected


def serialize_benchmark_result(result: BenchmarkResult) -> dict[str, Any]:
  """Convert a :class:`BenchmarkResult` into a JSON-serialisable mapping."""
  return _to_serialisable(result)


def _to_serialisable(value: Any) -> Any:
  if value is None:
    return None
  if isinstance(value, (str, int, float, bool)):
    return value
  if isinstance(value, Path):
    return str(value)
  if hasattr(value, "isoformat"):
    try:
      return value.isoformat()  # datetime compatible
    except Exception:  # pragma: no cover - defensive
      pass
  if hasattr(value, "to_dict"):
    return _to_serialisable(value.to_dict())
  if is_dataclass(value):
    result: dict[str, Any] = {}
    for field in fields(value):
      result[field.name] = _to_serialisable(getattr(value, field.name))
    return result
  if isinstance(value, Mapping):
    return {str(key): _to_serialisable(val) for key, val in value.items()}
  if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
    return [_to_serialisable(item) for item in value]
  return value


def render_summary(result: BenchmarkResult) -> str:
  """Render a concise textual summary for stderr."""
  questionnaire_list = ", ".join(result.info.questionnaires) or "none"
  model_lines = []
  summary = result.summary
  for model in result.info.models_tested:
    average = summary.average_scores_by_model.get(model, 0.0)
    model_lines.append(f"- {model}: {average:.2%} avg score")
  warnings = []
  if result.errors:
    warnings.append(
      f"- Encountered {len(result.errors)} execution or scoring warnings."
    )
  return "\n".join(
    [
      "Benchmark Summary",
      f"Questionnaires: {questionnaire_list}",
      f"Models tested: {', '.join(result.info.models_tested) or 'none'}",
      f"Total questions: {summary.total_questions}",
      "Model performance:",
      *model_lines,
      *(["Warnings:"] + warnings if warnings else []),
    ]
  )


def write_output(
  result: BenchmarkResult,
  *,
  output_path: str | None,
) -> None:
  """Write JSON output to stdout or a file."""
  payload = json.dumps(
    serialize_benchmark_result(result),
    indent=2,
    ensure_ascii=False,
  )
  if output_path:
    destination = Path(output_path)
    destination.write_text(payload + "\n", encoding="utf-8")
    click.echo(
      (
        f"Benchmark results written to {destination} "
        f"(questionnaires={len(result.info.questionnaires)}, "
        f"models={len(result.info.models_tested)})"
      ),
      err=True,
    )
    return
  click.echo(payload)


def execute_benchmark(
  questionnaires: list[Questionnaire],
  *,
  configs: Mapping[str, LLMConnectorConfig],
  models: list[str],
  llm_config: str | None,
  max_concurrency: int,
  total_population: int = 1,
  parallel_sessions: int = 1,
) -> BenchmarkResult:
  """Run the benchmark runner synchronously."""
  factory = ConversationFactory(configs)
  runner = BenchmarkRunner(
    factory,
    max_concurrency=max_concurrency,
  )
  return runner.run_sync(
    questionnaires,
    models,
    llm_config=llm_config,
    total_population=total_population,
    parallel_sessions=parallel_sessions,
  )


@click.command()
@click.option(
  "--questionnaire",
  help="Single questionnaire to run (filename without .yaml).",
)
@click.option(
  "--questionnaires",
  help="Comma-separated list of questionnaire IDs to run.",
)
@click.option(
  "--llm-config",
  default=DEFAULT_LLM_CONFIG,
  show_default=True,
  help="LLM configuration to use (filename without .yaml).",
)
@click.option(
  "--models",
  help="Comma-separated list of provider/model selectors to execute.",
)
@click.option(
  "--output",
  type=click.Path(),
  help="Optional path for JSON results (defaults to stdout).",
)
@click.option(
  "--list-questionnaires",
  "list_questionnaires_flag",
  is_flag=True,
  help="List available questionnaires and exit.",
)
@click.option(
  "--list-llm-configs",
  "list_llm_configs_flag",
  is_flag=True,
  help="List available LLM configuration files and exit.",
)
@click.option(
  "--config-dir",
  type=click.Path(),
  default=str(DEFAULT_CONFIG_DIR),
  show_default=True,
  help="Base configuration directory containing questionnaires/ and llms/ folders.",
)
@click.option(
  "--max-concurrency",
  type=click.INT,
  default=DEFAULT_MAX_CONCURRENCY,
  show_default=True,
  help="Maximum number of concurrent model executions.",
)
@click.option(
  "--total-population",
  type=click.INT,
  default=1,
  show_default=True,
  help="Total independent LLM completions to collect per questionnaire (population mode).",
)
@click.option(
  "--parallel-sessions",
  type=click.INT,
  default=1,
  show_default=True,
  help="Maximum concurrent LLM sessions when running in population mode.",
)
@click.option(
  "--verbose",
  is_flag=True,
  help="Enable verbose (DEBUG) logging.",
)
def main(
  questionnaire: str | None,
  questionnaires: str | None,
  llm_config: str,
  models: str | None,
  output: str | None,
  list_questionnaires_flag: bool,
  list_llm_configs_flag: bool,
  config_dir: str,
  max_concurrency: int,
  total_population: int,
  parallel_sessions: int,
  verbose: bool,
) -> None:
  """Rationale Benchmark for Large Language Models."""
  if max_concurrency < 1:
    raise click.BadParameter(
      "--max-concurrency must be >= 1",
      param_hint="--max-concurrency",
    )
  if total_population < 1:
    raise click.BadParameter(
      "--total-population must be >= 1",
      param_hint="--total-population",
    )
  if parallel_sessions < 1:
    raise click.BadParameter(
      "--parallel-sessions must be >= 1",
      param_hint="--parallel-sessions",
    )

  configure_logging(verbose)
  logger = structlog.get_logger(__name__)

  config_root = Path(config_dir).expanduser().resolve()
  if not config_root.exists():
    raise CliConfigurationError(f"Configuration directory not found: {config_root}")

  questionnaires_available = list_questionnaires(config_root)
  llm_configs_available = list_llm_configurations(config_root)

  if list_questionnaires_flag or list_llm_configs_flag:
    if list_questionnaires_flag:
      if questionnaires_available:
        for name in questionnaires_available:
          click.echo(name)
      else:
        click.echo("(no questionnaires found)")
    if list_llm_configs_flag:
      if llm_configs_available:
        for name in llm_configs_available:
          click.echo(name)
      else:
        click.echo("(no llm configurations found)")
    raise SystemExit(0)

  try:
    selected_questionnaires = parse_questionnaire_selection(
      questionnaire,
      questionnaires,
      discovered=questionnaires_available,
    )
  except CliConfigurationError as exc:
    raise exc

  if not selected_questionnaires:
    raise CliConfigurationError(
      "No questionnaires selected. Use --questionnaire or add files under "
      f"{config_root / 'questionnaires'}."
    )

  try:
    questionnaire_objects = load_multiple(selected_questionnaires, config_root)
  except QuestionnaireConfigError as exc:
    raise CliConfigurationError(str(exc)) from exc

  try:
    configs = load_llm_configurations(config_root, llm_config)
  except ConfigurationError as exc:
    raise CliConfigurationError(str(exc)) from exc

  try:
    selected_models = select_models(configs, models)
  except CliConfigurationError as exc:
    raise exc

  logger.info(
    "benchmark.start",
    questionnaires=selected_questionnaires,
    models=selected_models,
    llm_config=llm_config,
    total_population=total_population,
    parallel_sessions=parallel_sessions,
  )

  try:
    result = execute_benchmark(
      questionnaire_objects,
      configs=configs,
      models=selected_models,
      llm_config=llm_config,
      max_concurrency=max_concurrency,
      total_population=total_population,
      parallel_sessions=parallel_sessions,
    )
  except RunnerConfigError as exc:
    raise click.ClickException(str(exc)) from exc

  write_output(result, output_path=output)
  click.echo(render_summary(result), err=True)

  logger.info(
    "benchmark.complete",
    questionnaires=selected_questionnaires,
    models=selected_models,
    errors=len(result.errors),
  )

  exit_code = 1 if result.errors else 0
  raise SystemExit(exit_code)


if __name__ == "__main__":  # pragma: no cover
  main()
