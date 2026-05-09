#!/usr/bin/env python3
"""Ad-hoc conversation test for Dashscope's Qwen3 Max model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import click

from rationale_benchmark.llm.conversation import ConversationTurn, LLMResponse
from rationale_benchmark.llm.conversation_factory import (
  LLMConversationFactory,
)
from rationale_benchmark.llm.exceptions import (
  ConfigurationError,
  LLMConnectorError,
  ValidationFailedError,
)

DEFAULT_CONFIG_PATH = Path("config/llms/default-llms.yaml")
DEFAULT_MODEL_ALIAS = "dashscope/qwen3-max"
DEFAULT_QUESTIONS: tuple[str, ...] = (
  "Give a two-sentence summary of quantum error correction.",
  "List three creative uses for a paperclip and explain the reasoning.",
  "How would you describe the benefits of daily journaling to a teenager?",
)


def _response_schema(response_format: str) -> dict[str, Any]:
  if response_format == "json":
    answer_schema: dict[str, Any] = {
      "anyOf": [
        {"type": "object"},
        {"type": "array"},
        {"type": "string"},
        {"type": "number"},
        {"type": "integer"},
        {"type": "boolean"},
        {"type": "null"},
      ]
    }
  else:
    answer_schema = {"type": "string"}

  return {
    "type": "object",
    "additionalProperties": False,
    "properties": {"answer": answer_schema},
    "required": ["answer"],
    "title": "qwen_response",
  }


def _normalise_selector(
  factory: LLMConversationFactory,
  config_path: Path,
  selector: str,
) -> str:
  """Return a provider/model key present in ``config_path`` for ``selector``.

  Args:
    factory: Conversation factory whose loader inspects the configuration.
    config_path: Path to the YAML configuration file.
    selector: Provider/model alias supplied by the user.

  Returns:
    Provider/model key exactly as stored in the configuration.

  Raises:
    ConfigurationError: If ``selector`` cannot be resolved uniquely.
  """
  configs = factory.loader.load(config_path)
  if selector in configs:
    return selector

  provider_hint, _, model_name = selector.partition("/")
  if not model_name:
    model_name = provider_hint
    provider_hint = ""

  candidates: list[str] = []
  for key in configs:
    provider, _, candidate_model = key.partition("/")
    if candidate_model != model_name:
      continue
    if provider_hint and provider_hint not in provider:
      continue
    candidates.append(key)

  if not candidates:
    raise ConfigurationError(
      f"Model '{selector}' not found in {config_path}",
      config_file=str(config_path),
      field="models",
    )

  if len(candidates) > 1:
    options = ", ".join(sorted(candidates))
    raise ConfigurationError(
      (
        f"Model alias '{selector}' is ambiguous; matching keys: "
        f"{options}"
      ),
      config_file=str(config_path),
      field="models",
    )

  return candidates[0]


def _print_pair(
  index: int,
  question: str,
  answer: str,
  *,
  show_metadata: bool,
  metadata: dict[str, object] | None = None,
) -> None:
  """Pretty-print the question/answer pair as plain text."""
  click.echo(f"[{index}] Question:")
  click.echo(question)
  click.echo("Answer:")
  reply = answer.strip() or "(empty response)"
  click.echo(reply)
  if show_metadata and metadata:
    encoded = json.dumps(metadata, ensure_ascii=False)
    click.echo(f"Metadata: {encoded}")
  click.echo("-" * 60)


def _emit_error(message: str) -> None:
  """Write ``message`` to stderr."""
  click.echo(message, err=True)


def _resolve_questions(questions: Sequence[str]) -> Iterable[str]:
  """Return the questions to ask, falling back to defaults."""
  if questions:
    return questions
  return DEFAULT_QUESTIONS


def _conversation_pairs(
  turns: Sequence[ConversationTurn],
) -> list[tuple[ConversationTurn, ConversationTurn]]:
  """Return ordered user/assistant pairs from archived turns."""
  pairs: list[tuple[ConversationTurn, ConversationTurn]] = []
  pending_user: ConversationTurn | None = None
  for turn in turns:
    if turn.role == "user":
      if not turn.include_in_request:
        continue
      pending_user = turn
      continue
    if turn.role != "assistant":
      continue
    if pending_user is None:
      continue
    if not turn.include_in_request:
      continue
    pairs.append((pending_user, turn))
    pending_user = None
  return pairs


def _print_transcript(
  pairs: Sequence[tuple[ConversationTurn, ConversationTurn]],
  *,
  show_metadata: bool,
  metadata_map: dict[int, dict[str, object]],
) -> None:
  """Emit the conversation pairs indexed in chronological order."""
  for index, (user_turn, assistant_turn) in enumerate(pairs, start=1):
    metadata = metadata_map.get(index)
    _print_pair(
      index,
      user_turn.content,
      assistant_turn.content,
      show_metadata=show_metadata,
      metadata=metadata,
    )


@click.command()
@click.option(
  "--config",
  "config_path",
  type=click.Path(path_type=Path),
  default=DEFAULT_CONFIG_PATH,
  show_default=True,
  help="Path to the LLM configuration YAML.",
)
@click.option(
  "--model",
  "model_selector",
  default=DEFAULT_MODEL_ALIAS,
  show_default=True,
  help="Provider/model selector or alias to query.",
)
@click.option(
  "--question",
  "questions",
  multiple=True,
  help="Custom question to ask (can be specified multiple times).",
)
@click.option(
  "--json-output",
  is_flag=True,
  help="Write the results as a JSON array instead of formatted text.",
)
@click.option(
  "--max-attempts",
  type=int,
  default=None,
  help="Override the retry budget for each request.",
)
@click.option(
  "--show-metadata",
  is_flag=True,
  help="Include provider metadata in the formatted output.",
)
@click.option(
  "--response-format",
  type=click.Choice(["text", "json"]),
  default="text",
  show_default=True,
  help="Desired response format.",
)
def main(
  config_path: Path,
  model_selector: str,
  questions: Sequence[str],
  json_output: bool,
  max_attempts: int | None,
  show_metadata: bool,
  response_format: str,
) -> None:
  """Run a short conversation with the Dashscope Qwen3 Max model.

  Args:
    config_path: Path to the LLM configuration file.
    model_selector: Provider/model key or alias to select the configuration.
    questions: Questions supplied via the CLI.
    json_output: Emit JSON instead of human-readable text.
    max_attempts: Override for the connector retry budget.
    show_metadata: Whether to print provider metadata in text mode.
  """
  factory = LLMConversationFactory()
  try:
    resolved_selector = _normalise_selector(
      factory,
      config_path,
      model_selector,
    )
    conversation = factory.create_from_config(
      config_path,
      resolved_selector,
    )
  except ConfigurationError as exc:
    _emit_error(f"Configuration error: {exc}")
    sys.exit(1)
  except LLMConnectorError as exc:
    _emit_error(f"Connector setup failed: {exc}")
    sys.exit(1)

  prompts = list(_resolve_questions(questions))
  responses: list[LLMResponse] = []
  output_schema = _response_schema(response_format)

  try:
    for index, prompt in enumerate(prompts, start=1):
      response = conversation.ask(
        prompt,
        output_schema,
        max_attempts=max_attempts,
      )
      responses.append(response)
  except KeyboardInterrupt:
    _emit_error("\nInterrupted by user.")
    sys.exit(130)
  except ValidationFailedError as exc:
    _emit_error(f"Validation failed: {exc}")
    sys.exit(2)
  except LLMConnectorError as exc:
    _emit_error(f"Conversation failed: {exc}")
    sys.exit(2)

  archive = conversation.archive()
  pairs = _conversation_pairs(archive.turns)

  record_list: list[dict[str, object]] = []
  metadata_lookup: dict[int, dict[str, object]] = {}

  for index, ((user_turn, assistant_turn), response) in enumerate(
    zip(pairs, responses),
    start=1,
  ):
    metadata_lookup[index] = response.metadata
    record_list.append(
      {
        "index": index,
        "question": user_turn.content,
        "response_text": assistant_turn.content,
        "parsed": response.parsed,
        "metadata": response.metadata,
      }
    )

  if json_output:
    click.echo(json.dumps(record_list, ensure_ascii=False, indent=2))
  else:
    _print_transcript(
      pairs,
      show_metadata=show_metadata,
      metadata_map=metadata_lookup,
    )
    click.echo("Conversation complete.")


if __name__ == "__main__":
  main()
