"""Prompt rendering utilities used by the benchmark runner."""

from __future__ import annotations

import json

from rationale_benchmark.questionnaire.output_schema import resolve_output_schema
from rationale_benchmark.runner.types import PromptContext


def build_prompt(context: PromptContext) -> str:
  """Render a prompt for a questionnaire question."""

  prompt_parts: list[str] = []
  if context.section_instructions:
    prompt_parts.append(context.section_instructions.strip())
  prompt_parts.append(context.question.prompt.strip())
  schema_text = json.dumps(
    resolve_output_schema(context.question),
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
  )
  prompt_parts.append(
    "Respond with JSON that matches this schema exactly:\n"
    f"{schema_text}"
  )
  return "\n\n".join(prompt_parts)


__all__ = ["build_prompt"]
