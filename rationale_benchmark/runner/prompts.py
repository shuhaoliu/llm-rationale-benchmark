"""Prompt rendering utilities used by the benchmark runner."""

from __future__ import annotations

from rationale_benchmark.runner.types import PromptContext


def build_prompt(context: PromptContext) -> str:
  """Render a prompt for a questionnaire question."""

  prompt_parts: list[str] = []
  if context.section_instructions:
    prompt_parts.append(context.section_instructions.strip())
  prompt_parts.append(context.question.prompt.strip())
  return "\n\n".join(prompt_parts)


__all__ = ["build_prompt"]
