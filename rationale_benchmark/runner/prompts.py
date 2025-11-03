"""Prompt rendering utilities used by the benchmark runner."""

from __future__ import annotations

from rationale_benchmark.runner.types import PromptContext


def build_prompt(context: PromptContext) -> str:
  """Render a prompt for a questionnaire question."""

  return context.question.prompt


__all__ = ["build_prompt"]
