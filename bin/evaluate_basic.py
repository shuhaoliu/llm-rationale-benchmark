#!/usr/bin/env python
"""Run the basic evaluator over one runner JSONL output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rationale_benchmark.evaluator import EvaluatorError, evaluate_basic


def main() -> int:
  """Evaluate one runner output file and report chart paths."""
  parser = argparse.ArgumentParser(
    description="Evaluate one raw runner JSONL output file.",
  )
  parser.add_argument("runner_output", help="JSONL output emitted by the runner")
  args = parser.parse_args()

  try:
    result = evaluate_basic(Path(args.runner_output))
  except EvaluatorError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1

  print(f"Wrote {result.section_scores_pdf}")
  print(f"Wrote {result.section_delta_pdf}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
