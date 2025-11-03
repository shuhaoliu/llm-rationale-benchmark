"""Validate questionnaire YAML files using the project CLI wrapper.

This entry point delegates to ``rationale_benchmark.questionnaire.validate_cli``
so the repository exposes a stable ``bin/`` command. Invoke the script without
arguments to scan ``config/questionnaires`` for ``*.yaml`` files; pass explicit
paths to validate only the listed files. Validation failures are written to
stderr and the process exits non-zero.

Example:
  uv run python bin/validate_questionnaire.py --json config/questionnaires/*.yaml

See ``uv run python bin/validate_questionnaire.py --help`` for option details.
"""

from rationale_benchmark.questionnaire.validate_cli import main


if __name__ == "__main__":
  main()
