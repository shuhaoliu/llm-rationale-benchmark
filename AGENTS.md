# Agent Guidelines

This reference keeps agents aligned on the core practices for the rationale benchmark project.

## Python Development

- Indent with 2 spaces, keep lines ≤88 chars, favor double-quoted f-strings, and break lines with parentheses.
- Group imports as standard library, third-party, then local modules; avoid wildcards and prefer absolute paths.
- Follow naming conventions (`snake_case` functions/variables, `UPPER_SNAKE_CASE` constants, `PascalCase` classes).
- Keep functions focused, add type hints for params and returns, and document public APIs with Google-style docstrings.
- Consult the relevant material under `docs/` before authoring or modifying code.

## Testing & Quality

- Plan unit and mock tests before implementing features; tests must land with the change.
- Use pytest with Arrange-Act-Assert structure, descriptive names, and fixtures mirroring source layout in `tests/`.
- Prefer generation of small, independent tests and ensure new behavior is validated.

## Dependencies & Environment

- Manage everything with `uv`: check `pyproject.toml` before adding packages, use `uv add`/`uv add --dev`, and run commands via `uv run`.
- Maintain a project virtual environment with `uv venv`; keep `.venv/` ignored and document setup steps in the README when they change.

## Workflow

- Create feature branches using `feat/`, `bugfix/`, `refactor/`, or `docs/` prefixes.
- Write conventional commits (`feat:`, `fix:`, `docs:`, `test:`) and ensure PRs ship with tests, documentation updates, and a full local test run.
- During reviews, prioritize correctness, regression risk, and missing coverage over stylistic feedback.

## Error Handling & Documentation

- Raise specific exceptions, encapsulate domain errors with custom types, and add context when logging.
- Explain complex logic paths and keep README content current with shipped capabilities.

## CLI & Runtime Expectations

- Use Click for CLI surfaces, provide clear help text, support JSON output, and handle interrupts gracefully.
- Offer progress signals for long tasks, and keep option naming consistent between commands.

## Performance & Reliability

- Respect provider limits with exponential backoff and connection pooling.
- Stream or use generators for large payloads, release resources promptly, and monitor memory usage in tests.
- Constrain async concurrency with semaphores, catch timeouts, and ensure retries are idempotent.

## Additional References

- `docs/architecture.md`: module responsibilities and execution flow.
- `docs/interfaces.md`: data models, provider integrations, and expected behaviors.
- `docs/configs.md`: configuration formats, validation rules, and environment variable policy.
