# Agent Guidelines

This document keeps agents aligned on the core practices for a standard
Python project.

## Working Style

Behavioral guidelines to reduce common LLM coding mistakes. These should be
merged with any project-specific instructions when they exist.

Tradeoff: these guidelines bias toward caution over speed. For trivial tasks,
use judgment.

### Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

- State assumptions explicitly before implementing.
- If multiple interpretations exist, present them instead of picking one
  silently.
- Call out simpler approaches when they exist, and push back on unnecessary
  complexity.
- If something is unclear, stop and name the confusion before proceeding.

### Simplicity First

Minimum code that solves the problem. Nothing speculative.

- Do not add features beyond what was requested.
- Avoid abstractions for single-use code.
- Do not add configurability or flexibility unless it was requested.
- Avoid error handling for impossible scenarios.
- If a solution feels overcomplicated, simplify it.

### Surgical Changes

Touch only what you must. Clean up only your own mess.

- Do not "improve" adjacent code, comments, or formatting unless the task
  requires it.
- Do not refactor unrelated code.
- Match the existing style of the codebase.
- If you notice unrelated dead code, mention it instead of deleting it.
- Remove imports, variables, and functions made unused by your own changes.

### Goal-Driven Execution

Define success criteria and verify them.

- Turn requests into verifiable goals before implementing.
- For bug fixes, reproduce the issue with a test first when practical.
- For feature work, decide how the behavior will be validated before coding.
- For multi-step tasks, write a short plan and pair each step with a concrete
  verification check.

## Python Development

- Indent with 2 spaces.
- Keep lines at 88 characters or fewer.
- Favor double-quoted f-strings.
- Break long expressions with parentheses instead of line continuations.
- Group imports as standard library, third-party, then local modules.
- Avoid wildcard imports and prefer absolute imports.
- Follow naming conventions:
  - `snake_case` for functions and variables
  - `UPPER_SNAKE_CASE` for constants
  - `PascalCase` for classes
- Keep functions focused and small.
- Add type hints for parameters and return values.
- Document public APIs with Google-style docstrings.
- Consult relevant material under `docs/` before authoring or modifying code.

## Testing & Quality

- Plan unit and mock tests before implementing features.
- Tests should land in the same change as the code they validate.
- Use `pytest`.
- Structure tests with Arrange-Act-Assert.
- Use descriptive test names.
- Mirror source layout under `tests/` and use fixtures where they help.
- Prefer small, independent tests.
- Ensure every new behavior is validated.

## Dependencies & Environment

- Manage dependencies with `uv`.
- Check `pyproject.toml` before adding packages.
- Use `uv add` for runtime dependencies.
- Use `uv add --dev` for development dependencies.
- Run project commands through `uv run`.
- Maintain the project environment with `uv venv`.
- Keep `.venv/` ignored.
- Update the README when setup steps change.

## Workflow

- Create branches with one of these prefixes:
  - `feat/`
  - `bugfix/`
  - `refactor/`
  - `docs/`
- Use conventional commits such as:
  - `feat:`
  - `fix:`
  - `docs:`
  - `test:`
- Ship tests and documentation updates with code changes when applicable.
- Run the full local test suite before opening a PR.
- During review, prioritize correctness, regression risk, and missing coverage
  over stylistic feedback.

## Error Handling & Documentation

- Raise specific exceptions.
- Encapsulate domain errors with custom exception types.
- Add useful context when logging errors.
- Explain complex logic paths with concise documentation.
- Keep README content aligned with shipped capabilities.

## CLI & Runtime Expectations

- Use Click for CLI surfaces.
- Provide clear help text.
- Support JSON output where it makes sense.
- Handle interrupts gracefully.
- Offer progress signals for long-running tasks.
- Keep option naming consistent between commands.
