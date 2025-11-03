# Rationale Benchmark CLI Specification

## Purpose
The command-line interface (CLI) is the primary entry point for running the
rationale benchmark end to end. It packages questionnaire discovery, LLM
configuration selection, benchmark execution, and result emission behind a
single executable (`rationale-benchmark`). The CLI must stay scriptable for
automation, expose informative errors, and align with project-wide configuration
and logging policies.

## Command Surface

### Entry Point
- Installed console script: `rationale-benchmark`.
- Preferred invocation during development: `uv run rationale-benchmark`.
- Python module entry point: `rationale_benchmark.cli:main`.

### Core Workflow
1. Parse CLI options and environment variables.
2. Resolve configuration directories (`config/questionnaires/`,
   `config/llms/`) with optional overrides.
3. Load the requested questionnaire definitions and LLM configuration via the
   configuration and questionnaire modules.
4. Instantiate the runner with validated inputs, concurrency parameters, and
   output settings.
5. Execute questionnaires against each selected model.
6. Persist JSON results (file or stdout) and render a human-readable summary.
7. Exit with `0` on success, non-zero for validation or runtime failures.

### Options
- `--questionnaire <name>`: Run a single questionnaire (stem of `.yaml` file).
- `--questionnaires <names>`: Comma-separated list of questionnaire IDs; cannot
  be combined with `--questionnaire`.
- `--llm-config <name>`: LLM configuration file stem (default: `default-llms`).
- `--models <provider/model,...>`: Optional filter limiting execution to the
  listed models present in the chosen LLM config.
- `--config-dir <path>`: Override base directory holding `llms/` and
  `questionnaires/` subdirectories (defaults to `./config`).
- `--output <path>`: Write JSON report to a specific file; omit to print to
  stdout.
- `--list-questionnaires`: List discovered questionnaire IDs and exit.
- `--list-llm-configs`: List available LLM configuration stems and exit.
- `--verbose`: Elevate logging to verbose/DEBUG mode.
- `--max-concurrency <int>`: (Planned) Override default concurrency ceiling
  passed to the runner.
- `--help`: Show usage.

When mutually exclusive options are supplied together (e.g., both
`--questionnaire` and `--questionnaires`), the CLI must fail with status `2`
and display a concise error along with the usage text.

## Configuration Resolution
- `config_dir` defaults to `cwd/config`; `--config-dir` allows custom roots.
- Questionnaire directory: `${config_dir}/questionnaires/`. Files must end with
  `.yaml`. IDs equal filename stems (`moral-reasoning.yaml` → `moral-reasoning`).
- LLM configuration directory: `${config_dir}/llms/`. `default-llms.yaml` must
  exist. Overrides merge onto the default per rules in `docs/configs.md`.
- Environment variable placeholders (`${ENV_VAR}`) resolve during load; missing
  variables raise `ConfigurationError` before execution.
- Listing commands (`--list-questionnaires`, `--list-llm-configs`) reuse the
  same discovery logic but skip validation that requires full execution (e.g.,
  they do not load questionnaires when only listing LLM configs).

## Questionnaire Selection Semantics
- Without selection flags, run all questionnaires discovered in
  `questionnaires/`, sorted by filename for determinism.
- `--questionnaire` accepts a single ID; `--questionnaires` parses a
  comma-delimited list and removes whitespace.
- Each requested questionnaire must exist; missing IDs produce
  `QuestionnaireConfigError` identifying the first missing file.
- Loaded questionnaires undergo schema and semantic validation defined in
  `docs/02-questionnaire/design.md` before execution.

## LLM Configuration Semantics
- Always load `default-llms.yaml`; overlay the file named by `--llm-config`.
- Validate provider keys, required fields, and environment substitution as
  described in `docs/01-llm/design.md` and `docs/configs.md`.
- If `--models` is provided, filter the merged configuration to the specified
  provider/model selectors. Unknown selectors raise `RunnerConfigError` and
  list known options from the active config.
- Provide deterministic ordering: models sorted alphabetically unless an
  override conveys priority metadata.

## Runner Integration
- Instantiate `BenchmarkRunner` with:
  - Validated questionnaires.
  - `LLMConversationFactory` derived from the merged LLM config.
  - Concurrency limit (default derived from provider hints; overridden by
    `--max-concurrency` when available).
  - Output descriptors (target path, stdout flag).
- Propagate CLI `--verbose` flag to logging and to runner diagnostics so
  question-level events surface when requested.
- Gracefully handle `KeyboardInterrupt`: persist any completed transcripts and
  exit with status `130`.

## Output Specification
- JSON report structure mirrors `docs/architecture.md` (`benchmark_info`,
  `results`, `summary`).
- Default behavior writes the JSON payload to stdout; when `--output` is
  supplied, write to the provided path and emit a confirmation message to stderr
  summarizing location and key metrics (e.g., questionnaires run, model count).
- CLI summary prints a table or bullet list showing per-model average score,
  questionnaires executed, and any warnings (e.g., retries, validator failures).
- Respect project logging policy: structured JSON logs at INFO level; human
  summaries restricted to stdout/stderr separation for shell scripting.

## Error Handling
- Input validation errors (unknown questionnaire, malformed option, missing
  config) exit with status `2`.
- Configuration loading errors raise `ConfigurationError` or
  `QuestionnaireConfigError`; report source file, key path, and guidance.
- Runtime errors within the runner return status `1`. CLI must surface a
  concise summary plus pointer to detailed logs or partial outputs.
- On partial success (some models fail), include failure reasons in the JSON
  output and mark the process exit code as non-zero while preserving successful
  model results.

## Environment & Secrets
- Do not read secrets from files by default; rely on `os.environ`.
- Encourage `.env` usage via README instructions; CLI should warn when an
  expected variable is missing rather than empty.
- Support `python-dotenv` integration by respecting environment already loaded
  by `uv run python -m dotenv run --`.

## Progress and Observability
- Provide incremental logging for questionnaire/model progress. Minimal default:
  emit per-model start/finish logs. Verbose mode includes per-question attempts,
  retries, and latency metrics.
- Optionally integrate with `runner` callbacks for progress bars once available,
  gated behind verbose flag to avoid polluting machine-readable output.

## Testing Expectations
- Unit tests cover:
  - Option parsing and mutually exclusive arguments.
  - Listing commands producing deterministic output.
  - Configuration resolution errors (missing files, unresolved environment
    variables).
- Integration tests (with fixtures and mock providers) verify full execution
  path, JSON output schema, and exit codes for mixed-success scenarios.
- Tests run via `uv run pytest`, organized under `tests/cli/`.

## Extensibility Guidelines
- Additional commands or subcommands must preserve compatibility with existing
  flags. Prefer additive flags over breaking changes.
- When introducing new outputs (e.g., CSV), gate behind explicit options to
  avoid altering default JSON payload.
- Keep CLI logic thin; push new business rules into configuration loaders,
  questionnaire validators, or runner components to maintain modularity.
