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
4. Resolve the effective population count for each questionnaire.
5. Instantiate one runner execution per questionnaire with validated inputs,
   concurrency parameters, and output settings.
6. Execute each questionnaire against the selected models.
7. Persist raw JSONL response records and render a human-readable summary.
8. Exit with `0` on success, non-zero for validation or runtime failures.

### Options
- `--questionnaire <name>`: Run a single questionnaire (stem of `.yaml` file).
- `--questionnaires <names>`: Comma-separated list of questionnaire IDs; cannot
  be combined with `--questionnaire`.
- `--llm-config <name>`: LLM configuration file stem (default: `default-llms`).
- `--models <provider/model,...>`: Optional filter limiting execution to the
  listed models present in the chosen LLM config.
- `--config-dir <path>`: Override base directory holding `llms/` and
  `questionnaires/` subdirectories (defaults to `./config`).
- `--output <path>`: Write JSONL response records to a specific file; omit to
  print to stdout.
- `--list-questionnaires`: List discovered questionnaire IDs and exit.
- `--list-llm-configs`: List available LLM configuration stems and exit.
- `--verbose`: Elevate logging to verbose/DEBUG mode.
- `--total-population <int>`: Optional positive integer overriding each
  questionnaire's `metadata.default_population`.
- `--max-concurrency <int>`: Maximum concurrent provider calls across LLMs,
  population members, and questionnaire sections; defaults to `5`.
- `--help`: Show usage.

When mutually exclusive options are supplied together (e.g., both
`--questionnaire` and `--questionnaires`), the CLI must fail with status `2`
and display a concise error along with the usage text.

## Configuration Resolution
- `config_dir` defaults to `cwd/config`; `--config-dir` allows custom roots.
- Questionnaire directory: `${config_dir}/questionnaires/`. Files must end with
  `.yaml`. IDs equal filename stems (`moral-reasoning.yaml` → `moral-reasoning`).
- LLM configuration directory: `${config_dir}/llms/`. `default-llms.yaml` is
  used only when `--llm-config` is omitted or set to `default-llms`. Selecting
  another file runs only the providers and models declared in that file.
- Environment variable placeholders (`${ENV_VAR}`) resolve during load; missing
  variables raise `ConfigurationError` before execution.
- Listing commands (`--list-questionnaires`, `--list-llm-configs`) reuse the
  same discovery logic but skip validation that requires full execution (e.g.,
  they do not load questionnaires when only listing LLM configs).

## Questionnaire Selection Semantics
- Without selection flags, run all questionnaires discovered in
  `questionnaires/`, sorted by filename for determinism. The CLI invokes the
  runner separately for each questionnaire because one runner execution accepts
  only one questionnaire.
- `--questionnaire` accepts a single ID; `--questionnaires` parses a
  comma-delimited list and removes whitespace. Multiple selected questionnaires
  produce multiple runner executions.
- Each requested questionnaire must exist; missing IDs produce
  `QuestionnaireConfigError` identifying the first missing file.
- Loaded questionnaires undergo schema and semantic validation defined in
  `docs/02-questionnaire/design.md` before execution.
- Each questionnaire must define a positive `metadata.default_population`.
  When `--total-population` is supplied, it must be positive and overrides that
  metadata default for the current run.

## LLM Configuration Semantics
- Load the file named by `--llm-config`; do not merge models from other LLM
  configuration files.
- Validate provider keys, required fields, and environment substitution as
  described in `docs/01-llm/design.md` and `docs/configs.md`.
- If `--models` is provided, filter the merged configuration to the specified
  provider/model selectors. Unknown selectors raise `RunnerConfigError` and
  list known options from the active config.
- Provide deterministic ordering: models sorted alphabetically unless an
  override conveys priority metadata.

## Runner Integration
- Instantiate `BenchmarkRunner` with:
  - One validated questionnaire.
  - `LLMConversationFactory` derived from the merged LLM config.
  - Effective `total_population` for that questionnaire, resolved from
    `--total-population` or `metadata.default_population`.
  - Provider-call concurrency limit from `--max-concurrency`, defaulting to
    `5`.
  - Output descriptors (target path, stdout flag).
- The runner may query different LLMs, population members, and sections in
  parallel. The theoretical maximum parallelism for one runner execution is
  `#SpecifiedLLMs * #Population * #Sections`; `--max-concurrency` caps the real
  number of concurrent provider calls.
- Propagate CLI `--verbose` flag to logging and to runner diagnostics so
  question-level events surface when requested.
- Gracefully handle `KeyboardInterrupt`: preserve any completed JSONL records and
  exit with status `130`.

## Output Specification
- Output records use the raw JSONL contract defined in
  `docs/03-runner/design.md`.
- Default behavior writes JSONL records to stdout; when `--output` is supplied,
  write to the provided path and emit a confirmation message to stderr
  summarizing location and key metrics (e.g., questionnaire ID, model count,
  population size, records written).
- CLI summary prints a concise run summary showing records written and any
  runner warnings or errors (e.g., retries, validation failures).
- Respect project logging policy: structured JSON logs at INFO level; human
  summaries restricted to stdout/stderr separation for shell scripting.

## Error Handling
- Input validation errors (unknown questionnaire, malformed option, missing
  config) exit with status `2`.
- Configuration loading errors raise `ConfigurationError` or
  `QuestionnaireConfigError`; report source file, key path, and guidance.
- Runtime errors within the runner return status `1`. CLI must surface a
  concise summary plus pointer to detailed logs or partial outputs.
- On partial success (some models fail), include failure reasons in JSONL
  `errors` fields when record-level output is possible and mark the process
  exit code as non-zero while preserving successful raw records.

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
  - Positive integer validation for `--total-population` and
    `--max-concurrency`.
  - CLI `--total-population` override precedence over
    `metadata.default_population`.
  - Listing commands producing deterministic output.
  - Configuration resolution errors (missing files, unresolved environment
    variables).
- Integration tests (with fixtures and mock providers) verify full execution
  path, JSONL output schema, and exit codes for mixed-success scenarios.
- Tests run via `uv run pytest`, organized under `tests/cli/`.

## Extensibility Guidelines
- Additional commands or subcommands must preserve compatibility with existing
  flags. Prefer additive flags over breaking changes.
- When introducing new outputs (e.g., CSV), gate behind explicit options to
  avoid altering the default JSONL payload.
- Keep CLI logic thin; push new business rules into configuration loaders,
  questionnaire validators, or runner components to maintain modularity.
