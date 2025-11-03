# Project Architecture Guidelines

## Overview
The rationale benchmark coordinates questionnaire-driven evaluations across
multiple LLM providers. The system is intentionally modular so questionnaire
authors, connector engineers, and CLI users can iterate independently while
sharing consistent validation, scoring, and result formats. Architecture
decisions prioritize:
- Configuration-driven behavior (YAML + environment variables) over code edits.
- Deterministic validation and scoring before any network traffic.
- Clear seams for new LLM providers and questionnaire formats.

Refer to `docs/01-llm/design.md` and `docs/02-questionnaire/design.md` for
deep-dive guidance on the connector and questionnaire subsystems that underpin
this document.

## Execution Flow
1. **CLI startup (`cli.py`)** discovers configuration directories, resolves the
   requested questionnaire IDs and LLM configuration name, and validates command
   arguments.
2. **Configuration loading** merges `config/llms/default-llms.yaml` with any
   overrides, resolves `${ENV_VAR}` references, and validates every declared
   `provider/model` selector.
3. **Questionnaire loading** parses YAML files from
   `config/questionnaires/`, runs schema + semantic validation, and constructs
   domain models (`Questionnaire`, `Section`, `Question`, `ScoringRule`).
4. **Conversation assembly** uses `LLMConversationFactory` to create
   `LLMConversation` instances for each requested `provider/model`, wiring the
   appropriate provider adapter and retry policy.
5. **Benchmark execution** iterates loaded questionnaires, prompts each model,
   applies answer validators, and aggregates scores and free-form responses.
6. **Result emission** writes JSON reports, logs structured telemetry, and
   surfaces CLI summaries.

## Core Components

### CLI & Orchestration (`cli.py`, `runner/executor.py`)
- Click-based interface exposes commands for listing questionnaires, previewing
  available LLM configurations, and running benchmarks.
- Accepts comma-delimited questionnaire IDs (`--questionnaires`) and an LLM
  configuration name (`--llm-config`).
- Delegates to configuration loaders and questionnaire loaders, then instantiates
  the benchmark runner with validated inputs.
- Handles exit codes and graceful interruption (Ctrl+C) to ensure partially
  collected results are persisted.

### Configuration Layer (`llm/config_loader.py`, `config/llms/`)
- Consumes YAML files documented in `docs/01-llm/configs.md`.
- Always loads `default-llms.yaml` first; additional files overlay provider and
  default definitions following the merge rules described in the LLM configs
  spec (scalar replacement, dictionary merge, list replacement for `models`).
- Resolves environment variables at load time, raising `ConfigurationError` for
  unresolved secrets or unsupported provider keys.
- Produces a validated mapping of `LLMConnectorConfig` objects keyed by
  `provider/model` so downstream consumers avoid string parsing.

### LLM Connector Layer (`llm/conversation.py`, `llm/factory.py`, `llm/providers/`)
- Governed by `docs/01-llm/design.md`.
- `LLMConversationFactory` instantiates `LLMConversation` objects from validated
  config entries, applying CLI-supplied system prompts or falling back to
  configuration defaults.
- `LLMConversation` maintains transcript history, retry orchestration, optional
  response validators, and archive semantics to prevent reuse after finalization.
- Provider adapters implement a `BaseProviderClient` contract for OpenAI,
  OpenAI-compatible, Anthropic, Gemini, and future providers. Adapters own
  payload construction, structured output hints, and response parsing.
- Connection reuse and optional client caching minimize redundant HTTP session
  creation during benchmark runs.

### Questionnaire Module (`questionnaire/loader.py`, `questionnaire/validator.py`)
- Detailed in `docs/02-questionnaire/design.md`.
- Discovers YAML questionnaire files, preventing directory traversal and
  enforcing `.yaml` suffixes.
- Validation pipeline performs schema checks, semantic validation (unique IDs,
  scoring constraints, type-specific rules), and surfaces precise error
  locations via `QuestionnaireConfigError`.
- Domain models normalize rating weight lists into dictionary form to unify
  scoring logic across question types.
- Exposes helpers for listing questionnaires, loading one or many, and running
  standalone validation via `bin/validate_questionnaire.py`.

### Runner Module (`runner/`)
- Operates on a validated `Questionnaire` and a `LLMConversationFactory`
  instance, deferring provider-specific setup to the connector layer.
- Splits execution into two phases:
  - `runner/executor.py` queries all configured LLMs concurrently. Within each
    conversation it advances through questionnaire sections question-by-question,
    builds prompts from question metadata, and calls
    `LLMConversation.ask()` with retry/backoff semantics. After the final
    question, it archives the transcript for evaluation.
  - `runner/evaluator.py` consumes archived conversations, applies question-type
    validators, computes `QuestionScore` aggregates, and assembles the final
    benchmark report.
- Ensures per-model isolation so network failures or validator issues for one
  provider do not block other conversations from completing.
- Captures timing metadata and structured responses needed downstream for JSON
  reports, CLI summaries, and additional analytics.

### Support Utilities
- `bin/validate_questionnaire.py`: CLI helper validating questionnaire files,
  shared with automation pipelines.
- Shared logging utilities configure structured JSON logs (via `structlog`)
  across CLI, loaders, and benchmark execution.

## Data Flow & Outputs
- **Transcripts**: `LLMConversation.archive()` produces immutable snapshots used
  for post-hoc analysis and debugging.
- **Benchmark JSON Report** (written to disk/stdout as configured):
  ```json
  {
    "benchmark_info": {
      "questionnaires": ["string"],
      "llm_config": "string",
      "execution_timestamp": "ISO8601",
      "models_tested": ["provider/model"]
    },
    "results": [
      {
        "questionnaire": "string",
        "model": "provider/model",
        "question_id": "string",
        "response": "string",
        "reasoning": "string",
        "awarded_score": "float",
        "total_score": "float",
        "latency_ms": "int"
      }
    ],
    "summary": {
      "questionnaires_run": "int",
      "total_questions": "int",
      "models_tested": "int",
      "average_scores_by_questionnaire": {},
      "average_scores_by_model": {},
      "cost_estimates": {}
    }
  }
  ```
- **Logging**: Structured JSON logs capture provider, model, latency, retry
  attempts, and validation outcomes at INFO level. DEBUG traces optionally
  include sanitized request/response snippets.

## Error Handling Strategy
### Configuration Errors
- Invalid YAML, unresolved environment variables, unsupported providers, or
  schema violations raise `ConfigurationError` before runtime.
- Questionnaire validation failures raise `QuestionnaireConfigError` with file
  path and contextual pointer.

### Provider & Network Errors
- Provider adapters categorize errors (authentication, rate limit, network,
  model errors) and feed them into retry policies defined by configuration.
- Exponential backoff with jitter prevents synchronized retries; max attempts
  are configurable per provider or via defaults.
- Exhausted retries surface user-friendly CLI errors while allowing other
  models to continue.

### Benchmark Execution Errors
- Conversation-level failures archive partial transcripts for inspection.
- Runner aggregates encountered errors and reports them at the end of
  execution, while still streaming partial results where safe.

## Observability & Performance
- Rate limiting handled via semaphores or token buckets per provider to respect
  RPM/TPM quotas.
- Optional metrics hooks capture latency distributions, validator failure rates,
  and retry counts (see `docs/01-llm/design.md`).
- Large result sets stream to disk to control memory usage; prompts and
  responses are not eagerly duplicated.
- Async provider adapters share HTTP sessions and enforce `max_concurrent`
  limits derived from configuration.

## Extensibility
- **LLM Providers**: Add new adapters under `llm/providers/` by implementing the
  base client contract; register provider keys in the config loader.
- **Question Types**: Extend `QuestionType` registry with validation/scoring
  hooks and document in `docs/02-questionnaire/design.md`.
- **Scoring**: Introduce new aggregation strategies by extending the evaluator
  while preserving `QuestionScore` inputs.
- **Surface Areas**: The CLI keeps business logic in loaders/runners so additional
  interfaces (REST API, web UI) can reuse the same orchestration pipeline.

## Reference Documents
- `docs/01-llm/design.md`: Connector architecture, conversation lifecycle,
  provider adapter contracts.
- `docs/01-llm/configs.md`: Detailed LLM configuration schema, merge rules, and
  validation expectations.
- `docs/02-questionnaire/design.md`: Questionnaire schema, validation pipeline,
  scoring model, and extensibility hooks.
- `docs/02-questionnaire/guide.md`: Author-facing instructions for crafting
  questionnaires that pass validation and integrate with the benchmark.
