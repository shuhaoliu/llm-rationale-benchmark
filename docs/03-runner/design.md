# Runner Module Design

## Purpose and Scope
- Coordinate questionnaire execution against one or more LLM providers using
  standardized conversations and scoring semantics.
- Guarantee per-model isolation so connector, validator, or scoring failures do
  not prevent other models from completing.
- Capture reproducible artifacts (transcripts, scoring breakdowns, telemetry)
  that downstream consumers use for CLI summaries, JSON reports, and analytics.

Out of scope: configuration parsing (handled by the configuration layer),
questionnaire validation/modeling (owned by the questionnaire module), and
provider-specific transport logic (owned by the LLM connector layer).

## Inputs and Dependencies
- `Questionnaire` objects loaded and validated by `questionnaire.loader`.
- `LLMConversationFactory` initialized by the CLI with resolved configuration
  mappings.
- Runner configuration (concurrency limits, retry overrides, output paths)
  supplied through CLI options and environment variables; defaults documented in
  `docs/configs.md`.
- Scoring utilities and data models defined in `docs/interfaces.md`.
- Structured logging helpers shared across the project (JSON output via
  `structlog`).

The runner module never mutates questionnaire or connector configuration. It
derives execution-time metadata (timestamps, latency metrics, attempt counts)
and produces immutable result objects.

## High-Level Execution Flow
1. The CLI aggregates selected questionnaires and a target LLM configuration,
   then instantiates `BenchmarkRunner` (wrapper defined in `runner/executor.py`)
   with a `LLMConversationFactory`, concurrency policy, and output descriptors.
2. `BenchmarkRunner` prepares execution plans for each `provider/model`
   requested, including questionnaire metadata and derived system prompts.
3. Executor launches per-model tasks concurrently. Each task:
   - Creates a fresh `LLMConversation`.
   - Iterates questionnaire sections and questions sequentially.
   - Builds prompts and validator hooks for the current question.
   - Calls `LLMConversation.ask()` with runner-managed retry/backoff overrides.
   - Records `LLMResponse` plus structured metadata for evaluation.
   - Archives the conversation when all questions are answered or a fatal error
     occurs.
4. Completed transcripts feed into `runner/evaluator.py`, which applies
   question-level validators, computes `QuestionScore`/`SectionScore` objects,
   and produces aggregate statistics.
5. Runner collates execution metadata, scores, and raw answers into a benchmark
   report that upstream components serialize to JSON and display in the CLI.
6. Final status includes success/failure per model, encountered errors, and any
   warnings related to validation or scoring anomalies.

## Module Structure
- `runner/__init__.py`: Exposes public runner APIs (e.g., `BenchmarkRunner`,
  `BenchmarkResult`).
- `runner/executor.py`: Orchestrates questionnaire execution, concurrency, and
  transcript collection.
- `runner/evaluator.py`: Performs validation, scoring, aggregation, and summary
  generation over archived conversations.
- `runner/prompts.py` (planned): Helper utilities for constructing question
  prompts, system prompt overrides, and validator bindings.
- `runner/types.py`: Data classes describing execution inputs, intermediate
  artifacts, and emitted results.

## Executor Responsibilities

### Execution Planning
- Validate that every requested `provider/model` exists in the conversation
  factory configuration; raise a descriptive `RunnerConfigError` when a model is
  missing.
- Merge questionnaire `system_prompt` with CLI overrides and per-model metadata.
- Produce an immutable `ModelExecutionPlan` describing questionnaire IDs,
  section ordering, retry policies, and output destinations.

### Concurrency Model
- Default to asynchronous execution using `asyncio`, with a configurable
  semaphore limiting the number of simultaneous provider calls. The semaphore
  defaults to the minimum of the CLI `--max-concurrency` flag and provider
  `max_concurrent` hints derived from configuration metadata.
- Each model plan runs in its own task to preserve isolation. Tasks share a
  `RunnerTracer` for logging but never share mutable transcript state.
- Executor collects task results via `asyncio.gather(return_exceptions=True)` so
  individual failures become structured errors without cancelling other tasks.

### Question Loop
- For each question:
  - Construct a `PromptContext` containing questionnaire metadata, section
    instructions, and question prompt. Reuse templates from `runner/prompts.py`
    to maintain consistent formatting.
  - Resolve validators appropriate for the `QuestionType` using the questionnaire
    module's helper registry.
  - Call `conversation.ask(prompt, validator=..., max_attempts=override)` and
    capture `LLMResponse`. If validator returns `False`, executor triggers
    retry/backoff until max attempts are exhausted.
  - Record per-attempt metadata (latency, retry count, validator failures) in a
    mutable `QuestionRunTrace`. Only the final successful response is forwarded
    to evaluation.
- On exhaustion of retries, mark the question result as failed, capture the last
  error, and continue executing the remaining questions (unless CLI requests
  abort-on-error). Failed questions still produce partial artifacts for post-run
  analysis.

### Conversation Finalization
- After a questionnaire finishes (or aborts), call `conversation.archive()` to
  obtain immutable `ConversationTurn` history. Store alongside per-question
  traces inside a `ModelExecutionResult`.
- Ensure archives cannot be reused by verifying `LLMConversation.is_archived`
  before returning the result. Attempting to reuse triggers logged warnings.

### Error Handling
- Catch `ConfigurationError`, `ConversationArchivedError`, and provider-specific
  exceptions. Normalize them into `RunnerError` objects containing:
  - `model`: provider/model identifier.
  - `stage`: `"prompt"`, `"validation"`, `"network"`, `"scoring"`, etc.
  - `message`, `details`, `retry_count`.
- Continue execution for unaffected models. Summaries include counts of
  warnings vs. fatal errors; CLI uses this to determine exit codes.

## Evaluator Responsibilities

### Inputs
- `ModelExecutionResult` containing archived transcripts, per-question
  responses, and error traces.
- Questionnaire metadata including scoring rules from `Question` and
  `ScoringRule`.

### Processing Steps
1. Normalize responses into a `QuestionAnswer` structure capturing the raw
   answer, reasoning field (if provided), token usage, and latency.
2. Apply question-type validators (rating range checks, option membership). On
   failure, emit `QuestionScore` with `awarded=0`, annotate the failure, and
   continue scoring.
3. Compute per-question scores using questionnaire scoring weights. Combine
   into `SectionScore` and `QuestionnaireScore` aggregates.
4. Generate model-level summaries (average score, completion rate, total cost)
   plus cross-model aggregates required for CLI/JSON reports.
5. Produce a `BenchmarkResult` object containing:
   - `benchmark_info` snapshot (questionnaires executed, models tested,
     timestamps).
  - `model_results`: list of `ModelBenchmarkResult` entries with question-level
    data.
  - `summary`: totals by questionnaire and by model.

### Error and Warning Propagation
- Carry forward executor warnings (e.g., retries, validator failures) into the
  scoring output so the CLI can display them.
- Distinguish between recoverable scoring issues (invalid answer format,
  missing rationale) and unrecoverable evaluation errors (transcript missing,
  scoring rule misconfigured). Unrecoverable errors short-circuit evaluation for
  the affected model while preserving other results.

## Data Contracts

### runner/types.py (proposed)
```python
@dataclass(frozen=True)
class PromptContext:
  questionnaire_id: str
  section_name: str
  question: Question
  system_prompt: str

@dataclass
class QuestionRunTrace:
  question_id: str
  prompt: str
  response: LLMResponse | None
  attempts: int
  latency_ms: int | None
  errors: list[RunnerError]

@dataclass
class ModelExecutionResult:
  model: str
  questionnaire_id: str
  transcript: list[ConversationTurn]
  question_traces: list[QuestionRunTrace]
  started_at: datetime
  completed_at: datetime | None
  errors: list[RunnerError]

@dataclass
class BenchmarkResult:
  benchmark_info: BenchmarkInfo
  model_results: list[ModelBenchmarkResult]
  summary: BenchmarkSummary
```

Concrete summary structures mirror the JSON schema defined in
`docs/architecture.md`. Keeping these as dataclasses simplifies serialization
and unit testing.

## Observability and Telemetry
- Emit structured logs per question attempt with fields:
  `questionnaire_id`, `model`, `question_id`, `attempt`, `latency_ms`,
  `validator_passed`, `retry_reason`.
- Record timing metrics at both question and model granularity. The executor
  exposes hooks (`on_question_start`, `on_question_end`, `on_model_complete`)
  that can be wired to metrics exporters or CLI progress bars.
- Preserve prompt/response text in transcripts while redacting sensitive fields
  (API keys, environment references). Traces persist on disk only when the user
  enables a `--save-transcripts` flag.

## Extensibility
- Adding new question types only requires validator/weight logic updates in the
  questionnaire module; the runner consumes the normalized API.
- Concurrency strategies remain pluggable: executor exposes an interface for
  alternative schedulers (e.g., thread pool for sync providers, job queue for
  distributed execution).
- Additional output formats (CSV, Parquet) can wrap `BenchmarkResult` without
  modifying executor logic.
- Future enhancements may include adaptive retry policies, cost estimation
  plugins, and partial questionnaire execution for skipped sections.

## Testing Strategy
- Unit tests for executor planning and question loops using stubbed
  `LLMConversation` objects that simulate retries, timeouts, and validator
  failures.
- Integration tests covering end-to-end execution against fixture
  questionnaires and mock providers to assert transcript archiving and scoring
  accuracy.
- Property-based tests for evaluator scoring to verify invariants (awarded score
  never exceeds total, section totals equal question sums).
- Regression suites verifying JSON report stability and error propagation to the
  CLI.

## Future Work
- Implement a resumable execution mode that persists progress to disk and
  resumes after interruptions.
- Support streaming responses with incremental validation once provider adapters
  expose streaming hooks.
- Integrate optional cost tracking by consuming token usage metadata from
  `LLMResponse`.
