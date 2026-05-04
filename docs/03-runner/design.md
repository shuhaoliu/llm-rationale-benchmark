# Runner Module Design

## Purpose and Scope
- Coordinate questionnaire execution against one or more LLMs and capture raw
  responses for downstream processing.
- Administer one loaded and validated questionnaire for the effective
  population size requested by the CLI. A single runner execution accepts
  exactly one questionnaire and one or more CLI-specified LLM IDs.
- Write questionnaire answers and provider metadata to two JSONL files:
  `responses.jsonl` and `metadata.jsonl`. Each line in both files represents
  one complete questionnaire administration for one CLI-specified LLM ID and
  one population member.
- Canonicalize answers to the format required by the questionnaire before
  writing response records.
- Validate response shape, question coverage, and answer format before writing
  records. Invalid answers are retried before becoming structured errors.

Out of scope: configuration parsing (handled by the configuration layer),
questionnaire loading/modeling (owned by the questionnaire module),
provider-specific transport logic (owned by the LLM connector layer), scoring,
aggregation, analysis, and report generation.

## Inputs and Dependencies
- `Questionnaire` objects already loaded and validated by
  `questionnaire.loader`.
- Runner configuration supplied by the CLI, including concurrency limits, retry
  overrides, selected LLM IDs, and the output directory or explicit output
  paths.
- Effective population size resolved by the CLI from `--total-population` when
  supplied, otherwise from each questionnaire's
  `metadata.default_population`.
- `LLMConversationFactory` initialized by the CLI with resolved provider
  configuration.
- Structured logging helpers shared across the project.

The runner module never mutates questionnaire or connector configuration. It
does not read scoring utilities and does not perform analysis over raw results.
Its only interpretation of answers is canonicalization and validation that the
returned response conforms to the questionnaire's declared answer requirements
and can be mapped back to the questionnaire sections and questions by ID.

## High-Level Execution Flow
1. The CLI loads and validates one questionnaire for this runner execution,
   resolves runner configuration, resolves the effective population size, and
   instantiates `BenchmarkRunner` from `runner/executor.py`.
2. `BenchmarkRunner` creates execution plans for each CLI-specified LLM ID and
   population member.
3. The executor runs independent questionnaire administrations concurrently,
   subject to runner and provider concurrency limits.
4. Within one questionnaire administration, sections may run concurrently
   because their conversations are independent.
5. Within one section, questions are always queried sequentially. Later
   questions include conversation history from earlier questions in the same
   section.
6. For each question, the executor canonicalizes the model answer to the token
   required by the questionnaire. If canonicalization or validation fails, the
   question is retried within the configured retry budget.
7. When all sections in an administration complete, the executor validates the
   assembled response object against the questionnaire structure.
8. The runner appends aligned JSON records to `responses.jsonl` and
   `metadata.jsonl`.

## Module Structure
- `runner/__init__.py`: Exposes public runner APIs such as `BenchmarkRunner`
  and runner result types.
- `runner/executor.py`: Builds execution plans, schedules work, queries LLMs,
  canonicalizes and validates answers, and appends aligned JSONL records.
- `runner/prompts.py` (planned): Helper utilities for constructing section and
  question prompts.
- `runner/types.py`: Data classes describing execution inputs, canonical
  response records, metadata records, question traces, and runner errors.

The runner module does not include `runner/evaluator.py`. Scoring,
aggregation, and analytics are deferred to separate components that consume the
JSONL output.

## Executor Responsibilities

### Execution Planning
- Validate that every requested LLM ID from the CLI is available through the
  conversation factory configuration. Preserve the exact CLI-specified LLM ID
  in output records.
- Produce immutable execution plans for the questionnaire, each LLM ID, and
  population index.
- Carry the effective population size as an input from the CLI instead of
  resolving it internally.
- Resolve the output destination before issuing provider requests. By default,
  create a new run directory named
  `<questionnaire-id>-<llm-profile>-<timestamp>` under `results/`.
- Validate that both output files can be opened for append/write before issuing
  provider requests.

### Concurrency Model
- Use asynchronous execution with a configurable semaphore limiting
  simultaneous provider calls. The semaphore size comes from the CLI
  `--max-concurrency` argument, which defaults to `5`.
- Run parallel and independent administrations over the population of the same
  LLM. For example, an effective population size of `20` creates 20
  independent questionnaire administrations for each selected LLM.
- Run administrations for different LLM IDs independently so one LLM failure
  does not cancel unrelated LLMs.
- Run different sections within the same questionnaire administration in
  parallel when capacity allows.
- The theoretical maximum parallel provider-call frontier is:
  `#SpecifiedLLMs * #Population * #Sections`. The real number of concurrent
  provider calls is capped by `--max-concurrency`.
- Preserve strict sequential execution for questions within the same section.
- Use `asyncio.gather(return_exceptions=True)` or equivalent structured task
  collection so individual failures become structured runner errors.

### Question Loop
- For each section:
  - Start a fresh section-scoped `LLMConversation` using the questionnaire
    system prompt and that section's instructions.
  - Do not include question-answer pairs from any other section.
- For each question in section order:
  - Construct a prompt containing questionnaire metadata needed for context,
    the section instructions, the current question ID and prompt, and the
    prior in-section conversation history.
  - Call `LLMConversation.ask()` with runner-managed retry/backoff overrides.
  - Canonicalize the model answer against the current question's declared
    answer requirements.
  - Preserve the raw provider response and provider metadata in the metadata
    trace for that question.
  - Add the question and canonical answer to the section conversation history
    before the next question starts.
- If canonicalization or validation fails, retry the same question. The failed
  raw answer and validation message are retained in the metadata trace.
- If retries are exhausted, record a structured error for that question and
  continue other independent work unless the runner configuration requests
  abort-on-error.

### Answer Canonicalization
- Canonicalization is performed after each provider response and before the
  answer is accepted into the response record or section conversation history.
- Rating questions must produce the canonical numeric string accepted by the
  questionnaire validator, such as `"1"` through `"5"` for `rating-5`.
- Choice questions must produce one declared option key. The canonicalizer may
  accept exact option keys, exact option labels, and simple unambiguous forms
  such as `"Option A"` when `A` is a declared key.
- Ambiguous, missing, malformed, or out-of-range answers fail validation and
  trigger a retry.
- The runner does not score answers during canonicalization. It only converts
  accepted answers to questionnaire-valid tokens and records validation errors
  when conversion fails.

### Response Assembly
- Build one response object per questionnaire administration.
- The response object mirrors the questionnaire structure by ID:
  - Each section entry includes the section ID. In the current questionnaire
    model, `Section.name` is the canonical section identifier.
  - Each question entry includes the question ID.
  - Each question entry stores the canonical answer token accepted by the
    questionnaire validator.
- Validate that every emitted section/question reference exists in the loaded
  questionnaire.
- Validate that required section/question IDs are present or that missing
  answers have explicit error entries.
- Do not compute scores, section totals, averages, summaries, or cross-model
  comparisons.

## JSONL Output Contract
The runner writes two JSONL files for each run:

- `responses.jsonl`: canonical questionnaire answers, suitable for scoring.
- `metadata.jsonl`: raw provider responses, provider metadata, attempts, and
  diagnostics needed to audit or reproduce the response record.

By default, the CLI creates a new run directory under `results/` named
`<questionnaire-id>-<llm-profile>-<timestamp>` and writes both files there.
The LLM profile is a filesystem-safe slug derived from the CLI `--llm-config`
value. The timestamp must be filesystem-safe and precise enough to avoid
collisions for normal repeated runs, for example `2026-05-04T16-30-00Z`.

The files are line-aligned: line `N` in `responses.jsonl` and line `N` in
`metadata.jsonl` describe the same questionnaire administration. Each line is
one JSON object with the same run identity fields.

### `responses.jsonl`
Each response line contains canonical answers only:

```json
{
  "questionnaire": {
    "name": "burnout-survey",
    "path": "config/questionnaires/burnout-survey.yaml"
  },
  "llm_id": "gpt-4",
  "population_index": 0,
  "query_time": "2026-04-27T10:30:00Z",
  "response": {
    "sections": [
      {
        "id": "emotional_exhaustion",
        "questions": [
          {
            "id": "ee_1",
            "response": "4"
          }
        ]
      }
    ]
  },
  "errors": []
}
```

Field requirements:
- `questionnaire.name`: questionnaire ID or name from the loaded questionnaire.
- `questionnaire.path`: source path when available from the loader.
- `llm_id`: exact LLM ID string specified in the CLI.
- `population_index`: zero-based index of the independent questionnaire
  administration for that LLM.
- `query_time`: timestamp of the first provider query for this JSONL record.
- `response`: canonical response object matching the questionnaire
  section/question structure. Section and question IDs must be included so
  downstream components can resolve answers without relying on ordering alone.
- `errors`: structured runner errors for missing, failed, or invalid answers.
  Empty when validation succeeds without recoverable errors.

### `metadata.jsonl`
Each metadata line contains the raw provider outputs and diagnostics for the
same administration:

```json
{
  "questionnaire": {
    "name": "burnout-survey",
    "path": "config/questionnaires/burnout-survey.yaml"
  },
  "llm_id": "gpt-4",
  "population_index": 0,
  "query_time": "2026-04-27T10:30:00Z",
  "metadata": {
    "sections": [
      {
        "id": "emotional_exhaustion",
        "questions": [
          {
            "id": "ee_1",
            "attempts": [
              {
                "attempt": 1,
                "raw_response": "Option 4",
                "canonical_response": "4",
                "validation_error": null,
                "provider_metadata": {
                  "provider": "openai",
                  "model": "gpt-4"
                }
              }
            ]
          }
        ]
      }
    ]
  },
  "errors": []
}
```

Field requirements:
- The run identity fields must match the corresponding response line exactly:
  `questionnaire`, `llm_id`, `population_index`, and `query_time`.
- `metadata`: metadata object matching the questionnaire section/question
  structure. Section and question IDs must be included so downstream components
  can resolve attempts without relying on ordering alone.
- `attempts`: ordered provider attempts for the question. Failed validation
  attempts must include the raw answer and `validation_error`; successful
  attempts must include the accepted `canonical_response`.
- `provider_metadata`: provider and connector metadata exposed by
  `LLMResponse.metadata`, with sensitive fields redacted.
- `errors`: structured runner errors for missing, failed, or invalid answers.
  Empty when validation succeeds without recoverable errors.

The writer must append aligned record pairs atomically with respect to
concurrent tasks, for example by serializing writes through a single async
writer task or a file lock. A response line must not be written without the
matching metadata line.

## Error Handling
- Catch configuration, conversation lifecycle, validation, timeout, and
  provider-specific exceptions.
- Normalize errors into `RunnerError` objects containing:
  - `llm_id`: exact CLI-specified LLM ID.
  - `questionnaire_id`: loaded questionnaire ID or name.
  - `population_index`: independent administration index.
  - `section_id` and `question_id` when the error is question-specific.
  - `stage`: `"planning"`, `"prompt"`, `"validation"`, `"network"`,
    `"write"`, or `"runtime"`.
  - `message`, `details`, and `retry_count`.
- Continue execution for unaffected LLMs, population members, and sections.
- Include recoverable errors in both JSONL records. Fatal planning or write
  errors should fail the run before issuing more provider requests.

## Data Contracts

### runner/types.py (proposed)
```python
@dataclass(frozen=True)
class RunnerExecutionPlan:
  questionnaire: Questionnaire
  questionnaire_path: Path | None
  llm_id: str
  population_index: int
  output_dir: Path

@dataclass
class QuestionRunTrace:
  section_id: str
  question_id: str
  canonical_response: str | None
  attempts: list[QuestionAttemptTrace]
  errors: list[RunnerError]

@dataclass
class QuestionAttemptTrace:
  attempt: int
  raw_response: LLMResponse | None
  canonical_response: str | None
  validation_error: str | None
  provider_metadata: dict[str, Any]

@dataclass
class ResponseRecord:
  questionnaire_name: str
  questionnaire_path: Path | None
  llm_id: str
  population_index: int
  query_time: datetime
  response: dict[str, Any]
  errors: list[RunnerError]

@dataclass
class MetadataRecord:
  questionnaire_name: str
  questionnaire_path: Path | None
  llm_id: str
  population_index: int
  query_time: datetime
  metadata: dict[str, Any]
  errors: list[RunnerError]

@dataclass
class OutputRecordPair:
  response_record: ResponseRecord
  metadata_record: MetadataRecord
```

These types describe runner-owned artifacts only. Scored result types belong to
downstream analysis components.

## Observability and Telemetry
- Emit structured logs per question attempt with fields:
  `questionnaire_id`, `llm_id`, `population_index`, `section_id`,
  `question_id`, `attempt`, `latency_ms`, and `retry_reason`.
- Record timing metadata for diagnostics in `metadata.jsonl`, but keep
  `responses.jsonl` focused on canonical answers and recoverable errors.
- Expose progress callbacks for the CLI, such as administration start,
  question complete, administration complete, and record written.
- Redact sensitive configuration fields before logging or writing provider
  metadata.

## Extensibility
- New question types only require prompt construction, canonicalization, and
  validation support in the questionnaire-facing helpers used by the runner.
- Scoring and analytics can evolve independently by consuming the stable
  `responses.jsonl` contract and, when needed, the aligned `metadata.jsonl`
  audit trail.
- Additional output formats can be implemented as downstream converters from
  JSONL rather than new runner responsibilities.
- Alternative scheduling strategies can replace the default async executor if
  they preserve the same concurrency and ordering guarantees.

## Testing Strategy
- Unit tests for execution planning using stubbed questionnaires and LLM IDs.
- Unit tests for population scheduling verifying independent administrations
  are created for the effective population size.
- Concurrency tests verifying population members and sections can run
  concurrently while questions within a section remain sequential.
- Conversation-context tests verifying later questions receive only prior
  in-section history.
- JSONL writer tests verifying one complete aligned record pair per
  administration and atomic writes under concurrent completion.
- Validation tests verifying emitted section/question IDs must exist in the
  loaded questionnaire and that missing answers become explicit errors.
- Canonicalization tests verifying rating answers, choice keys, choice labels,
  and simple forms such as `"Option A"` become questionnaire-valid tokens, and
  invalid answers are retried.
- Integration tests with fixture questionnaires and mock providers to assert
  split JSONL output shape.

## Future Work
- Implement a resumable execution mode that persists completed JSONL records and
  skips administrations already captured.
- Support streaming responses with incremental validation once provider
  adapters expose streaming hooks.
- Integrate optional cost tracking in downstream analysis by consuming token
  usage metadata from raw provider responses.
