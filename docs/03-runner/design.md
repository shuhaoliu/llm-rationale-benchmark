# Runner Module Design

## Purpose and Scope
- Coordinate questionnaire execution against one or more LLMs and capture raw
  responses for downstream processing.
- Administer one loaded and validated questionnaire for the effective
  population size requested by the CLI. A single runner execution accepts
  exactly one questionnaire and one or more CLI-specified LLM IDs.
- Write all raw LLM responses to one JSONL output file. Each line represents one
  complete questionnaire administration for one CLI-specified LLM ID and one
  population member.
- Validate response shape and question coverage before writing records.

Out of scope: configuration parsing (handled by the configuration layer),
questionnaire loading/modeling (owned by the questionnaire module),
provider-specific transport logic (owned by the LLM connector layer), scoring,
aggregation, analysis, and report generation.

## Inputs and Dependencies
- `Questionnaire` objects already loaded and validated by
  `questionnaire.loader`.
- Runner configuration supplied by the CLI, including concurrency limits, retry
  overrides, selected LLM IDs, and the output JSONL path.
- Effective population size resolved by the CLI from `--total-population` when
  supplied, otherwise from each questionnaire's
  `metadata.default_population`.
- `LLMConversationFactory` initialized by the CLI with resolved provider
  configuration.
- Structured logging helpers shared across the project.

The runner module never mutates questionnaire or connector configuration. It
does not read scoring utilities and does not perform analysis over raw results.
Its only interpretation of answers is validation that the returned raw response
can be mapped back to the questionnaire sections and questions by ID.

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
6. When all sections in an administration complete, the executor validates the
   assembled response object against the questionnaire structure.
7. The runner appends one raw JSON record to the single JSONL output file.

## Module Structure
- `runner/__init__.py`: Exposes public runner APIs such as `BenchmarkRunner`
  and raw response result types.
- `runner/executor.py`: Builds execution plans, schedules work, queries LLMs,
  validates raw response shape, and appends JSONL records.
- `runner/prompts.py` (planned): Helper utilities for constructing section and
  question prompts.
- `runner/types.py`: Data classes describing execution inputs, raw response
  records, question traces, and runner errors.

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
- Validate that the output path can be opened for append/write before issuing
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
  - Preserve the raw provider response for that question without scoring or
    analysis.
  - Add the question and answer to the section conversation history before the
    next question starts.
- If retries are exhausted, record a structured error for that question and
  continue other independent work unless the runner configuration requests
  abort-on-error.

### Raw Response Assembly
- Build one response object per questionnaire administration.
- The response object mirrors the questionnaire structure by ID:
  - Each section entry includes the section ID. In the current questionnaire
    model, `Section.name` is the canonical section identifier.
  - Each question entry includes the question ID.
  - Each question entry stores the raw `LLMResponse` payload or a raw answer
    value plus provider metadata exposed by the connector.
- Validate that every emitted section/question reference exists in the loaded
  questionnaire.
- Validate that required section/question IDs are present or that missing
  answers have explicit error entries.
- Do not compute scores, section totals, averages, summaries, or cross-model
  comparisons.

## JSONL Output Contract
The runner writes a single JSONL file for the whole run. Each line is one JSON
object:

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
            "response": {
              "raw": "4",
              "metadata": {
                "provider": "openai",
                "model": "gpt-4"
              }
            }
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
- `response`: raw response object matching the questionnaire section/question
  structure. Section and question IDs must be included so downstream components
  can resolve answers without relying on ordering alone.
- `errors`: structured runner errors for missing, failed, or invalid raw
  answers. Empty when validation succeeds without recoverable errors.

The writer must append records atomically with respect to concurrent tasks, for
example by serializing writes through a single async writer task or a file lock.

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
- Include recoverable errors in the JSONL record. Fatal planning or write
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
  output_path: Path

@dataclass
class QuestionRunTrace:
  section_id: str
  question_id: str
  response: LLMResponse | None
  attempts: int
  errors: list[RunnerError]

@dataclass
class RawResponseRecord:
  questionnaire_name: str
  questionnaire_path: Path | None
  llm_id: str
  population_index: int
  query_time: datetime
  response: dict[str, Any]
  errors: list[RunnerError]
```

These types describe runner-owned artifacts only. Scored result types belong to
downstream analysis components.

## Observability and Telemetry
- Emit structured logs per question attempt with fields:
  `questionnaire_id`, `llm_id`, `population_index`, `section_id`,
  `question_id`, `attempt`, `latency_ms`, and `retry_reason`.
- Record timing metadata for diagnostics, but keep JSONL output focused on raw
  responses and errors.
- Expose progress callbacks for the CLI, such as administration start,
  question complete, administration complete, and record written.
- Redact sensitive configuration fields before logging or writing provider
  metadata.

## Extensibility
- New question types only require prompt construction and raw response
  validation support in the questionnaire-facing helpers used by the runner.
- Scoring and analytics can evolve independently by consuming the stable JSONL
  output contract.
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
- JSONL writer tests verifying one complete record per administration and
  atomic writes under concurrent completion.
- Validation tests verifying emitted section/question IDs must exist in the
  loaded questionnaire and that missing answers become explicit errors.
- Integration tests with fixture questionnaires and mock providers to assert
  raw JSONL output shape.

## Future Work
- Implement a resumable execution mode that persists completed JSONL records and
  skips administrations already captured.
- Support streaming responses with incremental validation once provider
  adapters expose streaming hooks.
- Integrate optional cost tracking in downstream analysis by consuming token
  usage metadata from raw provider responses.
