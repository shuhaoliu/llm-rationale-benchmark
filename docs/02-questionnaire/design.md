# Questionnaire Module Design

### Purpose and Scope
The questionnaire module ingests YAML questionnaires, validates their structure,
and exposes loaded objects to the benchmark runner. It standardizes how
social-psychology style question sets are represented, scored, and evolved over
time. The module owns:
- File discovery and parsing for `config/questionnaires/*.yaml`
- Schema validation for questionnaires, sections, questions, and scoring specs
- Data model construction (`Questionnaire`, `Section`, `Question`, `Scoring`)
- Scoring helpers that apply answer-validation rules and compute per-question
  points

### YAML Schema Overview
Each questionnaire file contains one top-level `questionnaire` mapping.

```yaml
questionnaire:
  id: "burnout-survey"          # slug-friendly identifier (required)
  name: "Burnout Inventory"     # human readable title (required)
  description: "..."            # optional, shown in CLI help
  version: 1                    # integer incremented on schema changes
  system_prompt: |              # required system prompt passed to LLMs
    You are participating in an interview.
    Answer each question as yourself.
    Follow the requested answer format and do not provide explanations.
  metadata:                     # required metadata and provenance values
    default_population: 5       # fallback if CLI --total-population is absent
    author: "Psych Lab"
    published: "2024-06-01"

  sections:
    - name: "Workload"
      human:
        average: 18.6
        population: 128
      instructions: "Rate how often..."  # optional helper copy
      questions:
        - id: "workload_01"
          type: "rating-5"
          prompt: "I feel overwhelmed by tasks."
          scoring:
            total: 5
            weights: [0, 1, 3, 4, 5]
        - id: "workload_02"
          type: "choice"
          prompt: "Which statement best reflects your current workload?"
          options:
            low: "Manageable"
            medium: "Challenging"
            high: "Overwhelming"
          scoring:
            total: 3
            weights:
              low: 0
              medium: 2
              high: 3
```

#### Section Requirements
- `name`: unique within the questionnaire.
- `human`: optional human baseline with `average` and `population`.
  - `average`: average human score for the section.
  - `population`: positive integer count of collected human answers.
- `instructions`: optional text surfaced in the UI/CLI before the section.
- `questions`: non-empty list of question declarations.

#### System Prompt Requirements
- `system_prompt` must be a non-empty string.
- The prompt seeds every LLM conversation initiated for the questionnaire.
- Multi-line prompts are allowed using YAML block scalars.

#### Metadata Requirements
- `metadata` is required.
- `metadata.default_population` is required and must be a positive integer.
- `default_population` suggests how many independent times the full
  questionnaire is administered to each LLM when the CLI does not provide
  `--total-population`. For example, `default_population: 5` produces five
  complete answer sets per LLM by default.
- A valid positive CLI `--total-population` value overrides
  `metadata.default_population`.
- Additional metadata keys may store provenance, experiment identifiers,
  demographics, or IRB references.

#### Question Requirements
- `id`: unique across whole questionnaire (namespaced by section in practice).
- `type`: supported values `"rating-5"`, `"rating-7"`, `"rating-11"`, `"choice"`.
  Future types must register validation rules and scoring semantics.
- `prompt`: plain-text or markdown question text.
- `options`: required only for `"choice"` questions. Mapping of option keys to
  human readable labels. Keys double as canonical answer IDs.
- `scoring`: mapping containing
  - `total`: integer maximum score attainable for the question.
  - `weights`: scoring definition shaped by question type.
    - Rating questions (`rating-X`) accept a list of length `X` with integer
      weights ordered from rating `1` to rating `X`.
    - Choice questions require a mapping whose keys match the declared `options`.

### Validation Pipeline
Every questionnaire load now triggers an automatic sanity check that runs before
any domain objects are instantiated. The same validation routine powers the
`bin/validate_questionnaire.py` helper so that local authors and automated jobs
share identical guarantees. Failures surface as `QuestionnaireConfigError`
instances that include the originating file path, 1-based line number (when the
YAML parser provides it), and a concise explanation of the defect.

Validation occurs in three stages to keep responsibilities clear and maintain
friendly error messages.

1. **Schema validation** using `pydantic`-style models or custom validators to
   ensure required keys exist and types align (string vs. mapping vs. list).
2. **Semantic validation** to catch domain-specific expectations:
   - Question IDs must be globally unique within a questionnaire.
   - Section names must be distinct.
   - Rating question weight lists must include exactly `X` entries and stay
     within `[0, total]`.
   - Choice question weights must cover exactly the declared options.
   - `total` must be positive and at least the max defined weight.
   - Scoring definitions must match the declared `QuestionType` (e.g., rating
     questions may not supply option maps).
   - `metadata.default_population` must be present and positive.
   - `human.population`, when present, must be positive.
   - CLI `--total-population`, when provided, must be positive.
   - Metadata fields must adhere to documented shapes.
3. **Cross-file validation** (optional future enhancement) to verify that
   questionnaire IDs and versions do not conflict across files.

Validation errors are raised as `QuestionnaireConfigError` with contextual paths
(e.g., `sections[0].questions[1].scoring.weights[3]`) to simplify debugging.

### Data Model
The module produces domain objects mirroring the schema and aligning with
`docs/interfaces.md`.

```python
@dataclass
class ScoringRule:
  total: int
  weights: dict[str, int]

@dataclass
class Question:
  id: str
  type: QuestionType         # Enum wrapper over raw string types
  prompt: str
  options: dict[str, str] | None
  scoring: ScoringRule

@dataclass
class HumanBaseline:
  average: float
  population: int

@dataclass
class Section:
  name: str
  instructions: str | None
  questions: list[Question]
  human: HumanBaseline | None

@dataclass
class Questionnaire:
  id: str
  name: str
  description: str | None
  version: int | None
  metadata: dict[str, str | int]
  system_prompt: str
  default_population: int
  sections: list[Section]

@dataclass
class PopulationResult:
  questionnaire_id: str
  total_population: int
  parallel_sessions: int
  results: list[QuestionnaireResult]   # one entry per completed session
```

An enum-backed `QuestionType` guards supported types and enables richer
behaviour (e.g., default answer validators). Optional fields default to `None`
or `{}` to maintain predictable serialization.

During load, rating question weight lists are normalized into dictionaries whose
keys are stringified integers (`"1"`, `"2"`, …) so downstream scoring logic can
address all question types uniformly.

### Answer Validation
Before scoring, every answer passes a format check derived from `QuestionType`.

| Type        | Validator                                                     |
|-------------|---------------------------------------------------------------|
| `rating-5`  | integer in `[1, 5]`                                           |
| `rating-7`  | integer in `[1, 7]`                                           |
| `rating-11` | integer in `[1, 11]`                                          |
| `choice`    | string key matching one of the declared `options`             |

Validators surface precise error messages describing the expected range or
option set. They return canonical tokens (`"1"`, `"5"`, `option_key`) used as
lookups into `scoring.weights`.

### Scoring Flow
Scoring is handled by pure functions to stay deterministic and testable.

1. Accept a `Question` and a validated answer token.
2. Look up the weight for that token; raise if unmapped (indicates config bug).
3. Return a `QuestionScore` structure:
   ```python
   @dataclass
   class QuestionScore:
     question_id: str
     awarded: int
     total: int
   ```
4. Aggregate question scores per section and questionnaire to produce a
   normalized score (e.g., sum of `awarded` / sum of `total`). Aggregation lives
   in the benchmark runner but relies on the consistent shape from this module.

### Population Dispatch
Some questionnaires are designed to be administered to a large population of
respondents. Rather than collecting a single answer, the goal is to observe the
*distribution* of answers — histograms, mean ± SD, per-question entropy, etc.
— across many independent LLM sessions seeded by the same system prompt.

The questionnaire metadata and runner together control this mode:

| Parameter | Type | Description |
|-----------|------|-------------|
| `default_population` | int | Required YAML fallback answer-set count per LLM. |
| `total_population` | int | Optional CLI override for the answer-set count. |
| `parallel_sessions` | int | Maximum concurrent completions. |

**Population resolution** — the runner uses a valid positive
`--total-population` CLI value when one is provided. Otherwise, it falls back to
the questionnaire's required `metadata.default_population` value.

**Execution model** — the runner schedules the resolved `total_population`
independent administrations per LLM, keeping at most `parallel_sessions`
in-flight simultaneously. Each administration starts from a fresh context and
uses the same `system_prompt`.

The scheduler prioritizes the first wave of work across population instances,
then across questionnaire section rounds, and finally across LLMs. For example,
with 2 population instances, 2 sections, 3 LLMs, and `parallel_sessions` set to
6, the first 6 in-flight requests target every LLM for both sections of
population instance 0 before starting population instance 1.

Sections within a questionnaire are independent. The runner may send different
sections to an LLM concurrently, and section requests must not include
question-answer pairs from other sections in their context.

Questions within the same section are ordered. The runner **MUST** query them in
sequence, carrying the question-answer pairs from preceding questions in that
section as context for later questions.

**Output** — each completed session produces a `QuestionnaireResult`. The runner
collects all results into a `PopulationResult` aggregate (see Data Model) that
can be persisted and analysed to characterise how the LLM's answers are
distributed across the population.

### File Discovery and Loading
- Questionnaires live under `config/questionnaires/`.
- Loader exposes:
  - `list_questionnaires(config_dir: Path) -> list[str]`
  - `load_questionnaire(name: str, config_dir: Path) -> Questionnaire`
  - `load_multiple(names: Sequence[str], config_dir: Path) -> list[Questionnaire]`
- Loader enforces `.yaml` suffix and prevents directory traversal.
- YAML parsing uses `yaml.safe_load` with line-number metadata surfaced in
  validation errors when possible.

### Extensibility Strategy
- New question types register a `QuestionTypeSpec` containing validator and
  scoring hooks.
- Weights schema can expand to support partial credit or sub-scores by adding
  optional fields (e.g., `subscales`). Versioning via `questionnaire.version`
  signals consumers to adjust behaviour.
- Metadata block offers a flexible location for experiment identifiers,
  demographics, or IRB references without schema changes. The reserved
  `default_population` key is required and supplies the fallback repeated
  administration count.

### Testing Considerations
- Unit tests cover: schema validation, per-type answer validation, scoring, and
  loader error reporting.
- Fixtures mirror the example YAML to ensure docs and implementation stay in
  sync.
- Round-trip tests ensure loaded questionnaires can serialize back to YAML
  without data loss (excluding comment formatting).
- Population dispatch tests mock the LLM client and assert that exactly
  `total_population` complete administrations are requested with at most
  `parallel_sessions` in-flight at any point, verifying the count, concurrency
  ceiling, and population-before-section-before-LLM scheduling order.
- Population resolution tests assert that a valid positive CLI
  `--total-population` overrides `metadata.default_population`, and that the
  metadata default is used when the CLI value is absent.
- Section dispatch tests assert that sections may execute concurrently without
  sharing context, while questions within each section are queried sequentially
  with prior in-section question-answer pairs included.

### Future Enhancements
- Add conditional logic (skip patterns) between questions while keeping YAML
  declarative.
- Support multi-select (`choice-multi`) and open-text prompts with rubric-based
  scoring.
- Expose schema metadata via JSON Schema for IDE validation and documentation.
