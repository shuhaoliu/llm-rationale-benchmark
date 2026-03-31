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
    You are a neutral assistant administering the burnout inventory.
  metadata:                     # optional, arbitrary string key/value pairs
    author: "Psych Lab"
    published: "2024-06-01"

  sections:
    - name: "Workload"
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
- `instructions`: optional text surfaced in the UI/CLI before the section.
- `questions`: non-empty list of question declarations.

#### System Prompt Requirements
- `system_prompt` must be a non-empty string.
- The prompt seeds every LLM conversation initiated for the questionnaire.
- Multi-line prompts are allowed using YAML block scalars.

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
   - Optional but defined fields (like `metadata`) must adhere to documented
     shapes.
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
class Section:
  name: str
  instructions: str | None
  questions: list[Question]

@dataclass
class Questionnaire:
  id: str
  name: str
  description: str | None
  version: int | None
  metadata: dict[str, str]
  system_prompt: str
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

The runner exposes two parameters that control this mode:

| Parameter           | Type | Description                                                                   |
|---------------------|------|-------------------------------------------------------------------------------|
| `total_population`  | int  | Total number of independent LLM completions to collect for the questionnaire. |
| `parallel_sessions` | int  | Maximum number of completions to run concurrently; governs throughput and rate-limit headroom. |

**Execution model** — the runner schedules `total_population` sessions, keeping
at most `parallel_sessions` in-flight simultaneously. Each session is fully
independent: a fresh conversation context, the same `system_prompt`, and the
same sequence of questions. No state is shared between sessions.

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
  demographics, or IRB references without schema changes.

### Testing Considerations
- Unit tests cover: schema validation, per-type answer validation, scoring, and
  loader error reporting.
- Fixtures mirror the example YAML to ensure docs and implementation stay in
  sync.
- Round-trip tests ensure loaded questionnaires can serialize back to YAML
  without data loss (excluding comment formatting).
- Population dispatch tests mock the LLM client and assert that exactly
  `total_population` completions are requested with at most `parallel_sessions`
  in-flight at any point, verifying both the count and the concurrency ceiling.

### Future Enhancements
- Add conditional logic (skip patterns) between questions while keeping YAML
  declarative.
- Support multi-select (`choice-multi`) and open-text prompts with rubric-based
  scoring.
- Expose schema metadata via JSON Schema for IDE validation and documentation.
