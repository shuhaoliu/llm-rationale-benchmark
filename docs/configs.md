# Configuration Management Specifications

## Overview
Configuration drives almost every aspect of the rationale benchmark workflow.
Two independent configuration domains exist:
1. **LLM connector configurations** stored in `config/llms/`.
2. **Questionnaire definitions** stored in `config/questionnaires/`.

Both rely on YAML files, environment variable substitution, and deterministic
validation so CLI runs fail fast and reproducibly. Review this document alongside
`docs/01-llm/configs.md`, `docs/01-llm/design.md`, and
`docs/02-questionnaire/design.md` before adding or modifying configuration.

## Directory Layout
- `config/llms/`  
  Houses connector definitions. `default-llms.yaml` **must** exist and acts as
  the base configuration loaded by every run. Additional files (e.g.,
  `research-models.yaml`) overlay the defaults.
- `config/questionnaires/`  
  Contains questionnaire YAML files, one per questionnaire. Filenames should
  mirror the questionnaire ID (`burnout-survey.yaml`) to simplify discovery.

The CLI selects configurations by filename stem:
- `uv run rationale-benchmark --llm-config research-models`
- `uv run rationale-benchmark --questionnaires burnout-survey,cognitive-biases`

## LLM Configuration Schema
Canonical structure (see `docs/01-llm/configs.md` for a complete specification):

```yaml
defaults:
  timeout: 30
  max_retries: 3
  temperature: 0.7
  max_tokens: 1000

providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    models:
      - "gpt-4o"
      - "gpt-3.5-turbo"
    default_params:
      response_format: "json"
    provider_specific:
      organization: "${OPENAI_ORG_ID}"
```

### Defaults Section
- Optional in override files, required in `default-llms.yaml`.
- Recognized fields: `timeout`, `max_retries`, `temperature`, `max_tokens`,
  `system_prompt`, and any additional parameters supported by connectors.
- Defaults merge into every provider unless overridden explicitly.

### Provider Entries
Each provider map **must** include:
- `api_key`: almost always an environment placeholder (`${ENV_VAR}`).
- `models`: list of model identifiers exposed to the CLI (e.g.,
  `openai/gpt-4o` becomes a runnable selector).

Optional fields:
- `endpoint` or `base_url` (mutually exclusive depending on adapter).
- Provider-specific overrides (`timeout`, `max_retries`, `default_params`,
  `provider_specific`, `metadata`).

#### Supported Provider Keys
| Key pattern                         | Notes                                                            |
|-------------------------------------|------------------------------------------------------------------|
| `openai`                            | Official OpenAI endpoints.                                       |
| `{provider}_openai_compatible`      | OpenAI-compatible APIs (e.g., `openrouter_openai_compatible`).   |
| `anthropic`                         | Claude family models.                                            |
| `gemini`                            | Google Gemini models (requires explicit `endpoint`).             |

Unknown provider keys raise `ConfigurationError`.

### Environment Variable Resolution
- Express secrets as `${ENV_VAR}`; the loader performs shell-style substitution.
- Missing variables immediately raise `ConfigurationError` naming the unresolved
  placeholder.
- Never store raw secrets in YAML; if unavoidable, document the rationale and
  restrict file access.

### Merge Rules
1. Load `default-llms.yaml`.
2. Load the requested override and overlay it.

Overlay semantics:
- Scalars replace base values.
- Dictionaries merge recursively with override values winning.
- Lists (notably `models`) replace the base list entirely.
- Providers absent from the override remain untouched.

### Validation
- YAML parsing errors halt execution with `ConfigurationError`.
- Required sections (`providers`) and keys (`api_key`, `models`) must exist.
- Numeric bounds enforced (`0.0 ≤ temperature ≤ 2.0`, non-negative timeouts).
  Set `timeout: 0` to disable client-side request deadlines.
- Streaming-specific parameters (`stream`, `streaming`, `stream_options`) are
  disallowed.
- Provider names must match the supported registry.

## Questionnaire Configuration Schema
Detailed examples appear in `docs/02-questionnaire/design.md` and
`docs/02-questionnaire/guide.md`. Each questionnaire file contains a single
top-level `questionnaire` mapping:

```yaml
questionnaire:
  id: "burnout-survey"
  name: "Burnout Inventory"
  description: "Measuring perceived burnout."
  version: 1
  system_prompt: |
    You are participating in an interview.
    Answer each question as yourself.
    Follow the requested answer format and do not provide explanations.
  metadata:
    default_population: 5
    author: "Psych Lab"
  sections:
    - name: "Workload"
      human:
        average: 18.6
        population: 128
      instructions: "Rate the statements."
      questions:
        - id: "workload_01"
          type: "rating-5"
          prompt: "I feel overwhelmed by tasks."
          scoring:
            total: 5
            weights: [0, 1, 3, 4, 5]
```

### Required Elements
- `id`: slug-style identifier unique across files.
- `name`: human-readable title.
- `sections`: non-empty list; each section contains `questions`.
- `system_prompt`: non-empty string passed to every LLM conversation.
- `metadata.default_population`: positive integer fallback used when the CLI
  does not provide `--total-population`.

### Section Requirements
- `name`: unique section label within the questionnaire.
- `human`: optional known human baseline. `human.average` is the average human
  section score, and `human.population` is the positive number of collected
  human answers.
- `questions`: non-empty list of question declarations.

### Question Requirements
- Supported `type` values: `rating-5`, `rating-7`, `rating-11`, `choice`.
- `options` required for `choice` questions, mapping keys to labels.
- `scoring.total`: positive integer specifying the maximum achievable score.
- `scoring.weights`: list or mapping depending on type (normalized to dict
  during load).

### Validation Pipeline
1. **Schema validation** ensures types and required keys are present.
2. **Semantic validation** enforces unique IDs, weight lengths, option coverage,
   numeric ranges (`0 ≤ weight ≤ total`), positive
   `metadata.default_population`, and positive `human.population` when a human
   baseline is provided.
3. **(Planned)** Cross-file validation guards against duplicate questionnaire
   IDs and version regressions.

Validation errors raise `QuestionnaireConfigError` with file path and contextual
pointer (e.g., `sections[0].questions[1].scoring.weights[3]`).

### Scoring Helpers
- `QuestionType` enum provides validator functions for each supported type.
- Answers normalize to canonical tokens (`"1"`, `"5"`, `option_key`) before
  scoring.
- Scoring utilities return deterministic `QuestionScore` values consumed by the
  benchmark evaluator.

## Environment Variable Management
- Use uppercase snake case (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- Maintain `.env.example` with placeholders for required variables.
- Document newly required variables in `README.md` and relevant docs.

## Validation Tooling
- Run `uv run python bin/validate_questionnaire.py` against individual files or
  directories to reuse the same validation pipeline used at runtime.
- Configuration loaders include unit tests that cover merge behavior,
  environment substitution, and failure modes; add new tests when adjusting
  schemas or validation rules.

## Best Practices
- Keep YAML indentation at 2 spaces; wrap long prompts using block scalars.
- Prefer small, focused override files instead of duplicating the default LLM
  configuration wholesale.
- Version questionnaires whenever question wording, scoring, or metadata
  semantics change.
- Treat `metadata.default_population` as a safe run default only; a valid
  positive CLI `--total-population` value overrides it for that invocation.
- Treat configuration changes as code: accompany them with tests where
  applicable and update documentation references.
