# Evaluator Scripts Design

## Purpose and Scope
Evaluator scripts consume output directories emitted by the runner module and
produce human-readable analysis. The basic evaluator focuses on one runner
output directory at a time, using the output contract defined in
`docs/03-runner/design.md`. It loads the questionnaire referenced by
`responses.jsonl`, scores every answer, compares section scores with the
questionnaire's recorded human baselines, writes PDF bar graphs directly into
the same runner output directory, and emits a machine-readable JSON summary for
per-question analysis.

Out of scope: calling LLM providers, mutating questionnaire files, changing
runner output records, collecting new human baselines, or ranking models across
unrelated questionnaires.

## Basic Evaluator

### Command Contract
The basic evaluator should accept exactly one input:

- `<runner-output-dir>`: directory emitted by the runner. It must contain
  `responses.jsonl`, whose lines are questionnaire administrations as specified
  in `docs/03-runner/design.md`.

Preferred development invocation:

```bash
uv run python bin/evaluate_basic.py results/risky-choice-framing-default-llms-2026-05-04T16-30-00Z
```

The script should fail before analysis if the runner output directory is
missing, if `responses.jsonl` is missing or empty, or if `responses.jsonl`
contains records from more than one questionnaire. Keeping one questionnaire
per evaluation keeps section-baseline comparisons unambiguous.

The runner-emitted `responses.jsonl` file is the only data input. It must
contain all runner records needed for scoring, grouping, and charting.

### JSON Output
The basic evaluator should also write `question-analysis.json` into the runner
output directory. The file contains a large JSON array where each element
describes one questionnaire question:

- `questionnaire_id`
- `section_name`
- `question_id`
- `question_prompt`
- `population`: count of scorable responses collected for the question
- `responses`: ordered list of
  `{"option": ..., "count": ..., "percentage": ..., "delta": ...}`

For each response entry:

- `option` is the canonical answer token.
  - rating questions use stringified scale values such as `"1"` through `"5"`.
  - choice questions use declared option keys such as `"low"` or `"high"`.
- `count` is the number of scorable responses that selected the option.
- `percentage` is `count / population`. If `population` is zero, the evaluator
  should emit `0.0`.
- `delta` is the difference from the human baseline percentage for that option
  when such a per-question baseline exists. Current questionnaires only define
  section-level human baselines, so the evaluator should emit `null` when a
  per-question option baseline is unavailable.

Question entries should follow questionnaire section/question order, and
response entries should follow the canonical option order for the question.

### Input Resolution
For each record in `<runner-output-dir>/responses.jsonl`, the evaluator reads:

- `questionnaire.name`: questionnaire ID used to locate
  `config/questionnaires/<id>.yaml`.
- `questionnaire.path`: optional provenance from the runner output.
- `llm_id`: model identifier used for grouping scores.
- `population_index`: independent administration index.
- `response.sections[].id`: section identifier matching `Section.name`.
- `response.sections[].questions[].id`: question identifier.
- `response.sections[].questions[].response`: canonical answer token to
  validate and score.
- `errors`: recoverable runner errors that should be excluded from scored
  aggregates when they affect an answer.

The evaluator should load questionnaires through the existing questionnaire
loader so schema validation and semantic checks match the rest of the project.
It should use the project's default questionnaire location,
`config/questionnaires/`, and should not expose a configuration-directory option.
The `questionnaire.path` value in the JSONL file is metadata only; it should not
be required for loading.

### Scoring Flow
For each record:

1. Match every output section to a loaded questionnaire section by name.
2. Match every output question to the corresponding questionnaire question by
   ID.
3. Validate the canonical answer using the questionnaire scoring rules.
4. Convert the validated answer into a `QuestionScore`.
5. Sum awarded and total points per section.
6. Store per-question, per-section, per-record, and per-model aggregates.

Section scores should be reported as awarded points. When a normalized view is
useful, compute it explicitly as `awarded / total` and label it separately. Do
not compare normalized LLM scores to raw human averages unless the questionnaire
metadata states that the human baseline is normalized on the same scale.

### Human Baseline Comparison
Human baselines live on questionnaire sections:

```yaml
sections:
  - name: "Risk Seeking"
    human:
      average: 18.6
      population: 128
```

The basic evaluator should compare each scored section against
`section.human.average` when present. If a section has no human baseline, it
should be omitted from human-comparison charts unless the chart clearly marks
the missing baseline.

For each model and section, compute:

- mean LLM section score across all population records.
- number of scored records.
- human average and human population when available.
- delta from human average (`llm_mean - human_average`).
- absolute delta (`abs(llm_mean - human_average)`).

### Visual Outputs
The evaluator should write PDF bar graphs and `question-analysis.json` directly
into the existing runner output directory. For example,
`results/risky-choice-framing-default-llms-2026-05-04T16-30-00Z/` receives the
artifacts alongside `responses.jsonl` and `metadata.jsonl`.

- `section-scores.pdf`: grouped bars by section. Each group shows the human
  average and one bar per evaluated LLM.
- `section-delta.pdf`: bars showing each model's delta from the human average
  per section. Positive values mean the model scored higher than the human
  baseline; negative values mean lower.
- `question-analysis.json`: per-question counts, percentages, and human-baseline
  deltas for each canonical response option.

Charts should include:

- questionnaire ID in the title.
- axis labels with score units.
- readable section labels.
- legend entries for every model and the human baseline.
- visible output when only one model or one section is present.

### Error Handling
Raise a clear, specific error when:

- the runner output directory cannot be read.
- `responses.jsonl` cannot be read.
- a JSONL line is malformed.
- records refer to multiple questionnaires.
- the questionnaire cannot be loaded from the default questionnaire directory.
- a section or question ID from the runner output does not exist in the
  questionnaire.
- an answer cannot be validated against the question type.
- no scored answers remain after excluding errored records.

Recoverable runner errors already live in the JSONL records. A question-level
runner error should exclude only that question from scoring when the rest of the
record is valid.

### Testing Expectations
Unit tests for the basic evaluator should use small fixture questionnaires and
runner output directories under `tests/evaluator/`. Cover:

- loading a runner output directory and resolving its questionnaire.
- scoring rating and choice answers.
- grouping section means by `llm_id`.
- comparing section means against `human.average`.
- writing `question-analysis.json` with stable question order, counts, and
  percentages.
- creating non-empty PDF chart files directly under the runner output
  directory.
- failing on mixed-questionnaire runner outputs.
- failing on unknown section or question IDs.

Run the relevant tests with:

```bash
uv run pytest tests/evaluator/
```

The full project suite should still pass with:

```bash
uv run pytest
```
