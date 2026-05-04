# Evaluator Scripts Design

## Purpose and Scope
Evaluator scripts consume output files emitted by the runner module and produce
human-readable analysis. The basic evaluator focuses on one runner JSONL output
at a time, using the output contract defined in `docs/03-runner/design.md`. It
loads the questionnaire referenced by that file, scores every answer, compares
section scores with the questionnaire's recorded human baselines, and writes PDF
bar graphs under `results/`.

Out of scope: calling LLM providers, mutating questionnaire files, changing
runner output records, writing separate data summaries, collecting new human
baselines, or ranking models across unrelated questionnaires.

## Basic Evaluator

### Command Contract
The basic evaluator should accept exactly one input:

- `<runner-output>`: JSONL file emitted by the runner. Each line is one
  questionnaire administration, as specified in `docs/03-runner/design.md`.

Preferred development invocation:

```bash
uv run python bin/evaluate_basic.py results/risky-choice-framing.jsonl
```

The script should fail before analysis if the runner output file is missing,
empty, or contains records from more than one questionnaire. Keeping one
questionnaire per evaluation keeps section-baseline comparisons unambiguous.

The runner-emitted JSONL file is the only data input. It must contain all runner
records needed for scoring, grouping, and charting.

### Input Resolution
For each JSONL record, the evaluator reads:

- `questionnaire.name`: questionnaire ID used to locate
  `config/questionnaires/<id>.yaml`.
- `questionnaire.path`: optional provenance from the runner output.
- `llm_id`: model identifier used for grouping scores.
- `population_index`: independent administration index.
- `response.sections[].id`: section identifier matching `Section.name`.
- `response.sections[].questions[].id`: question identifier.
- `response.sections[].questions[].response.raw`: raw answer token to validate
  and score.
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
3. Validate the raw answer using the questionnaire scoring rules.
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
The evaluator should write PDF bar graphs under a results subdirectory whose
name matches the input JSONL file name without the `.jsonl` suffix. For example,
`results/risky-choice-framing.jsonl` writes charts under
`results/risky-choice-framing/`.

- `section-scores.pdf`: grouped bars by section. Each group shows the human
  average and one bar per evaluated LLM.
- `section-delta.pdf`: bars showing each model's delta from the human average
  per section. Positive values mean the model scored higher than the human
  baseline; negative values mean lower.

Charts should include:

- questionnaire ID in the title.
- axis labels with score units.
- readable section labels.
- legend entries for every model and the human baseline.
- visible output when only one model or one section is present.

### Error Handling
Raise a clear, specific error when:

- the runner output file cannot be read.
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
runner JSONL outputs under `tests/evaluator/`. Cover:

- loading a runner output file and resolving its questionnaire.
- scoring rating and choice answers.
- grouping section means by `llm_id`.
- comparing section means against `human.average`.
- creating non-empty PDF chart files under `results/<runner-output-stem>/`.
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
