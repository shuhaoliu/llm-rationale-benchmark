# Evaluator Question Analysis JSON Design

## Goal
Extend the basic evaluator so `bin/evaluate_basic.py` still writes the two PDF
charts and also writes `question-analysis.json` into the same runner output
directory.

## Constraints
- Reuse the existing evaluator scoring pass instead of creating a second
  analysis pipeline.
- Keep the output deterministic by following questionnaire section/question
  order.
- Do not require runner output changes.
- Stay backward-compatible with existing questionnaires.

## Output Shape
`question-analysis.json` is a JSON array. Each element represents one
questionnaire question and includes:

- `questionnaire_id`
- `section_name`
- `question_id`
- `question_prompt`
- `population`: count of scorable responses collected for the question
- `responses`: array of
  `{"option": ..., "count": ..., "percentage": ..., "delta": ...}`

`responses` should list the canonical answer options for the question:

- rating questions: stringified integers from the rating scale
- choice questions: declared option keys

`percentage` is `count / population`. When `population` is zero, emit `0.0`.

## Human Baseline Assumption
Current questionnaires only define section-level human baselines. They do not
define per-question response distributions, so the evaluator cannot derive a
correct per-option human baseline from existing data alone.

For this change:

- `delta` is included for every response entry.
- `delta` is `null` when no per-question human response baseline exists.

This keeps the JSON schema stable and avoids inventing unsupported values.

## Implementation Outline
1. Extend evaluator result types with a question-analysis collection and output
   path.
2. Reuse the existing scored question records to aggregate counts per canonical
   answer token.
3. Write `question-analysis.json` beside the existing PDFs.
4. Update `bin/evaluate_basic.py` to print the JSON path.
5. Update evaluator docs to document the third artifact and the `delta: null`
   behavior.

## Verification
- Add a failing evaluator test that asserts the JSON file is written with the
  expected schema and percentages.
- Keep the existing PDF tests passing.
- Run `uv run pytest tests/evaluator/test_basic.py`.
