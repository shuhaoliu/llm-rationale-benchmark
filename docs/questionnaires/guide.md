# Questionnaire Authoring Guide

This guide walks through creating and validating YAML questionnaires consumed by
the rationale benchmark CLI. Follow the steps to produce survey-style
questionnaires that align with social psychology research practices.

## 1. Understand the File Layout
- Store questionnaires in `config/questionnaires/`.
- One YAML file defines one questionnaire.
- File names should match the questionnaire ID for clarity
  (e.g., `burnout-survey.yaml`).

## 2. Questionnaire Skeleton
Create a new file with the top-level `questionnaire` mapping and metadata:

```yaml
questionnaire:
  id: "burnout-survey"
  name: "Burnout Inventory"
  description: "Measuring perceived burnout in knowledge workers."
  version: 1
  metadata:
    author: "Psych Lab"
    published: "2024-06-01"
  sections: []
```

### Required Fields
- `id`: slug-style identifier (lowercase letters, digits, hyphen).
- `name`: human-readable title used in CLI listings.
- `sections`: non-empty list once populated.

### Optional Fields
- `description`: free-form summary.
- `version`: increment when schema changes (e.g., add/remove questions).
- `metadata`: arbitrary key/value strings for provenance and IRB info.

## 3. Add Sections
Sections group related questions and appear sequentially in the CLI UI.

```yaml
  sections:
    - name: "Workload"
      instructions: "Rate how often each statement feels true."
      questions: []
```

- `name`: unique within the questionnaire.
- `instructions`: optional helper text shown before the section.
- `questions`: populate next.

## 4. Add Questions
Each question requires an ID, type, prompt, and scoring definition. Supported
types: `rating-5`, `rating-7`, `rating-11`, `choice`.

### Rating Question Template
```yaml
        - id: "workload_01"
          type: "rating-5"
          prompt: "I feel overwhelmed by the number of tasks on my plate."
          scoring:
            total: 5
            weights: [0, 1, 3, 4, 5]
```

- `id` must be unique across the entire questionnaire.
- Use consistent prefixes (`workload_XX`) to aid tracing.
- `prompt` accepts plain text or simple Markdown.
- `weights` is a list whose length matches the rating scale. Entry index `0`
  corresponds to rating `1`, index `1` to rating `2`, and so on.

### Choice Question Template
```yaml
        - id: "workload_02"
          type: "choice"
          prompt: "Which statement best reflects your current workload?"
          options:
            low: "Manageable with spare capacity"
            medium: "Challenging but sustainable"
            high: "Overwhelming and unsustainable"
          scoring:
            total: 3
            weights:
              low: 0
              medium: 2
              high: 3
```

- Provide `options` as key-to-label mapping.
- Ensure `weights` covers every option key exactly once.

## 5. Scoring Guidelines
- `total`: maximum achievable points for the question (positive integer).
- `weights`: integer scores per answer token.
- For rating questions, provide a list whose length equals the scale size. Keep
  weights monotonic unless the design explicitly requires reverse scoring.
- Reverse scoring is supported by assigning higher weights to lower numeric
  ratings.
- Confirm the highest weight does not exceed `total`; validations enforce this.

## 6. Answer Validation Rules
- `rating-*`: answers must be integers within the defined range. CLI clients
  will reject values outside the range before scoring.
- `choice`: answers must use the canonical option keys (`low`, `medium`, etc.).
- When designing surveys with paper originals, align printed option labels to
  these keys to minimize transcription errors.

## 7. Provide Multiple Sections
Repeat the section pattern for each thematic cluster. Example excerpt:

```yaml
    - name: "Recovery"
      instructions: "Reflect on your ability to recharge between work days."
      questions:
        - id: "recovery_01"
          type: "rating-7"
          prompt: "I can disconnect from work during personal time."
          scoring:
            total: 7
            weights: [0, 1, 2, 4, 5, 6, 7]
```

Ensure section ordering reflects the experience you intend for participants.

## 8. Validate Locally
1. Run the standalone sanity checker:
   `uv run python bin/validate_questionnaire.py config/questionnaires/burnout-survey.yaml`.
   It performs YAML parsing, schema validation, and semantic checks, reporting
   issues with precise line numbers when available.
2. Run `uv run rationale-benchmark --list-questionnaires` to confirm the new
   file is discoverable. Loading within the CLI automatically invokes the same
   sanity checks, so failures will halt the run with actionable error messages.
3. Execute `uv run rationale-benchmark --questionnaire burnout-survey --models gpt-4`
   (or another configured model) to verify parsing, validation, and scoring flow.
4. Add unit tests mirroring the questionnaire to `tests/` when introducing new
   scoring or question types.

## 9. Maintain Documentation
- Update `docs/questionnaires/design.md` when adding new question types or
  scoring behaviours.
- Record any experiment-specific constraints (e.g., required demographics) in
  the questionnaire `metadata`.
- Version bump questionnaires whenever question wording or scoring changes to
  keep result comparisons trustworthy.

## 10. Checklist Before Commit
- [ ] All sections and questions have unique IDs.
- [ ] Every question defines `scoring.total` and `scoring.weights`.
- [ ] Rating weight lists include exactly one entry per rating value.
- [ ] Choice weights cover every option key.
- [ ] `bin/validate_questionnaire.py` passes for the new or modified files.
- [ ] YAML lint passes (`uv run yamllint config/questionnaires/burnout-survey.yaml`).
- [ ] README snippets remain accurate if examples were edited.
