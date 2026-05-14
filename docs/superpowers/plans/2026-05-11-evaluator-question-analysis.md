# Evaluator Question Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `question-analysis.json` to the basic evaluator outputs and report
it from the CLI.

**Architecture:** Keep one evaluator scoring pass. Aggregate question-level
counts from the scored records already produced for section means, then write a
stable JSON artifact ordered by questionnaire structure.

**Tech Stack:** Python, pytest, JSON, existing evaluator/questionnaire models

---

### Task 1: Document the evaluator output contract

**Files:**
- Modify: `docs/04-evaluator/design.md`

- [ ] **Step 1: Update the design doc**

Add the new `question-analysis.json` artifact, its schema, and the `delta:
null` behavior when per-question human baselines are unavailable.

- [ ] **Step 2: Verify the doc reads consistently**

Check that the command contract, output descriptions, and testing expectations
all mention the new JSON artifact.

### Task 2: Add the failing evaluator test

**Files:**
- Modify: `tests/evaluator/test_basic.py`
- Test: `tests/evaluator/test_basic.py`

- [ ] **Step 1: Write a new failing test**

Assert that `evaluate_basic(...)` writes `question-analysis.json`, returns the
new output path, and serializes stable per-question response counts and
percentages.

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `uv run pytest tests/evaluator/test_basic.py -k question_analysis -v`

Expected: FAIL because the evaluator result has no JSON path and no JSON file
is written.

### Task 3: Implement the evaluator JSON aggregation

**Files:**
- Modify: `rationale_benchmark/evaluator/basic.py`
- Modify: `rationale_benchmark/evaluator/__init__.py`

- [ ] **Step 1: Add result models for question analysis**

Introduce frozen dataclasses for per-response and per-question analysis and add
the JSON output path to `EvaluationResult`.

- [ ] **Step 2: Reuse scored question records for aggregation**

Store canonical answer tokens alongside scored question records, aggregate
counts in questionnaire order, and compute percentages from the scorable
population.

- [ ] **Step 3: Write `question-analysis.json`**

Serialize the analysis list with stable formatting into the runner output
directory. Emit `delta: null` for now because questionnaires do not carry
per-question human response baselines.

- [ ] **Step 4: Run the targeted test to verify it passes**

Run: `uv run pytest tests/evaluator/test_basic.py -k question_analysis -v`

Expected: PASS

### Task 4: Wire the CLI and re-run verification

**Files:**
- Modify: `bin/evaluate_basic.py`
- Test: `tests/evaluator/test_basic.py`

- [ ] **Step 1: Print the JSON output path**

Extend the script output so it reports `question-analysis.json` in addition to
the two PDF files.

- [ ] **Step 2: Run the evaluator test file**

Run: `uv run pytest tests/evaluator/test_basic.py`

Expected: PASS

- [ ] **Step 3: Run the focused CLI/evaluator verification**

Run: `uv run pytest tests/evaluator/test_basic.py tests/cli/test_cli.py -k evaluate_basic`

Expected: PASS or no matching CLI tests if none cover this script.
