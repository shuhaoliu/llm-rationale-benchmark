"""Basic evaluator for runner output directories."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rationale_benchmark.questionnaire import (
  AnswerValidationError,
  Question,
  Questionnaire,
  QuestionnaireConfigError,
  QuestionType,
  score_question,
  validate_answer,
)
from rationale_benchmark.questionnaire.loader import load_questionnaire

DEFAULT_CONFIG_DIR = Path("config")


class EvaluatorError(RuntimeError):
  """Raised when evaluator input or processing fails."""


@dataclass(frozen=True)
class SectionMean:
  """Mean awarded section score for one model and section."""

  llm_id: str
  section_name: str
  mean: float
  record_count: int


@dataclass(frozen=True)
class HumanComparison:
  """Comparison between a model section mean and a human baseline."""

  llm_id: str
  section_name: str
  llm_mean: float
  record_count: int
  human_average: float
  human_population: int
  delta: float
  absolute_delta: float


@dataclass(frozen=True)
class QuestionResponseAnalysis:
  """Aggregated response stats for one canonical question option."""

  option: str
  count: int
  percentage: float
  delta: float | None


@dataclass(frozen=True)
class QuestionAnalysis:
  """Aggregated response stats for one questionnaire question."""

  questionnaire_id: str
  section_name: str
  question_id: str
  question_prompt: str
  population: int
  responses: list[QuestionResponseAnalysis]


@dataclass(frozen=True)
class EvaluationResult:
  """Result produced by the basic evaluator."""

  questionnaire_id: str
  output_dir: Path
  question_analysis_json: Path
  section_scores_pdf: Path
  section_delta_pdf: Path
  question_analyses: list[QuestionAnalysis]
  model_section_means: dict[tuple[str, str], SectionMean]
  human_comparisons: dict[tuple[str, str], HumanComparison]


@dataclass(frozen=True)
class _ScoredQuestion:
  question_id: str
  question_prompt: str
  answer_token: str
  awarded: int
  total: int


@dataclass(frozen=True)
class _RecordSectionScore:
  llm_id: str
  population_index: int
  section_name: str
  awarded: int
  total: int
  questions: list[_ScoredQuestion]


def evaluate_basic(
  runner_output: Path | str,
  *,
  config_dir: Path | str = DEFAULT_CONFIG_DIR,
) -> EvaluationResult:
  """Evaluate one runner output directory and write PDF section charts.

  Args:
    runner_output: Directory emitted by the benchmark runner.
    config_dir: Project configuration directory containing ``questionnaires/``.

  Returns:
    Aggregate section scores and chart output paths.

  Raises:
    EvaluatorError: If input cannot be read, scored, or charted.
  """

  output_dir = Path(runner_output)
  response_path = _resolve_response_path(output_dir)
  records = _load_jsonl(response_path)
  questionnaire_name = _single_questionnaire_name(records)
  questionnaire = _load_questionnaire(questionnaire_name, Path(config_dir))
  section_scores = _score_records(records, questionnaire)
  if not section_scores:
    raise EvaluatorError("No scored answers remain after excluding runner errors")

  model_section_means = _aggregate_section_means(section_scores)
  human_comparisons = _compare_human_baselines(
    model_section_means,
    questionnaire,
  )
  question_analyses = _aggregate_question_analyses(section_scores, questionnaire)

  question_analysis_json = output_dir / "question-analysis.json"
  section_scores_pdf = output_dir / "section-scores.pdf"
  section_delta_pdf = output_dir / "section-delta.pdf"
  _write_question_analysis_json(question_analysis_json, question_analyses)
  _write_section_scores_chart(
    section_scores_pdf,
    questionnaire,
    model_section_means,
  )
  _write_section_delta_chart(
    section_delta_pdf,
    questionnaire,
    human_comparisons,
  )
  return EvaluationResult(
    questionnaire_id=questionnaire.id,
    output_dir=output_dir,
    question_analysis_json=question_analysis_json,
    section_scores_pdf=section_scores_pdf,
    section_delta_pdf=section_delta_pdf,
    question_analyses=question_analyses,
    model_section_means=model_section_means,
    human_comparisons=human_comparisons,
  )


def _resolve_response_path(output_dir: Path) -> Path:
  if not output_dir.is_dir():
    raise EvaluatorError(f"Runner output directory '{output_dir}' cannot be read")
  return output_dir / "responses.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    raise EvaluatorError(f"Runner responses file '{path}' cannot be read")
  try:
    lines = path.read_text(encoding="utf-8").splitlines()
  except OSError as exc:
    raise EvaluatorError(f"Runner responses file '{path}' cannot be read") from exc
  if not any(line.strip() for line in lines):
    raise EvaluatorError(f"Runner responses file '{path}' is empty")

  records: list[dict[str, Any]] = []
  for line_number, line in enumerate(lines, start=1):
    if not line.strip():
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError as exc:
      raise EvaluatorError(f"Malformed JSONL line {line_number}: {exc}") from exc
    if not isinstance(value, dict):
      raise EvaluatorError(f"Malformed JSONL line {line_number}: expected object")
    records.append(value)
  if not records:
    raise EvaluatorError(f"Runner responses file '{path}' is empty")
  return records


def _single_questionnaire_name(records: list[dict[str, Any]]) -> str:
  names: set[str] = set()
  for index, record in enumerate(records, start=1):
    questionnaire = record.get("questionnaire")
    if not isinstance(questionnaire, dict):
      raise EvaluatorError(f"Record {index} is missing questionnaire metadata")
    name = questionnaire.get("name")
    if not isinstance(name, str) or not name:
      raise EvaluatorError(f"Record {index} is missing questionnaire.name")
    names.add(name)
  if len(names) > 1:
    joined = ", ".join(sorted(names))
    raise EvaluatorError(f"Runner output contains multiple questionnaires: {joined}")
  return next(iter(names))


def _load_questionnaire(name: str, config_dir: Path) -> Questionnaire:
  try:
    return load_questionnaire(name, config_dir)
  except QuestionnaireConfigError as exc:
    raise EvaluatorError(f"Unable to load questionnaire '{name}': {exc}") from exc


def _score_records(
  records: list[dict[str, Any]],
  questionnaire: Questionnaire,
) -> list[_RecordSectionScore]:
  sections_by_name = {section.name: section for section in questionnaire.sections}
  questions_by_section = {
    section.name: {question.id: question for question in section.questions}
    for section in questionnaire.sections
  }
  scored_sections: list[_RecordSectionScore] = []

  for record_index, record in enumerate(records, start=1):
    llm_id = _required_str(record, "llm_id", record_index)
    population_index = _required_int(record, "population_index", record_index)
    errored_questions = _record_errored_questions(record)
    response = record.get("response")
    if not isinstance(response, dict):
      raise EvaluatorError(f"Record {record_index} is missing response object")
    sections = response.get("sections")
    if not isinstance(sections, list):
      raise EvaluatorError(f"Record {record_index} is missing response.sections")

    for section_payload in sections:
      if not isinstance(section_payload, dict):
        raise EvaluatorError(f"Record {record_index} contains malformed section")
      section_name = section_payload.get("id")
      if not isinstance(section_name, str):
        raise EvaluatorError(f"Record {record_index} section is missing id")
      if section_name not in sections_by_name:
        raise EvaluatorError(f"Unknown section '{section_name}' in record {record_index}")

      questions = section_payload.get("questions")
      if not isinstance(questions, list):
        raise EvaluatorError(
          f"Record {record_index} section '{section_name}' is missing questions"
        )
      question_scores = []
      question_map = questions_by_section[section_name]
      for question_payload in questions:
        question_score = _score_question_payload(
          record_index=record_index,
          section_name=section_name,
          question_payload=question_payload,
          question_map=question_map,
          errored_questions=errored_questions,
        )
        if question_score is not None:
          question_scores.append(question_score)
      if question_scores:
        scored_sections.append(
          _RecordSectionScore(
            llm_id=llm_id,
            population_index=population_index,
            section_name=section_name,
            awarded=sum(score.awarded for score in question_scores),
            total=sum(score.total for score in question_scores),
            questions=question_scores,
          )
        )
  return scored_sections


def _score_question_payload(
  *,
  record_index: int,
  section_name: str,
  question_payload: Any,
  question_map: dict[str, Any],
  errored_questions: set[tuple[str, str]],
) -> _ScoredQuestion | None:
  if not isinstance(question_payload, dict):
    raise EvaluatorError(f"Record {record_index} contains malformed question")
  question_id = question_payload.get("id")
  if not isinstance(question_id, str):
    raise EvaluatorError(f"Record {record_index} question is missing id")
  if question_id not in question_map:
    raise EvaluatorError(
      f"Unknown question '{question_id}' in section '{section_name}'"
    )
  if (section_name, question_id) in errored_questions or question_payload.get("errors"):
    return None

  response = question_payload.get("response")
  if isinstance(response, dict):
    answer = _extract_answer(response)
  elif isinstance(response, str):
    answer = response
  else:
    raise EvaluatorError(
      f"Record {record_index} question '{question_id}' has no scorable response"
    )
  question = question_map[question_id]
  if question.type is not QuestionType.CHOICE and isinstance(answer, str):
    stripped = answer.strip()
    if stripped.isdigit():
      answer = int(stripped)
  try:
    answer = _canonicalize_choice_answer(question, answer)
    token = validate_answer(question, answer)
    score = score_question(question, token)
    return _ScoredQuestion(
      question_id=score.question_id,
      question_prompt=question.prompt,
      answer_token=token,
      awarded=score.awarded,
      total=score.total,
    )
  except (AnswerValidationError, QuestionnaireConfigError) as exc:
    raise EvaluatorError(
      f"Answer for question '{question_id}' cannot be validated: {exc}"
    ) from exc


def _required_str(record: dict[str, Any], key: str, record_index: int) -> str:
  value = record.get(key)
  if not isinstance(value, str) or not value:
    raise EvaluatorError(f"Record {record_index} is missing {key}")
  return value


def _required_int(record: dict[str, Any], key: str, record_index: int) -> int:
  value = record.get(key)
  if not isinstance(value, int) or isinstance(value, bool):
    raise EvaluatorError(f"Record {record_index} is missing {key}")
  return value


def _record_errored_questions(record: dict[str, Any]) -> set[tuple[str, str]]:
  errored: set[tuple[str, str]] = set()
  errors = record.get("errors", [])
  if not isinstance(errors, list):
    return errored
  for error in errors:
    if not isinstance(error, dict):
      continue
    section_id = error.get("section_id")
    question_id = error.get("question_id")
    if isinstance(section_id, str) and isinstance(question_id, str):
      errored.add((section_id, question_id))
  return errored


def _extract_answer(response: dict[str, Any]) -> Any:
  parsed = response.get("parsed")
  if isinstance(parsed, dict) and "answer" in parsed:
    return parsed["answer"]
  if parsed is not None and not isinstance(parsed, dict):
    return parsed

  raw = response.get("raw")
  if isinstance(raw, str):
    try:
      decoded = json.loads(raw)
    except json.JSONDecodeError:
      return raw
    if isinstance(decoded, dict) and "answer" in decoded:
      return decoded["answer"]
    return decoded
  return raw


def _canonicalize_choice_answer(question: Question, answer: Any) -> Any:
  if question.type is not QuestionType.CHOICE or not isinstance(answer, str):
    return answer
  if question.options is None or answer in question.options:
    return answer

  normalized_answer = _normalize_choice_text(answer)
  for key, label in question.options.items():
    if normalized_answer == _normalize_choice_text(label):
      return key
  label_matches = [
    key
    for key, label in question.options.items()
    if _contains_choice_phrase(normalized_answer, label)
  ]
  if len(label_matches) == 1:
    return label_matches[0]

  key_matches = [
    key
    for key in question.options
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(key)}(?![A-Za-z0-9])", answer)
  ]
  if len(key_matches) == 1:
    return key_matches[0]
  return answer


def _normalize_choice_text(value: str) -> str:
  lowered = value.strip().lower()
  return re.sub(r"\s+", " ", lowered.strip(" \t\r\n.:;,-_!()[]{}"))


def _contains_choice_phrase(normalized_answer: str, label: str) -> bool:
  normalized_label = _normalize_choice_text(label)
  if not normalized_label:
    return False
  pattern = rf"(?<![a-z0-9]){re.escape(normalized_label)}(?![a-z0-9])"
  return re.search(pattern, normalized_answer) is not None


def _aggregate_section_means(
  section_scores: list[_RecordSectionScore],
) -> dict[tuple[str, str], SectionMean]:
  grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
  for section_score in section_scores:
    grouped[(section_score.llm_id, section_score.section_name)].append(
      section_score.awarded
    )

  means: dict[tuple[str, str], SectionMean] = {}
  for (llm_id, section_name), values in grouped.items():
    means[(llm_id, section_name)] = SectionMean(
      llm_id=llm_id,
      section_name=section_name,
      mean=sum(values) / len(values),
      record_count=len(values),
    )
  return means


def _compare_human_baselines(
  means: dict[tuple[str, str], SectionMean],
  questionnaire: Questionnaire,
) -> dict[tuple[str, str], HumanComparison]:
  human_by_section = {
    section.name: section.human
    for section in questionnaire.sections
    if section.human is not None
  }
  comparisons = {}
  for key, section_mean in means.items():
    human = human_by_section.get(section_mean.section_name)
    if human is None:
      continue
    delta = section_mean.mean - human.average
    comparisons[key] = HumanComparison(
      llm_id=section_mean.llm_id,
      section_name=section_mean.section_name,
      llm_mean=section_mean.mean,
      record_count=section_mean.record_count,
      human_average=human.average,
      human_population=human.population,
      delta=delta,
      absolute_delta=abs(delta),
    )
  return comparisons


def _aggregate_question_analyses(
  section_scores: list[_RecordSectionScore],
  questionnaire: Questionnaire,
) -> list[QuestionAnalysis]:
  counts_by_question: dict[tuple[str, str], dict[str, int]] = defaultdict(
    lambda: defaultdict(int)
  )
  populations_by_question: dict[tuple[str, str], int] = defaultdict(int)

  for section_score in section_scores:
    for question_score in section_score.questions:
      key = (section_score.section_name, question_score.question_id)
      counts_by_question[key][question_score.answer_token] += 1
      populations_by_question[key] += 1

  analyses: list[QuestionAnalysis] = []
  for section in questionnaire.sections:
    for question in section.questions:
      key = (section.name, question.id)
      population = populations_by_question.get(key, 0)
      counts = counts_by_question.get(key, {})
      responses = [
        QuestionResponseAnalysis(
          option=option,
          count=counts.get(option, 0),
          percentage=_response_percentage(counts.get(option, 0), population),
          delta=None,
        )
        for option in _question_options(question)
      ]
      analyses.append(
        QuestionAnalysis(
          questionnaire_id=questionnaire.id,
          section_name=section.name,
          question_id=question.id,
          question_prompt=question.prompt,
          population=population,
          responses=responses,
        )
      )
  return analyses


def _question_options(question: Question) -> list[str]:
  if question.type is QuestionType.CHOICE:
    assert question.options is not None
    return list(question.options)
  assert question.type.rating_scale is not None
  return [str(index) for index in range(1, question.type.rating_scale + 1)]


def _response_percentage(count: int, population: int) -> float:
  if population == 0:
    return 0.0
  return count / population


def _write_question_analysis_json(
  path: Path,
  question_analyses: list[QuestionAnalysis],
) -> None:
  payload = [asdict(question_analysis) for question_analysis in question_analyses]
  path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_section_scores_chart(
  path: Path,
  questionnaire: Questionnaire,
  means: dict[tuple[str, str], SectionMean],
) -> None:
  sections = [section.name for section in questionnaire.sections]
  models = sorted({mean.llm_id for mean in means.values()})
  series: dict[str, list[float | None]] = {}
  human_values = [
    section.human.average if section.human is not None else None
    for section in questionnaire.sections
  ]
  series["Human average"] = human_values
  for model in models:
    series[model] = [
      means[(model, section_name)].mean if (model, section_name) in means else None
      for section_name in sections
    ]
  _write_bar_chart_pdf(
    path,
    title=f"{questionnaire.id} section scores",
    y_label="Awarded points",
    groups=sections,
    series=series,
    x_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    show_values=True,
  )


def _write_section_delta_chart(
  path: Path,
  questionnaire: Questionnaire,
  comparisons: dict[tuple[str, str], HumanComparison],
) -> None:
  sections = [
    section.name for section in questionnaire.sections if section.human is not None
  ]
  models = sorted({comparison.llm_id for comparison in comparisons.values()})
  series = {
    model: [
      comparisons[(model, section_name)].delta
      if (model, section_name) in comparisons
      else None
      for section_name in sections
    ]
    for model in models
  }
  _write_bar_chart_pdf(
    path,
    title=f"{questionnaire.id} section delta from human average",
    y_label="LLM mean minus human average",
    groups=sections,
    series=series,
    show_values=True,
    signed_values=True,
  )


def _write_bar_chart_pdf(
  path: Path,
  *,
  title: str,
  y_label: str,
  groups: list[str],
  series: dict[str, list[float | None]],
  x_ticks: list[float] | None = None,
  show_values: bool = False,
  signed_values: bool = False,
) -> None:
  width = 1100
  left = 360
  right = 60
  top = 82
  series_items = list(series.items())
  series_count = max(1, len(series_items))
  group_count = max(1, len(groups))
  legend_row_height = 18
  bottom = 76 + series_count * legend_row_height
  group_height = max(28, series_count * 9 + 12)
  height = max(595, top + bottom + group_count * group_height)
  plot_width = width - left - right
  values = [value for values in series.values() for value in values if value is not None]
  if not values:
    values = [0.0]
  tick_values = x_ticks or []
  x_min = min(0.0, min(values), *(tick_values or [0.0]))
  x_max = max(0.0, max(values), *(tick_values or [0.0]))
  if math.isclose(x_min, x_max):
    x_min -= 1.0
    x_max += 1.0
  padding = (x_max - x_min) * 0.08
  x_min -= padding
  x_max += padding

  commands = [
    f"1 1 1 rg 0 0 {width} {height} re f",
    f"0 0 0 rg BT /F1 16 Tf 36 {height - 36} Td ({_pdf_text(title)}) Tj ET",
    "0 0 0 rg /F1 10 Tf",
    f"BT /F1 10 Tf {left} {height - 58} Td ({_pdf_text(y_label)}) Tj ET",
    "0.82 0.82 0.82 RG 0.5 w",
    f"{left} {bottom} m {width - right} {bottom} l S",
    f"{left} {bottom} m {left} {height - top} l S",
  ]
  zero_x = _scale_x(0.0, x_min, x_max, left, plot_width)
  commands.append("0.45 0.45 0.45 RG 0.8 w")
  commands.append(f"{zero_x:.2f} {bottom} m {zero_x:.2f} {height - top} l S")
  for tick_value in tick_values:
    tick_x = _scale_x(tick_value, x_min, x_max, left, plot_width)
    commands.append("0.72 0.72 0.72 RG 0.4 w")
    commands.append(f"{tick_x:.2f} {bottom} m {tick_x:.2f} {height - top} l S")
    commands.append("0 0 0 rg")
    commands.append(
      f"BT /F1 8 Tf {tick_x - 7:.2f} {bottom - 16:.2f} Td "
      f"({_pdf_text(_format_tick(tick_value))}) Tj ET"
    )

  colors = [
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
  ]
  bar_height = min(8.0, max(4.0, (group_height - 8) / series_count))

  for group_index, group in enumerate(groups):
    group_top = height - top - group_index * group_height
    group_bottom = group_top - group_height
    label_y = group_bottom + (group_height / 2) - 3
    commands.append("0 0 0 rg")
    commands.append(
      f"BT /F1 8 Tf 36 {label_y:.2f} Td ({_pdf_text(group)}) Tj ET"
    )

    bars_height = bar_height * series_count
    start_y = group_bottom + (group_height - bars_height) / 2
    for series_index, (_label, values_for_series) in enumerate(series_items):
      if group_index >= len(values_for_series):
        continue
      value = values_for_series[group_index]
      if value is None:
        continue
      color = colors[series_index % len(colors)]
      value_x = _scale_x(value, x_min, x_max, left, plot_width)
      x = min(zero_x, value_x)
      y = start_y + series_index * bar_height
      bar_width = max(1.0, abs(value_x - zero_x))
      commands.append(f"{color[0]} {color[1]} {color[2]} rg")
      commands.append(f"{x:.2f} {y:.2f} {bar_width:.2f} {bar_height * 0.82:.2f} re f")
      if show_values:
        value_label = _format_value(value, signed=signed_values)
        value_label_x = value_x + 4 if value >= 0 else value_x - 34
        commands.append("0 0 0 rg")
        commands.append(
          f"BT /F1 7 Tf {value_label_x:.2f} {y - 0.2:.2f} Td "
          f"({_pdf_text(value_label)}) Tj ET"
        )

  legend_x = left
  for series_index, (label, _values_for_series) in enumerate(series_items):
    color = colors[series_index % len(colors)]
    y = 30 + series_index * legend_row_height
    commands.append(f"{color[0]} {color[1]} {color[2]} rg")
    commands.append(f"{legend_x:.2f} {y:.2f} 10 10 re f")
    commands.append("0 0 0 rg")
    commands.append(
      f"BT /F1 9 Tf {legend_x + 16:.2f} {y + 2:.2f} Td "
      f"({_pdf_text(label)}) Tj ET"
    )

  _write_pdf(path, "\n".join(commands), width=width, height=height)


def _scale_x(
  value: float,
  x_min: float,
  x_max: float,
  left: float,
  plot_width: float,
) -> float:
  return left + ((value - x_min) / (x_max - x_min)) * plot_width


def _format_tick(value: float) -> str:
  if math.isclose(value, round(value)):
    return str(int(round(value)))
  return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_value(value: float, *, signed: bool) -> str:
  if signed:
    return f"{value:+.2f}"
  return f"{value:.2f}"


def _pdf_text(value: str) -> str:
  ascii_value = value.encode("latin-1", errors="replace").decode("latin-1")
  return ascii_value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf(path: Path, content: str, *, width: int, height: int) -> None:
  stream = content.encode("latin-1", errors="replace")
  objects = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    (
      f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
      "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    ).encode("ascii"),
    b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    (
      b"<< /Length "
      + str(len(stream)).encode("ascii")
      + b" >>\nstream\n"
      + stream
      + b"\nendstream"
    ),
  ]
  output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
  offsets = [0]
  for index, obj in enumerate(objects, start=1):
    offsets.append(len(output))
    output.extend(f"{index} 0 obj\n".encode("ascii"))
    output.extend(obj)
    output.extend(b"\nendobj\n")
  xref_offset = len(output)
  output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
  output.extend(b"0000000000 65535 f \n")
  for offset in offsets[1:]:
    output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
  output.extend(
    (
      f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
      f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
  )
  path.write_bytes(output)
