from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml
from pydantic import ValidationError

from .errors import QuestionnaireConfigError
from .models import Question, Questionnaire, QuestionType, ScoringRule, Section
from .schema import (
  QuestionnaireFileSchema,
  QuestionnaireSchema,
  QuestionSchema,
  SectionSchema,
)


def list_questionnaires(config_dir: Path) -> list[str]:
  """Return questionnaire names discovered under config/questionnaires."""
  root = _questionnaire_root(config_dir)
  if not root.exists():
    return []
  names: list[str] = []
  for path in sorted(root.glob("*.yaml")):
    names.append(path.stem)
  return names


def load_questionnaire_file(
  path: Path,
  *,
  enforce_id_match: bool = True,
) -> Questionnaire:
  """Load and validate a questionnaire directly from its YAML file."""
  path = Path(path)
  if not path.is_file():
    raise QuestionnaireConfigError(
      f"Questionnaire file '{path}' not found",
      file_path=str(path),
    )
  if path.suffix != ".yaml":
    raise QuestionnaireConfigError(
      "Questionnaire files must use .yaml extension",
      file_path=str(path),
    )
  raw = _load_yaml(path)
  schema = _parse_schema(raw, path)
  questionnaire = _build_questionnaire(schema.questionnaire, path)
  if enforce_id_match:
    expected_name = path.stem
    if questionnaire.id != expected_name:
      raise QuestionnaireConfigError(
        (
          f"Questionnaire id '{questionnaire.id}' must match file name "
          f"'{expected_name}'"
        ),
        file_path=str(path),
        location="questionnaire.id",
      )
  return questionnaire


def load_questionnaire(name: str, config_dir: Path) -> Questionnaire:
  """Load and validate a questionnaire from YAML."""
  path = _resolve_questionnaire_path(name, config_dir)
  return load_questionnaire_file(path, enforce_id_match=True)


def load_multiple(names: Sequence[str], config_dir: Path) -> list[Questionnaire]:
  """Load multiple questionnaires in one call."""
  return [load_questionnaire(name, config_dir) for name in names]


def _questionnaire_root(config_dir: Path) -> Path:
  return Path(config_dir) / "questionnaires"


def _resolve_questionnaire_path(name: str, config_dir: Path) -> Path:
  if not name or Path(name).name != name:
    raise QuestionnaireConfigError(
      f"Invalid questionnaire name '{name}'",
      file_path=str(config_dir),
    )
  path = _questionnaire_root(config_dir) / f"{name}.yaml"
  if not path.is_file():
    raise QuestionnaireConfigError(
      f"Questionnaire file '{path}' not found",
      file_path=str(path),
    )
  return path


def _load_yaml(path: Path) -> Any:
  try:
    with path.open("r", encoding="utf-8") as handle:
      return yaml.safe_load(handle) or {}
  except yaml.YAMLError as exc:
    mark = getattr(exc, "problem_mark", None)
    line_number = None
    if mark is not None and getattr(mark, "line", None) is not None:
      line_number = mark.line + 1
    raise QuestionnaireConfigError(
      f"Failed to parse questionnaire YAML: {exc}",
      file_path=str(path),
      line_number=line_number,
    ) from exc


def _parse_schema(raw: Any, path: Path) -> QuestionnaireFileSchema:
  try:
    return QuestionnaireFileSchema.model_validate(raw)
  except ValidationError as exc:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"])
    message = first["msg"]
    raise QuestionnaireConfigError(
      f"Schema validation failed: {message}",
      file_path=str(path),
      location=location or None,
    ) from exc


def _build_questionnaire(
  schema: QuestionnaireSchema,
  path: Path,
) -> Questionnaire:
  metadata = dict(schema.metadata) if schema.metadata else {}
  sections = _build_sections(schema.sections, path)
  return Questionnaire(
    id=schema.id,
    name=schema.name,
    description=schema.description,
    version=schema.version,
    metadata=metadata,
    sections=sections,
  )


def _build_sections(sections: list[SectionSchema], path: Path) -> list[Section]:
  seen_names: set[str] = set()
  built_sections: list[Section] = []
  for index, section_schema in enumerate(sections):
    location = f"questionnaire.sections[{index}]"
    if section_schema.name in seen_names:
      raise QuestionnaireConfigError(
        f"Duplicate section name '{section_schema.name}'",
        file_path=str(path),
        location=f"{location}.name",
      )
    seen_names.add(section_schema.name)
    questions = _build_questions(section_schema.questions, path, location)
    built_sections.append(
      Section(
        name=section_schema.name,
        instructions=section_schema.instructions,
        questions=questions,
      )
    )
  return built_sections


def _build_questions(
  questions: list[QuestionSchema],
  path: Path,
  section_location: str,
) -> list[Question]:
  seen_ids: set[str] = set()
  built_questions: list[Question] = []
  for index, question_schema in enumerate(questions):
    location = f"{section_location}.questions[{index}]"
    if question_schema.id in seen_ids:
      raise QuestionnaireConfigError(
        f"Duplicate question id '{question_schema.id}'",
        file_path=str(path),
        location=f"{location}.id",
      )
    seen_ids.add(question_schema.id)
    question_type = _coerce_question_type(question_schema.type, path, location)
    options = _validate_options(question_type, question_schema, path, location)
    scoring = _build_scoring(
      question_type,
      question_schema,
      path,
      f"{location}.scoring",
    )
    built_questions.append(
      Question(
        id=question_schema.id,
        type=question_type,
        prompt=question_schema.prompt,
        options=options,
        scoring=scoring,
      )
    )
  return built_questions


def _coerce_question_type(
  raw_type: str,
  path: Path,
  location: str,
) -> QuestionType:
  try:
    return QuestionType(raw_type)
  except ValueError as exc:
    allowed = ", ".join(item.value for item in QuestionType)
    raise QuestionnaireConfigError(
      f"Unsupported question type '{raw_type}'. Allowed types: {allowed}",
      file_path=str(path),
      location=f"{location}.type",
    ) from exc


def _validate_options(
  question_type: QuestionType,
  question_schema: QuestionSchema,
  path: Path,
  location: str,
) -> dict[str, str] | None:
  options = question_schema.options
  if question_type is QuestionType.CHOICE:
    if not options:
      raise QuestionnaireConfigError(
        "Choice questions require 'options'",
        file_path=str(path),
        location=f"{location}.options",
      )
    return dict(options)
  if options:
    raise QuestionnaireConfigError(
      "Non-choice questions may not define 'options'",
      file_path=str(path),
      location=f"{location}.options",
    )
  return None


def _build_scoring(
  question_type: QuestionType,
  question_schema: QuestionSchema,
  path: Path,
  location: str,
) -> ScoringRule:
  total = question_schema.scoring.total
  if total <= 0:
    raise QuestionnaireConfigError(
      "Scoring total must be positive",
      file_path=str(path),
      location=f"{location}.total",
    )

  weights = question_schema.scoring.weights
  normalized_weights: dict[str, int]
  if question_type is QuestionType.CHOICE:
    normalized_weights = _build_choice_weights(
      question_schema, path, location
    )
  else:
    normalized_weights = _build_rating_weights(
      question_type, weights, path, location
    )

  max_weight = max(normalized_weights.values(), default=0)
  if max_weight > total:
    raise QuestionnaireConfigError(
      f"Scoring weights exceed total {total}",
      file_path=str(path),
      location=f"{location}.weights",
    )
  for token, weight in normalized_weights.items():
    if weight < 0:
      raise QuestionnaireConfigError(
        f"Negative weight {weight} for answer '{token}'",
        file_path=str(path),
        location=f"{location}.weights",
      )
  return ScoringRule(total=total, weights=normalized_weights)


def _build_choice_weights(
  question_schema: QuestionSchema,
  path: Path,
  location: str,
) -> dict[str, int]:
  weights = question_schema.scoring.weights
  if not isinstance(weights, dict):
    raise QuestionnaireConfigError(
      "Choice question weights must be a mapping",
      file_path=str(path),
      location=f"{location}.weights",
    )
  option_keys = set((question_schema.options or {}).keys())
  weight_keys = set(weights.keys())
  if option_keys != weight_keys:
    raise QuestionnaireConfigError(
      "Choice weights must match declared option keys",
      file_path=str(path),
      location=f"{location}.weights",
    )
  return dict(weights)


def _build_rating_weights(
  question_type: QuestionType,
  weights: list[int] | dict[str, int],
  path: Path,
  location: str,
) -> dict[str, int]:
  if not isinstance(weights, list):
    raise QuestionnaireConfigError(
      "Rating question weights must be a list",
      file_path=str(path),
      location=f"{location}.weights",
    )
  scale = question_type.rating_scale
  if scale is None:
    raise QuestionnaireConfigError(
      "Rating question type missing scale metadata",
      file_path=str(path),
      location=f"{location}.weights",
    )
  if len(weights) != scale:
    raise QuestionnaireConfigError(
      f"Expected {scale} weights for {question_type.value}",
      file_path=str(path),
      location=f"{location}.weights",
    )
  normalized = {str(index + 1): value for index, value in enumerate(weights)}
  return normalized
