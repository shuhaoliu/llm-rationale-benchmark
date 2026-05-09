from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .errors import QuestionnaireConfigError
from .models import (
  HumanBaseline,
  Question,
  Questionnaire,
  QuestionType,
  ScoringRule,
  Section,
)
from .output_schema import build_full_output_schema
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
  metadata = dict(schema.metadata)
  default_population = _extract_default_population(metadata, path)
  sections = _build_sections(schema.sections, path)
  return Questionnaire(
    id=schema.id,
    name=schema.name,
    description=schema.description,
    version=schema.version,
    metadata=metadata,
    default_population=default_population,
    system_prompt=schema.system_prompt,
    sections=sections,
  )


def _extract_default_population(metadata: Mapping[str, object], path: Path) -> int:
  location = "questionnaire.metadata.default_population"
  value = metadata.get("default_population")
  if not isinstance(value, int) or isinstance(value, bool) or value < 1:
    raise QuestionnaireConfigError(
      "metadata.default_population must be a positive integer",
      file_path=str(path),
      location=location,
    )
  return value


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
    human = _build_human_baseline(section_schema, path, location)
    questions = _build_questions(section_schema.questions, path, location)
    built_sections.append(
      Section(
        name=section_schema.name,
        instructions=section_schema.instructions,
        questions=questions,
        human=human,
      )
    )
  return built_sections


def _build_human_baseline(
  section_schema: SectionSchema,
  path: Path,
  location: str,
) -> HumanBaseline | None:
  if section_schema.human is None:
    return None
  population = section_schema.human.population
  if (
    not isinstance(population, int)
    or isinstance(population, bool)
    or population < 1
  ):
    raise QuestionnaireConfigError(
      "human.population must be a positive integer",
      file_path=str(path),
      location=f"{location}.human.population",
    )
  return HumanBaseline(
    average=section_schema.human.average,
    population=population,
  )


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
    output_schema = _build_output_schema(
      question_schema,
      question_type,
      options,
      path,
      location,
    )
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
        output_schema=output_schema,
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


def _build_output_schema(
  question_schema: QuestionSchema,
  question_type: QuestionType,
  options: dict[str, str] | None,
  path: Path,
  location: str,
) -> dict[str, Any]:
  raw_schema = question_schema.output_schema
  properties = raw_schema.get("properties")
  if not isinstance(properties, Mapping):
    raise QuestionnaireConfigError(
      "output_schema.properties must be a mapping",
      file_path=str(path),
      location=f"{location}.output_schema.properties",
    )

  answer_schema = properties.get("answer")
  if not isinstance(answer_schema, Mapping):
    raise QuestionnaireConfigError(
      "output_schema.properties.answer must be a mapping",
      file_path=str(path),
      location=f"{location}.output_schema.properties.answer",
    )

  required = raw_schema.get("required")
  if not isinstance(required, list) or "answer" not in required:
    raise QuestionnaireConfigError(
      "output_schema.required must include 'answer'",
      file_path=str(path),
      location=f"{location}.output_schema.required",
    )

  _validate_answer_schema(
    question_schema.id,
    question_type,
    answer_schema,
    options,
    path,
    location,
  )

  normalized = deepcopy(raw_schema)
  normalized["properties"] = {
    key: deepcopy(value)
    for key, value in properties.items()
  }
  normalized["required"] = list(required)
  return build_full_output_schema(question_schema.id, normalized)


def _validate_answer_schema(
  question_id: str,
  question_type: QuestionType,
  answer_schema: Mapping[str, Any],
  options: dict[str, str] | None,
  path: Path,
  location: str,
) -> None:
  answer_type = answer_schema.get("type")
  answer_location = f"{location}.output_schema.properties.answer"

  if question_type is QuestionType.CHOICE:
    if answer_type != "string":
      raise QuestionnaireConfigError(
        "Choice question output_schema answer type must be 'string'",
        file_path=str(path),
        location=f"{answer_location}.type",
      )
    enum_values = answer_schema.get("enum")
    if not isinstance(enum_values, list):
      raise QuestionnaireConfigError(
        "Choice question output_schema answer must declare an enum list",
        file_path=str(path),
        location=f"{answer_location}.enum",
      )
    option_keys = list((options or {}).keys())
    if enum_values != option_keys:
      raise QuestionnaireConfigError(
        "Choice question output_schema enum must match option keys exactly",
        file_path=str(path),
        location=f"{answer_location}.enum",
      )
    return

  scale = question_type.rating_scale
  if answer_type != "integer":
    raise QuestionnaireConfigError(
      "Rating question output_schema answer type must be 'integer'",
      file_path=str(path),
      location=f"{answer_location}.type",
    )
  if answer_schema.get("minimum") != 1:
    raise QuestionnaireConfigError(
      "Rating question output_schema minimum must be 1",
      file_path=str(path),
      location=f"{answer_location}.minimum",
    )
  if answer_schema.get("maximum") != scale:
    raise QuestionnaireConfigError(
      f"Rating question output_schema maximum must be {scale}",
      file_path=str(path),
      location=f"{answer_location}.maximum",
    )
