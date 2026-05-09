from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScoringSchema(BaseModel):
  total: int
  weights: list[int] | dict[str, int]

  model_config = ConfigDict(extra="forbid")


class QuestionSchema(BaseModel):
  id: str
  type: str
  prompt: str
  output_schema: dict[str, Any]
  options: dict[str, str] | None = None
  scoring: ScoringSchema

  model_config = ConfigDict(extra="forbid")


class HumanBaselineSchema(BaseModel):
  average: float
  population: int

  model_config = ConfigDict(extra="forbid")


class SectionSchema(BaseModel):
  name: str
  human: HumanBaselineSchema | None = None
  instructions: str | None = None
  questions: list[QuestionSchema] = Field(..., min_length=1)

  model_config = ConfigDict(extra="forbid")


class QuestionnaireSchema(BaseModel):
  id: str
  name: str
  description: str | None = None
  version: int | None = None
  metadata: dict[str, str | int]
  system_prompt: str = Field(..., min_length=1)
  sections: list[SectionSchema] = Field(..., min_length=1)

  model_config = ConfigDict(extra="forbid")


class QuestionnaireFileSchema(BaseModel):
  questionnaire: QuestionnaireSchema

  model_config = ConfigDict(extra="forbid")
