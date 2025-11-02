"""Validated configuration models for the LLM connector layer."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ProviderType(str, Enum):
  """Supported provider identifiers."""

  OPENAI = "openai"
  OPENAI_COMPATIBLE = "openai_compatible"
  ANTHROPIC = "anthropic"
  GEMINI = "gemini"


class ResponseFormat(str, Enum):
  """Available response rendering modes."""

  JSON = "json"
  TEXT = "text"


class RetryPolicy(BaseModel):
  """Retry behaviour applied when a turn needs to be retried."""

  model_config = ConfigDict(extra="forbid")

  max_attempts: int = Field(default=3, ge=1)
  initial_delay: float = Field(default=1.0, gt=0.0)
  max_delay: float = Field(default=60.0, ge=0.0)
  multiplier: float = Field(default=2.0, ge=1.0)
  jitter: float = Field(default=0.0, ge=0.0)

  def compute_delay(self, attempt: int) -> float:
    """Return the delay (in seconds) for the provided attempt."""

    if attempt < 1:
      raise ValueError("Attempt number must be >= 1")

    delay = self.initial_delay * (self.multiplier ** (attempt - 1))
    if self.max_delay:
      delay = min(delay, self.max_delay)
    return delay


class LLMConnectorConfig(BaseModel):
  """Concrete configuration for a provider/model tuple."""

  model_config = ConfigDict(extra="forbid")

  provider: ProviderType
  model: str
  api_key: Optional[str] = None
  endpoint: Optional[str] = None
  base_url: Optional[str] = None
  timeout_seconds: int = Field(default=30, ge=1)
  retry: RetryPolicy = Field(default_factory=RetryPolicy)
  response_format: ResponseFormat = ResponseFormat.JSON
  temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
  top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
  max_tokens: Optional[int] = Field(default=None, ge=1)
  system_prompt: Optional[str] = None
  metadata: Dict[str, Any] = Field(default_factory=dict)
  provider_specific: Dict[str, Any] = Field(default_factory=dict)
  requires_streaming: bool = False

  @field_validator("model")
  @classmethod
  def _validate_model(cls, value: str) -> str:
    if not value.strip():
      raise ValueError("Model identifier cannot be empty")
    return value

  @field_validator("endpoint")
  @classmethod
  def _validate_endpoint(cls, value: Optional[str]) -> Optional[str]:
    if value is not None and not value.strip():
      raise ValueError("Endpoint must be a non-empty string when provided")
    return value

  @field_validator("base_url")
  @classmethod
  def _validate_base_url(cls, value: Optional[str]) -> Optional[str]:
    if value is not None and not value.strip():
      raise ValueError("Base URL must be a non-empty string when provided")
    return value

  @field_validator("metadata")
  @classmethod
  def _ensure_metadata_mapping(
    cls, value: Dict[str, Any]
  ) -> Dict[str, Any]:
    return dict(value)

  @field_validator("provider_specific")
  @classmethod
  def _ensure_provider_specific_mapping(
    cls, value: Dict[str, Any]
  ) -> Dict[str, Any]:
    return dict(value)

  def model_key(self) -> str:
    """Return the fully-qualified provider/model key."""

    return f"{self.provider.value}/{self.model}"

  def cache_key(self) -> str:
    """Return a deterministic cache key excluding sensitive material."""

    payload = self.model_dump(exclude={"api_key"}, mode="json")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest

  def sanitized_snapshot(self) -> Dict[str, Any]:
    """Return a serialisable snapshot with secrets masked."""

    snapshot = self.model_dump(mode="json")
    if snapshot.get("api_key"):
      snapshot["api_key"] = "***"
    return snapshot


def format_validation_error(error: ValidationError) -> str:
  """Create a readable error message from a pydantic validation error."""

  details = []
  for issue in error.errors():
    location = ".".join(str(part) for part in issue.get("loc", []))
    details.append(f"{location or '<root>'}: {issue.get('msg')}")
  return "\n".join(details)

