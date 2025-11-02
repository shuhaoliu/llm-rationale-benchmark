"""Conversation orchestration for provider-agnostic interactions."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, List, Optional

from .config.connector_models import LLMConnectorConfig, ResponseFormat
from .exceptions import (
  ConversationArchivedError,
  LLMConnectorError,
  RetryableProviderError,
  StreamingNotSupportedError,
  ValidationFailedError,
)
from .provider_client import BaseProviderClient, ProviderResponse


ValidatorFn = Callable[["LLMResponse"], bool]


@dataclass
class ConversationTurn:
  """Single conversation turn in chronological order."""

  role: str
  content: str
  timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
  verification_errors: list[str] = field(default_factory=list)
  include_in_request: bool = True

  def to_dict(self) -> dict[str, Any]:
    return {
      "role": self.role,
      "content": self.content,
      "timestamp": self.timestamp.isoformat(),
      "verification_errors": list(self.verification_errors),
      "include_in_request": self.include_in_request,
    }


@dataclass
class LLMResponse:
  """Response returned from :meth:`LLMConversation.ask`."""

  text: str
  parsed: Any
  provider: str
  model: str
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConversationArchive:
  """Immutable archive exported after :meth:`LLMConversation.archive`."""

  config_snapshot: dict[str, Any]
  system_prompt: Optional[str]
  response_format: ResponseFormat
  turns: list[ConversationTurn]
  metadata: dict[str, Any]

  def to_dict(self) -> dict[str, Any]:
    return {
      "config": self.config_snapshot,
      "system_prompt": self.system_prompt,
      "response_format": self.response_format.value,
      "turns": [turn.to_dict() for turn in self.turns],
      "metadata": dict(self.metadata),
    }

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), ensure_ascii=False)


class ConversationState:
  ACTIVE = "active"
  ARCHIVED = "archived"


class LLMConversation:
  """Stateful conversation orchestrating retries and validation."""

  def __init__(
    self,
    config: LLMConnectorConfig,
    provider_client: BaseProviderClient,
    *,
    system_prompt: Optional[str] = None,
    time_source: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
  ) -> None:
    self.config = config
    self.provider_client = provider_client
    self.system_prompt = system_prompt if system_prompt is not None else config.system_prompt
    self.history: List[ConversationTurn] = []
    self.state = ConversationState.ACTIVE
    self._time_source = time_source or (lambda: datetime.now(UTC))
    self._sleep_fn = sleep_fn or time.sleep
    self._stats = {
      "provider_attempts": 0,
      "validator_failures": 0,
      "json_parse_failures": 0,
      "retryable_errors": 0,
    }

    if self.system_prompt:
      self.history.append(
        ConversationTurn(
          role="system",
          content=self.system_prompt,
          timestamp=self._time_source(),
        )
      )

  def ask(
    self,
    question: str,
    validator: Optional[ValidatorFn] = None,
    *,
    max_attempts: Optional[int] = None,
  ) -> LLMResponse:
    if not question.strip():
      raise ValueError("Question cannot be empty")

    if self.state != ConversationState.ACTIVE:
      raise ConversationArchivedError("Conversation is archived")

    attempts_allowed = max_attempts or self.config.retry.max_attempts
    attempts_allowed = max(1, attempts_allowed)

    user_turn = ConversationTurn(
      role="user",
      content=question,
      timestamp=self._time_source(),
    )
    self.history.append(user_turn)

    last_error: Optional[Exception] = None
    validator_errors: list[str] = []

    for attempt in range(1, attempts_allowed + 1):
      self._stats["provider_attempts"] += 1

      try:
        response = self._invoke_provider()
      except RetryableProviderError as exc:
        last_error = exc
        self._stats["retryable_errors"] += 1
        if attempt == attempts_allowed:
          raise LLMConnectorError(str(exc)) from exc
        self._sleep_for_attempt(attempt, retry_after=exc.retry_after)
        continue

      assistant_turn = ConversationTurn(
        role="assistant",
        content=response.content,
        timestamp=self._time_source(),
      )

      parsed, parse_error = self._parse_response(response)
      if parse_error:
        assistant_turn.verification_errors.append(parse_error)
        assistant_turn.include_in_request = False
        self._stats["json_parse_failures"] += 1
        self.history.append(assistant_turn)
        validator_errors.append(parse_error)
        last_error = None
        if attempt == attempts_allowed:
          raise ValidationFailedError(parse_error, errors=list(validator_errors))
        self._sleep_for_attempt(attempt)
        continue

      llm_response = LLMResponse(
        text=response.content,
        parsed=parsed,
        provider=self.config.provider.value,
        model=self.config.model,
        metadata=response.metadata,
      )

      if validator:
        try:
          is_valid = validator(llm_response)
        except Exception as exc:  # pragma: no cover - defensive
          assistant_turn.verification_errors.append(f"validator_error: {exc}")
          assistant_turn.include_in_request = False
          self.history.append(assistant_turn)
          raise

        if not is_valid:
          assistant_turn.verification_errors.append("validator_rejected")
          assistant_turn.include_in_request = False
          self._stats["validator_failures"] += 1
          self.history.append(assistant_turn)
          validator_errors.append("validator_rejected")
          if attempt == attempts_allowed:
            raise ValidationFailedError(
              "Validator rejected response",
              errors=list(validator_errors),
            )
          self._sleep_for_attempt(attempt)
          continue

      self.history.append(assistant_turn)
      return llm_response

    message = "Conversation failed without producing a valid response"
    raise LLMConnectorError(message) from last_error

  def archive(self) -> LLMConversationArchive:
    if self.state != ConversationState.ACTIVE:
      raise ConversationArchivedError("Conversation already archived")

    self.state = ConversationState.ARCHIVED

    metadata = {
      "attempts": self._stats["provider_attempts"],
      "validator_failures": self._stats["validator_failures"],
      "json_parse_failures": self._stats["json_parse_failures"],
      "retryable_errors": self._stats["retryable_errors"],
    }

    return LLMConversationArchive(
      config_snapshot=self.config.sanitized_snapshot(),
      system_prompt=self.system_prompt,
      response_format=self.config.response_format,
      turns=list(self.history),
      metadata=metadata,
    )

  def _invoke_provider(self) -> ProviderResponse:
    messages = [
      {
        "role": turn.role,
        "content": turn.content,
      }
      for turn in self.history
      if turn.include_in_request
    ]

    use_streaming = (
      self.config.requires_streaming
      or self.provider_client.requires_streaming()
    )

    try:
      return self.provider_client.generate(
        messages,
        response_format=self.config.response_format,
        stream=use_streaming,
      )
    except StreamingNotSupportedError as exc:  # pragma: no cover - guard rail
      raise LLMConnectorError(str(exc)) from exc

  def _parse_response(self, response: ProviderResponse) -> tuple[Any, Optional[str]]:
    if self.config.response_format == ResponseFormat.JSON:
      try:
        return json.loads(response.content), None
      except json.JSONDecodeError as exc:
        return None, f"json_parse_error: {exc.msg}"

    return response.content, None

  def _sleep_for_attempt(
    self, attempt: int, *, retry_after: Optional[int] = None
  ) -> None:
    delay = (
      float(retry_after)
      if retry_after is not None
      else self.config.retry.compute_delay(attempt)
    )
    if self.config.retry.jitter:
      delay += random.uniform(0, self.config.retry.jitter)
    self._sleep_fn(delay)

