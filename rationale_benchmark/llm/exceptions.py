"""Custom exception classes for the LLM connector module."""

from __future__ import annotations

from typing import Any, Optional


class LLMError(Exception):
  """Base exception for all LLM-related errors."""

  def __init__(self, message: str, cause: Optional[Exception] = None):
    super().__init__(message)
    self.cause = cause


class LLMConnectorError(LLMError):
  """Base exception for connector-specific errors."""

  def __init__(self, message: str, cause: Optional[Exception] = None):
    super().__init__(message, cause)


class ConfigurationError(LLMError):
  """Raised when configuration validation fails."""

  def __init__(
    self,
    message: str,
    *,
    config_file: Optional[str] = None,
    field: Optional[str] = None,
  ) -> None:
    super().__init__(message)
    self.config_file = config_file
    self.field = field


class ProviderError(LLMError):
  """Raised when provider-specific errors occur."""

  def __init__(
    self, provider: str, message: str, cause: Optional[Exception] = None
  ) -> None:
    super().__init__(f"[{provider}] {message}", cause)
    self.provider = provider


class RetryableProviderError(ProviderError):
  """Provider error that may succeed on retry."""

  def __init__(
    self,
    provider: str,
    message: str,
    *,
    retry_after: Optional[int] = None,
    cause: Optional[Exception] = None,
  ) -> None:
    super().__init__(provider, message, cause)
    self.retry_after = retry_after


class ResponseValidationError(LLMError):
  """Raised when response validation fails with detailed information."""

  def __init__(
    self,
    message: str,
    provider: Optional[str] = None,
    response_data: Optional[Any] = None,
    field_errors: Optional[dict[str, str]] = None,
    validation_context: Optional[dict[str, Any]] = None,
    recovery_suggestions: Optional[list[str]] = None,
  ) -> None:
    super().__init__(message)
    self.provider = provider
    self.response_data = response_data
    self.field_errors = field_errors or {}
    self.validation_context = validation_context or {}
    self.recovery_suggestions = recovery_suggestions or []

  def add_field_error(self, field_name: str, error_message: str) -> None:
    self.field_errors[field_name] = error_message

  def add_recovery_suggestion(self, suggestion: str) -> None:
    self.recovery_suggestions.append(suggestion)

  def get_detailed_message(self) -> str:
    details = [str(self)]

    if self.provider:
      details.append(f"Provider: {self.provider}")

    if self.field_errors:
      details.append("Field validation errors:")
      for field, error in self.field_errors.items():
        details.append(f"  - {field}: {error}")

    if self.validation_context:
      details.append("Validation context:")
      for key, value in self.validation_context.items():
        details.append(f"  - {key}: {value}")

    if self.recovery_suggestions:
      details.append("Recovery suggestions:")
      for index, suggestion in enumerate(self.recovery_suggestions, 1):
        details.append(f"  {index}. {suggestion}")

    return "\n".join(details)

  def has_field_errors(self) -> bool:
    return bool(self.field_errors)

  def get_field_error_count(self) -> int:
    return len(self.field_errors)

  @classmethod
  def create_aggregated_error(
    cls,
    errors: list["ResponseValidationError"],
    provider: Optional[str] = None,
  ) -> "ResponseValidationError":
    if not errors:
      raise ValueError("Cannot aggregate an empty error list")

    aggregated_field_errors: dict[str, str] = {}
    aggregated_context: dict[str, Any] = {}
    aggregated_suggestions: list[str] = []

    for error in errors:
      aggregated_field_errors.update(error.field_errors)
      aggregated_context.update(error.validation_context)
      aggregated_suggestions.extend(error.recovery_suggestions)

    unique_suggestions: list[str] = []
    seen = set()
    for suggestion in aggregated_suggestions:
      if suggestion not in seen:
        unique_suggestions.append(suggestion)
        seen.add(suggestion)

    message = (
      "Multiple validation errors occurred: "
      f"{len(errors)} errors affecting {len(aggregated_field_errors)} fields"
    )

    return cls(
      message=message,
      provider=provider or errors[0].provider,
      field_errors=aggregated_field_errors,
      validation_context=aggregated_context,
      recovery_suggestions=unique_suggestions,
    )


class StreamingNotSupportedError(LLMError):
  """Raised when streaming parameters are detected but not supported."""

  def __init__(
    self, message: str, blocked_params: Optional[list[str]] = None
  ) -> None:
    super().__init__(message)
    self.blocked_params = blocked_params or []


class AuthenticationError(ProviderError):
  """Raised when authentication with a provider fails."""

  def __init__(self, provider: str, message: str = "Authentication failed"):
    super().__init__(provider, message)


class RateLimitError(ProviderError):
  """Raised when provider rate limits are exceeded."""

  def __init__(
    self,
    provider: str,
    message: str = "Rate limit exceeded",
    retry_after: Optional[int] = None,
  ) -> None:
    super().__init__(provider, message)
    self.retry_after = retry_after


class ModelNotFoundError(ProviderError):
  """Raised when a requested model is unavailable."""

  def __init__(self, provider: str, model: str) -> None:
    super().__init__(provider, f"Model '{model}' not found or unavailable")
    self.model = model


class NetworkError(LLMError):
  """Raised when network-related errors occur."""

  def __init__(self, message: str, cause: Optional[Exception] = None):
    super().__init__(f"Network error: {message}", cause)


class ConversationHistoryError(LLMConnectorError):
  """Raised when conversation history validation fails."""

  def __init__(
    self,
    message: str,
    conversation_data: Optional[Any] = None,
    validation_errors: Optional[list[str]] = None,
    message_index: Optional[int] = None,
    field_name: Optional[str] = None,
  ) -> None:
    super().__init__(message)
    self.conversation_data = conversation_data
    self.validation_errors = validation_errors or []
    self.message_index = message_index
    self.field_name = field_name


class TimeoutError(LLMError):
  """Raised when a request times out."""

  def __init__(
    self, message: str = "Request timed out", timeout_seconds: Optional[int] = None
  ) -> None:
    super().__init__(message)
    self.timeout_seconds = timeout_seconds


class RetryExhaustedError(LLMError):
  """Raised when maximum retry attempts are exceeded."""

  def __init__(
    self,
    message: str,
    max_retries: int,
    last_exception: Optional[Exception] = None,
  ) -> None:
    super().__init__(message)
    self.max_retries = max_retries
    self.last_exception = last_exception


class ConversationArchivedError(LLMConnectorError):
  """Raised when an operation targets an archived conversation."""

  def __init__(self, message: str = "Conversation is archived") -> None:
    super().__init__(message)


class ValidationFailedError(LLMConnectorError):
  """Raised when validation fails after exhausting retries."""

  def __init__(self, message: str, *, errors: Optional[list[str]] = None) -> None:
    super().__init__(message)
    self.errors = errors or []

