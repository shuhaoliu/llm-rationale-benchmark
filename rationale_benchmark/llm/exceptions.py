"""Custom exception classes for LLM connector module."""

from typing import Any, Optional


class LLMError(Exception):
    """Base exception class for all LLM-related errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class LLMConnectorError(LLMError):
    """Base exception class for LLM connector-specific errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, cause)


class ConfigurationError(LLMError):
    """Raised when there are configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        field: Optional[str] = None,
    ):
        super().__init__(message)
        self.config_file = config_file
        self.field = field


class ProviderError(LLMError):
    """Raised when there are provider-specific errors."""

    def __init__(self, provider: str, message: str, cause: Optional[Exception] = None):
        super().__init__(f"[{provider}] {message}", cause)
        self.provider = provider


class ResponseValidationError(LLMError):
    """Raised when response validation fails."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        response_data: Optional[Any] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.response_data = response_data


class StreamingNotSupportedError(LLMError):
    """Raised when streaming parameters are detected but not supported."""

    def __init__(self, message: str, blocked_params: Optional[list[str]] = None):
        super().__init__(message)
        self.blocked_params = blocked_params or []


class AuthenticationError(ProviderError):
    """Raised when authentication with a provider fails."""

    def __init__(self, provider: str, message: str = "Authentication failed"):
        super().__init__(provider, message)


class RateLimitError(ProviderError):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        provider: str,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
    ):
        super().__init__(provider, message)
        self.retry_after = retry_after


class ModelNotFoundError(ProviderError):
    """Raised when a requested model is not available."""

    def __init__(self, provider: str, model: str):
        super().__init__(provider, f"Model '{model}' not found or not available")
        self.model = model


class NetworkError(LLMError):
    """Raised when network-related errors occur."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(f"Network error: {message}", cause)


class ConversationHistoryError(LLMConnectorError):
    """Raised when conversation history validation or processing fails."""

    def __init__(
        self,
        message: str,
        conversation_data: Optional[Any] = None,
        validation_errors: Optional[list[str]] = None,
        message_index: Optional[int] = None,
        field_name: Optional[str] = None,
    ):
        super().__init__(message)
        self.conversation_data = conversation_data
        self.validation_errors = validation_errors or []
        self.message_index = message_index
        self.field_name = field_name


class TimeoutError(LLMError):
    """Raised when requests timeout."""

    def __init__(
        self, message: str = "Request timed out", timeout_seconds: Optional[int] = None
    ):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class RetryExhaustedError(LLMError):
    """Raised when maximum retry attempts are exceeded."""

    def __init__(
        self, 
        message: str, 
        max_retries: int, 
        last_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.max_retries = max_retries
        self.last_exception = last_exception
