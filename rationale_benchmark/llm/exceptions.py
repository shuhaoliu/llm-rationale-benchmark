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
    """Raised when response validation fails with detailed field-level error reporting."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        response_data: Optional[Any] = None,
        field_errors: Optional[dict[str, str]] = None,
        validation_context: Optional[dict[str, Any]] = None,
        recovery_suggestions: Optional[list[str]] = None,
    ):
        """Initialize ResponseValidationError with comprehensive error details.
        
        Args:
          message: Primary error message
          provider: Name of the provider that generated the response
          response_data: Raw response data that failed validation
          field_errors: Dictionary mapping field names to specific error messages
          validation_context: Additional context about the validation failure
          recovery_suggestions: List of suggested actions to resolve the error
        """
        super().__init__(message)
        self.provider = provider
        self.response_data = response_data
        self.field_errors = field_errors or {}
        self.validation_context = validation_context or {}
        self.recovery_suggestions = recovery_suggestions or []
    
    def add_field_error(self, field_name: str, error_message: str) -> None:
        """Add a field-specific validation error.
        
        Args:
          field_name: Name of the field that failed validation
          error_message: Specific error message for this field
        """
        self.field_errors[field_name] = error_message
    
    def add_recovery_suggestion(self, suggestion: str) -> None:
        """Add a recovery suggestion for resolving the validation error.
        
        Args:
          suggestion: Suggested action to resolve the error
        """
        self.recovery_suggestions.append(suggestion)
    
    def get_detailed_message(self) -> str:
        """Get a comprehensive error message including all validation details.
        
        Returns:
          Detailed error message with field errors and recovery suggestions
        """
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
          for i, suggestion in enumerate(self.recovery_suggestions, 1):
            details.append(f"  {i}. {suggestion}")
        
        return "\n".join(details)
    
    def has_field_errors(self) -> bool:
        """Check if there are any field-specific validation errors.
        
        Returns:
          True if field errors exist, False otherwise
        """
        return bool(self.field_errors)
    
    def get_field_error_count(self) -> int:
        """Get the number of field validation errors.
        
        Returns:
          Number of field validation errors
        """
        return len(self.field_errors)
    
    @classmethod
    def create_aggregated_error(
        cls,
        errors: list["ResponseValidationError"],
        provider: Optional[str] = None,
    ) -> "ResponseValidationError":
        """Create an aggregated error from multiple validation errors.
        
        Args:
          errors: List of ResponseValidationError instances to aggregate
          provider: Provider name for the aggregated error
          
        Returns:
          New ResponseValidationError with aggregated information
        """
        if not errors:
          raise ValueError("Cannot create aggregated error from empty error list")
        
        # Aggregate field errors from all errors
        aggregated_field_errors = {}
        aggregated_suggestions = []
        aggregated_context = {}
        
        for error in errors:
          aggregated_field_errors.update(error.field_errors)
          aggregated_suggestions.extend(error.recovery_suggestions)
          aggregated_context.update(error.validation_context)
        
        # Remove duplicate suggestions while preserving order
        unique_suggestions = []
        seen_suggestions = set()
        for suggestion in aggregated_suggestions:
          if suggestion not in seen_suggestions:
            unique_suggestions.append(suggestion)
            seen_suggestions.add(suggestion)
        
        # Create primary message
        error_count = len(errors)
        field_count = len(aggregated_field_errors)
        message = f"Multiple validation errors occurred: {error_count} errors affecting {field_count} fields"
        
        return cls(
          message=message,
          provider=provider or errors[0].provider,
          field_errors=aggregated_field_errors,
          validation_context=aggregated_context,
          recovery_suggestions=unique_suggestions,
        )


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
