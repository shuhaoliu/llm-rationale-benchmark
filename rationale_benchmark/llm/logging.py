"""Structured logging and debugging support for LLM connector module."""

import logging
import os
import sys
from typing import Any, Dict, Optional

import structlog


class SensitiveDataFilter:
  """Filter to prevent sensitive data from being logged."""
  
  SENSITIVE_KEYS = {
    "api_key", "api_token", "authorization", "bearer", "password", "secret",
    "key", "token", "auth", "credential", "x-api-key", "openai_api_key",
    "anthropic_api_key", "gemini_api_key", "openrouter_api_key"
  }
  
  @classmethod
  def filter_sensitive_data(cls, data: Any) -> Any:
    """Recursively filter sensitive data from dictionaries and other structures.
    
    Args:
      data: Data structure to filter
      
    Returns:
      Filtered data structure with sensitive values replaced
    """
    if isinstance(data, dict):
      filtered = {}
      for key, value in data.items():
        if cls._is_sensitive_key(key):
          filtered[key] = cls._mask_sensitive_value(value)
        else:
          filtered[key] = cls.filter_sensitive_data(value)
      return filtered
    elif isinstance(data, list):
      return [cls.filter_sensitive_data(item) for item in data]
    elif isinstance(data, tuple):
      return tuple(cls.filter_sensitive_data(item) for item in data)
    else:
      return data
  
  @classmethod
  def _is_sensitive_key(cls, key: str) -> bool:
    """Check if a key name indicates sensitive data.
    
    Args:
      key: Key name to check
      
    Returns:
      True if key indicates sensitive data
    """
    key_lower = key.lower().replace("-", "_").replace(" ", "_")
    return any(sensitive in key_lower for sensitive in cls.SENSITIVE_KEYS)
  
  @classmethod
  def _mask_sensitive_value(cls, value: Any) -> str:
    """Mask a sensitive value for logging.
    
    Args:
      value: Sensitive value to mask
      
    Returns:
      Masked representation of the value
    """
    if value is None:
      return "[NONE]"
    
    value_str = str(value)
    if len(value_str) <= 8:
      return "[REDACTED]"
    else:
      # Show first 4 and last 4 characters with masking in between
      return f"{value_str[:4]}...{value_str[-4:]}"


def configure_logging(
  debug_mode: bool = False,
  log_level: Optional[str] = None,
  log_file: Optional[str] = None,
  structured: bool = True,
) -> None:
  """Configure structured logging for the LLM connector.
  
  Args:
    debug_mode: Enable debug mode with detailed logging
    log_level: Override log level (DEBUG, INFO, WARNING, ERROR)
    log_file: Optional file path for log output
    structured: Use structured JSON logging format
  """
  # Determine log level
  if log_level:
    level = getattr(logging, log_level.upper(), logging.INFO)
  elif debug_mode:
    level = logging.DEBUG
  else:
    level = logging.INFO
  
  # Configure structlog
  processors = [
    structlog.stdlib.filter_by_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    _add_request_context,
    _filter_sensitive_processor,
  ]
  
  if structured:
    processors.append(structlog.processors.JSONRenderer())
  else:
    processors.append(structlog.dev.ConsoleRenderer())
  
  structlog.configure(
    processors=processors,
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
  )
  
  # Configure standard library logging
  handlers = []
  
  # Console handler
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setLevel(level)
  handlers.append(console_handler)
  
  # File handler if specified
  if log_file:
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    handlers.append(file_handler)
  
  # Configure root logger
  logging.basicConfig(
    level=level,
    handlers=handlers,
    format="%(message)s" if structured else "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  )
  
  # Set specific logger levels
  logging.getLogger("aiohttp").setLevel(logging.WARNING)
  logging.getLogger("urllib3").setLevel(logging.WARNING)


def _add_request_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
  """Add request context to log events.
  
  Args:
    logger: Logger instance
    method_name: Log method name
    event_dict: Event dictionary
    
  Returns:
    Enhanced event dictionary with context
  """
  # Add process and thread information for debugging
  event_dict["process_id"] = os.getpid()
  
  # Add module context if available
  if hasattr(logger, "_context"):
    event_dict.update(logger._context)
  
  return event_dict


def _filter_sensitive_processor(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
  """Processor to filter sensitive data from log events.
  
  Args:
    logger: Logger instance
    method_name: Log method name
    event_dict: Event dictionary
    
  Returns:
    Filtered event dictionary
  """
  return SensitiveDataFilter.filter_sensitive_data(event_dict)


class LLMLogger:
  """Enhanced logger for LLM operations with context management."""
  
  def __init__(self, name: str, provider: Optional[str] = None):
    """Initialize LLM logger with context.
    
    Args:
      name: Logger name
      provider: Optional provider name for context
    """
    self.logger = structlog.get_logger(name)
    self.context = {}
    
    if provider:
      self.context["provider"] = provider
  
  def bind(self, **kwargs: Any) -> "LLMLogger":
    """Bind additional context to the logger.
    
    Args:
      **kwargs: Context key-value pairs
      
    Returns:
      New logger instance with bound context
    """
    new_logger = LLMLogger(self.logger.name)
    new_logger.context = {**self.context, **kwargs}
    new_logger.logger = self.logger.bind(**new_logger.context)
    return new_logger
  
  def debug(self, message: str, **kwargs: Any) -> None:
    """Log debug message with context.
    
    Args:
      message: Log message
      **kwargs: Additional context
    """
    filtered_kwargs = SensitiveDataFilter.filter_sensitive_data(kwargs)
    self.logger.debug(message, **filtered_kwargs)
  
  def info(self, message: str, **kwargs: Any) -> None:
    """Log info message with context.
    
    Args:
      message: Log message
      **kwargs: Additional context
    """
    filtered_kwargs = SensitiveDataFilter.filter_sensitive_data(kwargs)
    self.logger.info(message, **filtered_kwargs)
  
  def warning(self, message: str, **kwargs: Any) -> None:
    """Log warning message with context.
    
    Args:
      message: Log message
      **kwargs: Additional context
    """
    filtered_kwargs = SensitiveDataFilter.filter_sensitive_data(kwargs)
    self.logger.warning(message, **filtered_kwargs)
  
  def error(self, message: str, **kwargs: Any) -> None:
    """Log error message with context.
    
    Args:
      message: Log message
      **kwargs: Additional context
    """
    filtered_kwargs = SensitiveDataFilter.filter_sensitive_data(kwargs)
    self.logger.error(message, **filtered_kwargs)
  
  def log_request(
    self,
    method: str,
    url: str,
    model: str,
    request_data: Optional[Dict[str, Any]] = None,
    **kwargs: Any
  ) -> None:
    """Log LLM API request with context.
    
    Args:
      method: HTTP method
      url: Request URL
      model: Model name
      request_data: Request payload (will be filtered for sensitive data)
      **kwargs: Additional context
    """
    context = {
      "event_type": "llm_request",
      "method": method,
      "url": url,
      "model": model,
      **kwargs
    }
    
    if request_data:
      # Filter sensitive data from request
      filtered_request = SensitiveDataFilter.filter_sensitive_data(request_data)
      context["request_size"] = len(str(request_data))
      context["request_keys"] = list(request_data.keys()) if isinstance(request_data, dict) else []
      
      # Only include request data in debug mode
      if self.logger.isEnabledFor(logging.DEBUG):
        context["request_data"] = filtered_request
    
    self.info("LLM API request initiated", **context)
  
  def log_response(
    self,
    status_code: int,
    model: str,
    latency_ms: int,
    response_data: Optional[Dict[str, Any]] = None,
    token_count: Optional[int] = None,
    **kwargs: Any
  ) -> None:
    """Log LLM API response with context.
    
    Args:
      status_code: HTTP status code
      model: Model name
      latency_ms: Response latency in milliseconds
      response_data: Response payload (will be filtered for sensitive data)
      token_count: Token count if available
      **kwargs: Additional context
    """
    context = {
      "event_type": "llm_response",
      "status_code": status_code,
      "model": model,
      "latency_ms": latency_ms,
      **kwargs
    }
    
    if token_count:
      context["token_count"] = token_count
    
    if response_data:
      context["response_size"] = len(str(response_data))
      context["response_keys"] = list(response_data.keys()) if isinstance(response_data, dict) else []
      
      # Only include response data in debug mode
      if self.logger.isEnabledFor(logging.DEBUG):
        filtered_response = SensitiveDataFilter.filter_sensitive_data(response_data)
        context["response_data"] = filtered_response
    
    if status_code >= 400:
      self.error("LLM API request failed", **context)
    else:
      self.info("LLM API request completed", **context)
  
  def log_streaming_detection(
    self,
    blocked_params: list[str],
    provider: str,
    **kwargs: Any
  ) -> None:
    """Log streaming parameter detection and removal.
    
    Args:
      blocked_params: List of blocked streaming parameters
      provider: Provider name
      **kwargs: Additional context
    """
    context = {
      "event_type": "streaming_blocked",
      "provider": provider,
      "blocked_params": blocked_params,
      "blocked_count": len(blocked_params),
      **kwargs
    }
    
    self.warning("Streaming parameters detected and blocked", **context)
  
  def log_validation_error(
    self,
    error_type: str,
    field_errors: Dict[str, str],
    provider: str,
    **kwargs: Any
  ) -> None:
    """Log response validation error with context.
    
    Args:
      error_type: Type of validation error
      field_errors: Dictionary of field-specific errors
      provider: Provider name
      **kwargs: Additional context
    """
    context = {
      "event_type": "validation_error",
      "error_type": error_type,
      "provider": provider,
      "field_error_count": len(field_errors),
      "failed_fields": list(field_errors.keys()),
      **kwargs
    }
    
    # Include field errors in debug mode
    if self.logger.isEnabledFor(logging.DEBUG):
      context["field_errors"] = field_errors
    
    self.error("Response validation failed", **context)
  
  def log_queue_status(
    self,
    provider: str,
    queue_size: int,
    active_requests: int,
    **kwargs: Any
  ) -> None:
    """Log provider queue status for debugging.
    
    Args:
      provider: Provider name
      queue_size: Current queue size
      active_requests: Number of active requests
      **kwargs: Additional context
    """
    context = {
      "event_type": "queue_status",
      "provider": provider,
      "queue_size": queue_size,
      "active_requests": active_requests,
      **kwargs
    }
    
    self.debug("Provider queue status", **context)


def get_llm_logger(name: str, provider: Optional[str] = None) -> LLMLogger:
  """Get an LLM logger instance with optional provider context.
  
  Args:
    name: Logger name
    provider: Optional provider name
    
  Returns:
    LLMLogger instance
  """
  return LLMLogger(name, provider)


def is_debug_enabled() -> bool:
  """Check if debug logging is enabled.
  
  Returns:
    True if debug logging is enabled
  """
  return logging.getLogger().isEnabledFor(logging.DEBUG)


def log_system_info() -> None:
  """Log system information for debugging purposes."""
  logger = get_llm_logger("rationale_benchmark.llm.system")
  
  import platform
  import sys
  
  logger.info(
    "System information",
    python_version=sys.version,
    platform=platform.platform(),
    architecture=platform.architecture(),
    processor=platform.processor(),
  )