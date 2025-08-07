"""Unit tests for logging and debugging support."""

import logging
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

import pytest
import structlog

from rationale_benchmark.llm.logging import (
  SensitiveDataFilter,
  configure_logging,
  LLMLogger,
  get_llm_logger,
  is_debug_enabled,
  log_system_info,
)


class TestSensitiveDataFilter:
  """Test sensitive data filtering functionality."""
  
  def test_filter_sensitive_data_dict(self):
    """Test filtering sensitive data from dictionaries."""
    data = {
      "api_key": "sk-1234567890abcdef",
      "authorization": "Bearer token123",
      "normal_field": "safe_value",
      "nested": {
        "openai_api_key": "sk-nested123",
        "safe_nested": "safe_value",
      },
    }
    
    filtered = SensitiveDataFilter.filter_sensitive_data(data)
    
    assert filtered["api_key"] == "sk-1...cdef"
    assert filtered["authorization"] == "Bear...n123"
    assert filtered["normal_field"] == "safe_value"
    assert filtered["nested"]["openai_api_key"] == "sk-n...e123"
    assert filtered["nested"]["safe_nested"] == "safe_value"
  
  def test_filter_sensitive_data_list(self):
    """Test filtering sensitive data from lists."""
    data = [
      {"api_key": "secret123"},
      {"normal": "value"},
      ["nested", {"token": "abc123"}],
    ]
    
    filtered = SensitiveDataFilter.filter_sensitive_data(data)
    
    assert filtered[0]["api_key"] == "secr...t123"
    assert filtered[1]["normal"] == "value"
    assert filtered[2][1]["token"] == "[REDACTED]"  # Short value
  
  def test_filter_sensitive_data_tuple(self):
    """Test filtering sensitive data from tuples."""
    data = (
      {"password": "secret123"},
      "normal_value",
      {"key": "short"},
    )
    
    filtered = SensitiveDataFilter.filter_sensitive_data(data)
    
    assert isinstance(filtered, tuple)
    assert filtered[0]["password"] == "secr...t123"
    assert filtered[1] == "normal_value"
    assert filtered[2]["key"] == "[REDACTED]"
  
  def test_filter_sensitive_data_primitive_types(self):
    """Test filtering primitive types (should return unchanged)."""
    assert SensitiveDataFilter.filter_sensitive_data("string") == "string"
    assert SensitiveDataFilter.filter_sensitive_data(123) == 123
    assert SensitiveDataFilter.filter_sensitive_data(True) is True
    assert SensitiveDataFilter.filter_sensitive_data(None) is None
  
  def test_is_sensitive_key(self):
    """Test sensitive key detection."""
    sensitive_keys = [
      "api_key", "API_KEY", "Api-Key", "api key",
      "authorization", "bearer", "token", "secret",
      "openai_api_key", "anthropic_api_key", "password",
    ]
    
    for key in sensitive_keys:
      assert SensitiveDataFilter._is_sensitive_key(key), f"Key '{key}' should be sensitive"
    
    safe_keys = ["model", "temperature", "max_tokens", "content", "role"]
    
    for key in safe_keys:
      assert not SensitiveDataFilter._is_sensitive_key(key), f"Key '{key}' should not be sensitive"
  
  def test_mask_sensitive_value(self):
    """Test sensitive value masking."""
    # Long values
    assert SensitiveDataFilter._mask_sensitive_value("sk-1234567890abcdef") == "sk-1...cdef"
    assert SensitiveDataFilter._mask_sensitive_value("very_long_secret_token_123") == "very...t_123"
    
    # Short values
    assert SensitiveDataFilter._mask_sensitive_value("short") == "[REDACTED]"
    assert SensitiveDataFilter._mask_sensitive_value("abc") == "[REDACTED]"
    
    # None value
    assert SensitiveDataFilter._mask_sensitive_value(None) == "[NONE]"
    
    # Non-string values
    assert SensitiveDataFilter._mask_sensitive_value(123456789) == "1234...6789"


class TestLoggingConfiguration:
  """Test logging configuration functionality."""
  
  def setup_method(self):
    """Reset logging configuration before each test."""
    # Clear any existing structlog configuration
    structlog.reset_defaults()
    
    # Reset standard library logging
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.WARNING)
  
  def test_configure_logging_default(self):
    """Test default logging configuration."""
    configure_logging()
    
    # Check that structlog is configured
    logger = structlog.get_logger("test")
    assert logger is not None
    
    # Check that standard library logging is configured
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) > 0
  
  def test_configure_logging_debug_mode(self):
    """Test logging configuration with debug mode enabled."""
    configure_logging(debug_mode=True)
    
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
  
  def test_configure_logging_custom_level(self):
    """Test logging configuration with custom log level."""
    configure_logging(log_level="WARNING")
    
    root_logger = logging.getLogger()
    assert root_logger.level == logging.WARNING
  
  @patch('logging.FileHandler')
  def test_configure_logging_with_file(self, mock_file_handler):
    """Test logging configuration with file output."""
    mock_handler = Mock()
    mock_file_handler.return_value = mock_handler
    
    configure_logging(log_file="/tmp/test.log")
    
    mock_file_handler.assert_called_once_with("/tmp/test.log")
    mock_handler.setLevel.assert_called()
  
  def test_configure_logging_structured_vs_console(self):
    """Test structured vs console logging format."""
    # Test structured logging (default)
    configure_logging(structured=True)
    logger = structlog.get_logger("test")
    assert logger is not None
    
    # Test console logging
    configure_logging(structured=False)
    logger = structlog.get_logger("test")
    assert logger is not None


class TestLLMLogger:
  """Test LLMLogger enhanced logging functionality."""
  
  def setup_method(self):
    """Setup for each test."""
    configure_logging(debug_mode=True)
  
  @pytest.fixture
  def llm_logger(self):
    """Create LLMLogger for testing."""
    return LLMLogger("test_logger", "openai")
  
  def test_llm_logger_creation(self, llm_logger):
    """Test LLMLogger creation with context."""
    assert llm_logger.context["provider"] == "openai"
    assert llm_logger.logger is not None
  
  def test_llm_logger_bind(self, llm_logger):
    """Test binding additional context to logger."""
    bound_logger = llm_logger.bind(model="gpt-4", request_id="req_123")
    
    assert bound_logger.context["provider"] == "openai"
    assert bound_logger.context["model"] == "gpt-4"
    assert bound_logger.context["request_id"] == "req_123"
    
    # Original logger should be unchanged
    assert "model" not in llm_logger.context
  
  @patch('rationale_benchmark.llm.logging.SensitiveDataFilter.filter_sensitive_data')
  def test_llm_logger_debug(self, mock_filter, llm_logger):
    """Test debug logging with sensitive data filtering."""
    mock_filter.return_value = {"safe": "data"}
    
    with patch.object(llm_logger.logger, 'debug') as mock_debug:
      llm_logger.debug("Test message", api_key="secret", safe="data")
      
      mock_filter.assert_called_once_with({"api_key": "secret", "safe": "data"})
      mock_debug.assert_called_once_with("Test message", safe="data")
  
  @patch('rationale_benchmark.llm.logging.SensitiveDataFilter.filter_sensitive_data')
  def test_llm_logger_info(self, mock_filter, llm_logger):
    """Test info logging with sensitive data filtering."""
    mock_filter.return_value = {"filtered": "data"}
    
    with patch.object(llm_logger.logger, 'info') as mock_info:
      llm_logger.info("Info message", token="secret123")
      
      mock_filter.assert_called_once_with({"token": "secret123"})
      mock_info.assert_called_once_with("Info message", filtered="data")
  
  @patch('rationale_benchmark.llm.logging.SensitiveDataFilter.filter_sensitive_data')
  def test_llm_logger_warning(self, mock_filter, llm_logger):
    """Test warning logging with sensitive data filtering."""
    mock_filter.return_value = {"clean": "data"}
    
    with patch.object(llm_logger.logger, 'warning') as mock_warning:
      llm_logger.warning("Warning message", password="secret")
      
      mock_filter.assert_called_once()
      mock_warning.assert_called_once_with("Warning message", clean="data")
  
  @patch('rationale_benchmark.llm.logging.SensitiveDataFilter.filter_sensitive_data')
  def test_llm_logger_error(self, mock_filter, llm_logger):
    """Test error logging with sensitive data filtering."""
    mock_filter.return_value = {"safe": "data"}
    
    with patch.object(llm_logger.logger, 'error') as mock_error:
      llm_logger.error("Error message", credential="secret")
      
      mock_filter.assert_called_once()
      mock_error.assert_called_once_with("Error message", safe="data")
  
  def test_log_request(self, llm_logger):
    """Test LLM API request logging."""
    request_data = {
      "model": "gpt-4",
      "messages": [{"role": "user", "content": "Hello"}],
      "api_key": "secret123",
    }
    
    with patch.object(llm_logger, 'info') as mock_info:
      llm_logger.log_request(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        model="gpt-4",
        request_data=request_data,
        temperature=0.7,
      )
      
      mock_info.assert_called_once()
      call_args = mock_info.call_args
      assert call_args[0][0] == "LLM API request initiated"
      assert call_args[1]["event_type"] == "llm_request"
      assert call_args[1]["method"] == "POST"
      assert call_args[1]["model"] == "gpt-4"
      assert call_args[1]["temperature"] == 0.7
      assert call_args[1]["request_size"] == len(str(request_data))
  
  def test_log_response_success(self, llm_logger):
    """Test successful LLM API response logging."""
    response_data = {
      "choices": [{"message": {"content": "Hello world"}}],
      "usage": {"total_tokens": 10},
    }
    
    with patch.object(llm_logger, 'info') as mock_info:
      llm_logger.log_response(
        status_code=200,
        model="gpt-4",
        latency_ms=1500,
        response_data=response_data,
        token_count=10,
      )
      
      mock_info.assert_called_once()
      call_args = mock_info.call_args
      assert call_args[0][0] == "LLM API request completed"
      assert call_args[1]["event_type"] == "llm_response"
      assert call_args[1]["status_code"] == 200
      assert call_args[1]["latency_ms"] == 1500
      assert call_args[1]["token_count"] == 10
  
  def test_log_response_error(self, llm_logger):
    """Test error LLM API response logging."""
    with patch.object(llm_logger, 'error') as mock_error:
      llm_logger.log_response(
        status_code=429,
        model="gpt-4",
        latency_ms=500,
        error_message="Rate limit exceeded",
      )
      
      mock_error.assert_called_once()
      call_args = mock_error.call_args
      assert call_args[0][0] == "LLM API request failed"
      assert call_args[1]["status_code"] == 429
  
  def test_log_streaming_detection(self, llm_logger):
    """Test streaming parameter detection logging."""
    blocked_params = ["stream", "stream_options"]
    
    with patch.object(llm_logger, 'warning') as mock_warning:
      llm_logger.log_streaming_detection(
        blocked_params=blocked_params,
        provider="openai",
        model="gpt-4",
      )
      
      mock_warning.assert_called_once()
      call_args = mock_warning.call_args
      assert call_args[0][0] == "Streaming parameters detected and blocked"
      assert call_args[1]["event_type"] == "streaming_blocked"
      assert call_args[1]["blocked_params"] == blocked_params
      assert call_args[1]["blocked_count"] == 2
      assert call_args[1]["provider"] == "openai"
  
  def test_log_validation_error(self, llm_logger):
    """Test response validation error logging."""
    field_errors = {
      "choices": "Missing choices array",
      "usage": "Invalid usage format",
    }
    
    with patch.object(llm_logger, 'error') as mock_error:
      llm_logger.log_validation_error(
        error_type="response_structure",
        field_errors=field_errors,
        provider="openai",
        model="gpt-4",
      )
      
      mock_error.assert_called_once()
      call_args = mock_error.call_args
      assert call_args[0][0] == "Response validation failed"
      assert call_args[1]["event_type"] == "validation_error"
      assert call_args[1]["error_type"] == "response_structure"
      assert call_args[1]["field_error_count"] == 2
      assert call_args[1]["failed_fields"] == ["choices", "usage"]
  
  def test_log_queue_status(self, llm_logger):
    """Test provider queue status logging."""
    with patch.object(llm_logger, 'debug') as mock_debug:
      llm_logger.log_queue_status(
        provider="openai",
        queue_size=5,
        active_requests=2,
        total_processed=100,
      )
      
      mock_debug.assert_called_once()
      call_args = mock_debug.call_args
      assert call_args[0][0] == "Provider queue status"
      assert call_args[1]["event_type"] == "queue_status"
      assert call_args[1]["provider"] == "openai"
      assert call_args[1]["queue_size"] == 5
      assert call_args[1]["active_requests"] == 2
  
  @patch('logging.getLogger')
  def test_log_request_debug_mode_includes_data(self, mock_get_logger, llm_logger):
    """Test that request data is included in debug mode."""
    mock_logger = Mock()
    mock_logger.isEnabledFor.return_value = True
    llm_logger.logger = mock_logger
    
    request_data = {"model": "gpt-4", "api_key": "secret"}
    
    with patch.object(llm_logger, 'info') as mock_info:
      llm_logger.log_request(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        model="gpt-4",
        request_data=request_data,
      )
      
      call_args = mock_info.call_args[1]
      assert "request_data" in call_args
  
  @patch('logging.getLogger')
  def test_log_request_non_debug_mode_excludes_data(self, mock_get_logger, llm_logger):
    """Test that request data is excluded in non-debug mode."""
    mock_logger = Mock()
    mock_logger.isEnabledFor.return_value = False
    llm_logger.logger = mock_logger
    
    request_data = {"model": "gpt-4", "api_key": "secret"}
    
    with patch.object(llm_logger, 'info') as mock_info:
      llm_logger.log_request(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        model="gpt-4",
        request_data=request_data,
      )
      
      call_args = mock_info.call_args[1]
      assert "request_data" not in call_args


class TestLoggingUtilities:
  """Test logging utility functions."""
  
  def test_get_llm_logger(self):
    """Test getting LLM logger instance."""
    logger = get_llm_logger("test_module", "openai")
    
    assert isinstance(logger, LLMLogger)
    assert logger.context["provider"] == "openai"
  
  def test_get_llm_logger_without_provider(self):
    """Test getting LLM logger without provider context."""
    logger = get_llm_logger("test_module")
    
    assert isinstance(logger, LLMLogger)
    assert "provider" not in logger.context
  
  @patch('logging.getLogger')
  def test_is_debug_enabled_true(self, mock_get_logger):
    """Test debug enabled detection when debug is enabled."""
    mock_logger = Mock()
    mock_logger.isEnabledFor.return_value = True
    mock_get_logger.return_value = mock_logger
    
    assert is_debug_enabled() is True
    mock_logger.isEnabledFor.assert_called_once_with(logging.DEBUG)
  
  @patch('logging.getLogger')
  def test_is_debug_enabled_false(self, mock_get_logger):
    """Test debug enabled detection when debug is disabled."""
    mock_logger = Mock()
    mock_logger.isEnabledFor.return_value = False
    mock_get_logger.return_value = mock_logger
    
    assert is_debug_enabled() is False
    mock_logger.isEnabledFor.assert_called_once_with(logging.DEBUG)
  
  @patch('rationale_benchmark.llm.logging.get_llm_logger')
  @patch('platform.platform')
  @patch('platform.architecture')
  @patch('platform.processor')
  def test_log_system_info(self, mock_processor, mock_architecture, mock_platform, mock_get_logger):
    """Test system information logging."""
    mock_logger = Mock()
    mock_get_logger.return_value = mock_logger
    mock_platform.return_value = "Linux-5.4.0"
    mock_architecture.return_value = ("64bit", "ELF")
    mock_processor.return_value = "x86_64"
    
    log_system_info()
    
    mock_get_logger.assert_called_once_with("rationale_benchmark.llm.system")
    mock_logger.info.assert_called_once()
    
    call_args = mock_logger.info.call_args
    assert call_args[0][0] == "System information"
    assert "python_version" in call_args[1]
    assert "platform" in call_args[1]
    assert "architecture" in call_args[1]
    assert "processor" in call_args[1]


class TestSensitiveDataProtection:
  """Test sensitive data protection in logging."""
  
  def setup_method(self):
    """Setup for each test."""
    configure_logging(debug_mode=True)
  
  def test_sensitive_data_never_logged(self):
    """Test that sensitive data is never logged even in debug mode."""
    logger = get_llm_logger("test", "openai")
    
    # Capture log output
    with patch('structlog.get_logger') as mock_get_logger:
      mock_structlog_logger = Mock()
      mock_get_logger.return_value = mock_structlog_logger
      
      logger = LLMLogger("test", "openai")
      logger.logger = mock_structlog_logger
      
      # Log with sensitive data
      logger.info("Test message", api_key="sk-secret123", normal_field="safe")
      
      # Check that the logged data was filtered
      call_args = mock_structlog_logger.info.call_args[1]
      assert "api_key" not in str(call_args) or "sk-secret123" not in str(call_args)
  
  def test_nested_sensitive_data_protection(self):
    """Test protection of nested sensitive data."""
    data = {
      "request": {
        "headers": {
          "authorization": "Bearer secret-token",
          "content-type": "application/json",
        },
        "body": {
          "api_key": "sk-1234567890abcdef",
          "model": "gpt-4",
        },
      },
    }
    
    filtered = SensitiveDataFilter.filter_sensitive_data(data)
    
    # Check that nested sensitive data is filtered
    assert "secret-token" not in str(filtered)
    assert "sk-1234567890abcdef" not in str(filtered)
    assert filtered["request"]["body"]["model"] == "gpt-4"  # Safe data preserved
  
  def test_sensitive_data_in_different_formats(self):
    """Test sensitive data protection in different data formats."""
    test_cases = [
      {"X-API-Key": "secret123"},  # Header format
      {"api_token": "token_abc123"},  # Token format
      {"bearer_token": "bearer_xyz789"},  # Bearer format
      {"openai_api_key": "sk-openai123"},  # Provider-specific
      {"anthropic_api_key": "ant-key456"},  # Provider-specific
    ]
    
    for case in test_cases:
      filtered = SensitiveDataFilter.filter_sensitive_data(case)
      
      # Ensure original sensitive values are not present
      for key, value in case.items():
        assert value not in str(filtered), f"Sensitive value '{value}' found in filtered data"


class TestLoggingContextManagement:
  """Test logging context management and request tracing."""
  
  def test_request_context_preservation(self):
    """Test that request context is preserved across log calls."""
    logger = get_llm_logger("test", "openai")
    bound_logger = logger.bind(request_id="req_123", model="gpt-4")
    
    with patch.object(bound_logger.logger, 'info') as mock_info:
      bound_logger.info("First message")
      bound_logger.info("Second message")
      
      # Both calls should have the same context
      assert mock_info.call_count == 2
      # Context should be bound to the logger, not passed as kwargs
  
  def test_logger_context_isolation(self):
    """Test that different logger instances have isolated contexts."""
    logger1 = get_llm_logger("test1", "openai").bind(request_id="req_1")
    logger2 = get_llm_logger("test2", "anthropic").bind(request_id="req_2")
    
    assert logger1.context["provider"] == "openai"
    assert logger1.context["request_id"] == "req_1"
    
    assert logger2.context["provider"] == "anthropic"
    assert logger2.context["request_id"] == "req_2"
    
    # Contexts should be independent
    assert logger1.context != logger2.context
  
  def test_context_inheritance_in_bound_loggers(self):
    """Test that bound loggers inherit and extend context properly."""
    base_logger = get_llm_logger("test", "openai")
    level1_logger = base_logger.bind(request_id="req_123")
    level2_logger = level1_logger.bind(model="gpt-4", temperature=0.7)
    
    # Base logger should have minimal context
    assert base_logger.context == {"provider": "openai"}
    
    # Level 1 should inherit and add
    assert level1_logger.context["provider"] == "openai"
    assert level1_logger.context["request_id"] == "req_123"
    
    # Level 2 should inherit all previous context
    assert level2_logger.context["provider"] == "openai"
    assert level2_logger.context["request_id"] == "req_123"
    assert level2_logger.context["model"] == "gpt-4"
    assert level2_logger.context["temperature"] == 0.7


class TestLoggingPerformance:
  """Test logging performance and efficiency."""
  
  def test_sensitive_data_filtering_performance(self):
    """Test that sensitive data filtering doesn't significantly impact performance."""
    # Create a large data structure with mixed sensitive and safe data
    large_data = {}
    for i in range(1000):
      if i % 10 == 0:
        large_data[f"api_key_{i}"] = f"sk-secret{i:04d}"
      else:
        large_data[f"field_{i}"] = f"value_{i}"
    
    # Time the filtering operation
    import time
    start_time = time.time()
    filtered = SensitiveDataFilter.filter_sensitive_data(large_data)
    end_time = time.time()
    
    # Should complete reasonably quickly (less than 1 second for 1000 items)
    assert end_time - start_time < 1.0
    
    # Verify filtering worked
    sensitive_count = sum(1 for key in filtered.keys() if "api_key" in key)
    assert sensitive_count == 100  # Every 10th item
    
    # Verify no sensitive values leaked
    filtered_str = str(filtered)
    for i in range(0, 1000, 10):
      assert f"sk-secret{i:04d}" not in filtered_str
  
  @patch('rationale_benchmark.llm.logging.SensitiveDataFilter.filter_sensitive_data')
  def test_logging_calls_filter_once_per_call(self, mock_filter):
    """Test that sensitive data filtering is called exactly once per log call."""
    mock_filter.return_value = {"safe": "data"}
    
    logger = get_llm_logger("test", "openai")
    
    with patch.object(logger.logger, 'info'):
      logger.info("Test message", field1="value1", field2="value2")
      
      # Filter should be called exactly once
      mock_filter.assert_called_once_with({"field1": "value1", "field2": "value2"})
  
  def test_debug_mode_conditional_data_inclusion(self):
    """Test that expensive data is only included in debug mode."""
    logger = get_llm_logger("test", "openai")
    
    large_response_data = {"data": "x" * 10000}  # Large data
    
    # Mock logger to control debug mode
    with patch.object(logger.logger, 'isEnabledFor') as mock_is_enabled:
      with patch.object(logger, 'info') as mock_info:
        # Test non-debug mode
        mock_is_enabled.return_value = False
        logger.log_response(200, "gpt-4", 1000, large_response_data)
        
        call_args = mock_info.call_args[1]
        assert "response_data" not in call_args
        
        # Test debug mode
        mock_is_enabled.return_value = True
        logger.log_response(200, "gpt-4", 1000, large_response_data)
        
        call_args = mock_info.call_args[1]
        assert "response_data" in call_args