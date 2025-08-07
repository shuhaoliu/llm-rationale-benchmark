"""Unit tests for response validation error handling."""

import pytest
from unittest.mock import Mock, patch

from rationale_benchmark.llm.exceptions import ResponseValidationError
from rationale_benchmark.llm.validation import (
  ResponseValidator,
  aggregate_validation_errors,
  validate_response_content_quality,
)


class TestResponseValidationError:
  """Test ResponseValidationError with detailed field-level error reporting."""
  
  def test_basic_response_validation_error_creation(self):
    """Test basic ResponseValidationError creation."""
    error = ResponseValidationError(
      message="Validation failed",
      provider="openai",
      response_data={"invalid": "data"},
    )
    
    assert str(error) == "Validation failed"
    assert error.provider == "openai"
    assert error.response_data == {"invalid": "data"}
    assert error.field_errors == {}
    assert error.validation_context == {}
    assert error.recovery_suggestions == []
  
  def test_response_validation_error_with_field_errors(self):
    """Test ResponseValidationError with field-level errors."""
    field_errors = {
      "choices": "Missing required field 'choices'",
      "usage": "Usage field must be a dictionary",
    }
    
    error = ResponseValidationError(
      message="Multiple validation errors",
      provider="openai",
      field_errors=field_errors,
    )
    
    assert error.field_errors == field_errors
    assert error.has_field_errors()
    assert error.get_field_error_count() == 2
  
  def test_response_validation_error_with_context(self):
    """Test ResponseValidationError with validation context."""
    validation_context = {
      "response_type": "dict",
      "response_keys": ["choices", "model"],
      "expected_fields": ["choices", "model", "usage"],
    }
    
    error = ResponseValidationError(
      message="Context validation failed",
      provider="anthropic",
      validation_context=validation_context,
    )
    
    assert error.validation_context == validation_context
  
  def test_response_validation_error_with_recovery_suggestions(self):
    """Test ResponseValidationError with recovery suggestions."""
    recovery_suggestions = [
      "Check API response format",
      "Verify API credentials",
      "Review request parameters",
    ]
    
    error = ResponseValidationError(
      message="Validation failed with suggestions",
      provider="gemini",
      recovery_suggestions=recovery_suggestions,
    )
    
    assert error.recovery_suggestions == recovery_suggestions
  
  def test_add_field_error(self):
    """Test adding field errors to ResponseValidationError."""
    error = ResponseValidationError(
      message="Initial error",
      provider="openai",
    )
    
    error.add_field_error("choices", "Choices array is empty")
    error.add_field_error("model", "Model field is missing")
    
    assert error.field_errors["choices"] == "Choices array is empty"
    assert error.field_errors["model"] == "Model field is missing"
    assert error.get_field_error_count() == 2
  
  def test_add_recovery_suggestion(self):
    """Test adding recovery suggestions to ResponseValidationError."""
    error = ResponseValidationError(
      message="Error with suggestions",
      provider="anthropic",
    )
    
    error.add_recovery_suggestion("Check API documentation")
    error.add_recovery_suggestion("Verify API key permissions")
    
    assert len(error.recovery_suggestions) == 2
    assert "Check API documentation" in error.recovery_suggestions
    assert "Verify API key permissions" in error.recovery_suggestions
  
  def test_get_detailed_message(self):
    """Test getting detailed error message with all context."""
    error = ResponseValidationError(
      message="Comprehensive validation error",
      provider="openai",
      field_errors={
        "choices": "Missing choices array",
        "usage": "Invalid usage format",
      },
      validation_context={
        "response_size": 150,
        "response_keys": ["error"],
      },
      recovery_suggestions=[
        "Check API response format",
        "Verify request parameters",
      ],
    )
    
    detailed_message = error.get_detailed_message()
    
    assert "Comprehensive validation error" in detailed_message
    assert "Provider: openai" in detailed_message
    assert "Field validation errors:" in detailed_message
    assert "choices: Missing choices array" in detailed_message
    assert "usage: Invalid usage format" in detailed_message
    assert "Validation context:" in detailed_message
    assert "response_size: 150" in detailed_message
    assert "Recovery suggestions:" in detailed_message
    assert "1. Check API response format" in detailed_message
    assert "2. Verify request parameters" in detailed_message
  
  def test_get_detailed_message_minimal(self):
    """Test getting detailed message with minimal information."""
    error = ResponseValidationError(message="Simple error")
    
    detailed_message = error.get_detailed_message()
    
    assert detailed_message == "Simple error"
  
  def test_create_aggregated_error(self):
    """Test creating aggregated error from multiple validation errors."""
    error1 = ResponseValidationError(
      message="First error",
      provider="openai",
      field_errors={"choices": "Missing choices"},
      recovery_suggestions=["Check API format"],
      validation_context={"request_id": "req1"},
    )
    
    error2 = ResponseValidationError(
      message="Second error",
      provider="openai",
      field_errors={"usage": "Invalid usage"},
      recovery_suggestions=["Verify credentials", "Check API format"],  # Duplicate
      validation_context={"model": "gpt-4"},
    )
    
    aggregated = ResponseValidationError.create_aggregated_error([error1, error2], "openai")
    
    assert aggregated.provider == "openai"
    assert "Multiple validation errors occurred: 2 errors affecting 2 fields" in aggregated.message
    assert aggregated.field_errors["choices"] == "Missing choices"
    assert aggregated.field_errors["usage"] == "Invalid usage"
    assert len(aggregated.field_errors) == 2
    
    # Recovery suggestions should be deduplicated
    assert len(aggregated.recovery_suggestions) == 2
    assert "Check API format" in aggregated.recovery_suggestions
    assert "Verify credentials" in aggregated.recovery_suggestions
    
    # Context should be merged
    assert aggregated.validation_context["request_id"] == "req1"
    assert aggregated.validation_context["model"] == "gpt-4"
  
  def test_create_aggregated_error_empty_list(self):
    """Test creating aggregated error with empty error list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot create aggregated error from empty error list"):
      ResponseValidationError.create_aggregated_error([])


class TestResponseValidator:
  """Test ResponseValidator comprehensive validation functionality."""
  
  @pytest.fixture
  def openai_validator(self):
    """Create ResponseValidator for OpenAI."""
    return ResponseValidator("openai")
  
  @pytest.fixture
  def anthropic_validator(self):
    """Create ResponseValidator for Anthropic."""
    return ResponseValidator("anthropic")
  
  @pytest.fixture
  def gemini_validator(self):
    """Create ResponseValidator for Gemini."""
    return ResponseValidator("gemini")
  
  def test_validate_response_structure_success(self, openai_validator):
    """Test successful response structure validation."""
    response_data = {
      "choices": [{"message": {"content": "Hello", "role": "assistant"}}],
      "model": "gpt-4",
      "usage": {"total_tokens": 10},
    }
    
    required_fields = ["choices", "model", "usage"]
    
    # Should not raise any exception
    openai_validator.validate_response_structure(response_data, required_fields)
  
  def test_validate_response_structure_missing_fields(self, openai_validator):
    """Test response structure validation with missing fields."""
    response_data = {
      "choices": [{"message": {"content": "Hello", "role": "assistant"}}],
      # Missing "model" and "usage" fields
    }
    
    required_fields = ["choices", "model", "usage"]
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_validator.validate_response_structure(response_data, required_fields)
    
    error = exc_info.value
    assert error.provider == "openai"
    assert "Response validation failed for 2 fields" in error.message
    assert "model" in error.field_errors
    assert "usage" in error.field_errors
    assert "Required field 'model' is missing" in error.field_errors["model"]
    assert "Required field 'usage' is missing" in error.field_errors["usage"]
  
  def test_validate_response_structure_non_dict(self, openai_validator):
    """Test response structure validation with non-dictionary response."""
    response_data = "invalid response"
    required_fields = ["choices", "model"]
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_validator.validate_response_structure(response_data, required_fields)
    
    error = exc_info.value
    assert error.provider == "openai"
    assert "Response must be a dictionary, got str" in error.message
    assert "Check API response format" in error.recovery_suggestions
  
  def test_validate_openai_response_success(self, openai_validator):
    """Test successful OpenAI response validation."""
    response_data = {
      "choices": [{
        "message": {"content": "Hello world", "role": "assistant"},
        "finish_reason": "stop",
        "index": 0,
      }],
      "model": "gpt-4",
      "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
      },
      "object": "chat.completion",
    }
    
    # Should not raise any exception
    openai_validator.validate_openai_response(response_data)
  
  def test_validate_openai_response_invalid_choices(self, openai_validator):
    """Test OpenAI response validation with invalid choices."""
    response_data = {
      "choices": [],  # Empty choices array
      "model": "gpt-4",
      "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
      "object": "chat.completion",
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_validator.validate_openai_response(response_data)
    
    error = exc_info.value
    assert error.provider == "openai"
    assert "choices" in error.field_errors
    assert "must be a non-empty array" in error.field_errors["choices"]
  
  def test_validate_openai_response_invalid_usage(self, openai_validator):
    """Test OpenAI response validation with invalid usage."""
    response_data = {
      "choices": [{
        "message": {"content": "Hello", "role": "assistant"},
        "finish_reason": "stop",
        "index": 0,
      }],
      "model": "gpt-4",
      "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 10,  # Inconsistent total
      },
      "object": "chat.completion",
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      openai_validator.validate_openai_response(response_data)
    
    error = exc_info.value
    assert error.provider == "openai"
    assert "usage" in error.field_errors
    assert "token count inconsistency" in error.field_errors["usage"]
  
  def test_validate_anthropic_response_success(self, anthropic_validator):
    """Test successful Anthropic response validation."""
    response_data = {
      "content": [{"text": "Hello world", "type": "text"}],
      "model": "claude-3-opus-20240229",
      "role": "assistant",
      "stop_reason": "end_turn",
      "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    
    # Should not raise any exception
    anthropic_validator.validate_anthropic_response(response_data)
  
  def test_validate_anthropic_response_invalid_content(self, anthropic_validator):
    """Test Anthropic response validation with invalid content."""
    response_data = {
      "content": [{"text": "", "type": "text"}],  # Empty text
      "model": "claude-3-opus-20240229",
      "role": "assistant",
      "stop_reason": "end_turn",
      "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      anthropic_validator.validate_anthropic_response(response_data)
    
    error = exc_info.value
    assert error.provider == "anthropic"
    assert "content" in error.field_errors
    assert "must be a non-empty string" in error.field_errors["content"]
  
  def test_validate_gemini_response_success(self, gemini_validator):
    """Test successful Gemini response validation."""
    response_data = {
      "candidates": [{
        "content": {
          "parts": [{"text": "Hello world"}],
          "role": "model",
        },
        "finishReason": "STOP",
        "index": 0,
      }],
      "usageMetadata": {
        "promptTokenCount": 5,
        "candidatesTokenCount": 2,
      },
    }
    
    # Should not raise any exception
    gemini_validator.validate_gemini_response(response_data)
  
  def test_validate_gemini_response_invalid_candidates(self, gemini_validator):
    """Test Gemini response validation with invalid candidates."""
    response_data = {
      "candidates": [],  # Empty candidates
      "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
    }
    
    with pytest.raises(ResponseValidationError) as exc_info:
      gemini_validator.validate_gemini_response(response_data)
    
    error = exc_info.value
    assert error.provider == "gemini"
    assert "candidates" in error.field_errors
    assert "must be a non-empty array" in error.field_errors["candidates"]
  
  def test_validate_provider_response_unknown_provider(self):
    """Test validation with unknown provider falls back to generic validation."""
    validator = ResponseValidator("unknown_provider")
    
    response_data = {
      "text": "Hello world",
      "model": "unknown-model",
    }
    
    with patch.object(validator, 'validate_response_structure') as mock_validate:
      validator.validate_provider_response(response_data)
      mock_validate.assert_called_once_with(response_data, ["text", "model"])
  
  def test_create_validation_error_with_context(self, openai_validator):
    """Test creating validation error with comprehensive context."""
    response_data = {"invalid": "response"}
    additional_context = {"request_id": "req_123"}
    
    error = openai_validator.create_validation_error_with_context(
      "Test validation error",
      response_data,
      additional_context,
    )
    
    assert error.message == "Test validation error"
    assert error.provider == "openai"
    assert error.response_data == response_data
    assert error.validation_context["provider"] == "openai"
    assert error.validation_context["request_id"] == "req_123"
    assert error.validation_context["response_size"] == len(str(response_data))
    assert error.validation_context["response_keys"] == ["invalid"]
    
    # Should have common recovery suggestions
    assert len(error.recovery_suggestions) >= 3
    assert any("API documentation" in suggestion for suggestion in error.recovery_suggestions)


class TestValidationErrorAggregation:
  """Test validation error aggregation functionality."""
  
  def test_aggregate_validation_errors_success(self):
    """Test successful aggregation of multiple validation errors."""
    error1 = ResponseValidationError(
      message="First error",
      provider="openai",
      field_errors={"choices": "Missing choices"},
    )
    
    error2 = ResponseValidationError(
      message="Second error",
      provider="openai",
      field_errors={"usage": "Invalid usage"},
    )
    
    aggregated = aggregate_validation_errors([error1, error2], "openai")
    
    assert aggregated.provider == "openai"
    assert "Multiple validation errors occurred" in aggregated.message
    assert len(aggregated.field_errors) == 2
    assert "choices" in aggregated.field_errors
    assert "usage" in aggregated.field_errors
  
  def test_aggregate_validation_errors_empty_list(self):
    """Test aggregation with empty error list raises ValueError."""
    with pytest.raises(ValueError):
      aggregate_validation_errors([])


class TestContentQualityValidation:
  """Test response content quality validation."""
  
  def test_validate_response_content_quality_success(self):
    """Test successful content quality validation."""
    content = "This is a good response with proper content."
    
    issues = validate_response_content_quality(content, min_length=10)
    
    assert issues == []
  
  def test_validate_response_content_quality_too_short(self):
    """Test content quality validation with too short content."""
    content = "Hi"
    
    issues = validate_response_content_quality(content, min_length=10)
    
    assert len(issues) == 1
    assert "Content too short" in issues[0]
    assert "2 characters (minimum: 10)" in issues[0]
  
  def test_validate_response_content_quality_too_long(self):
    """Test content quality validation with too long content."""
    content = "A" * 1000
    
    issues = validate_response_content_quality(content, max_length=500)
    
    assert len(issues) == 1
    assert "Content too long" in issues[0]
    assert "1000 characters (maximum: 500)" in issues[0]
  
  def test_validate_response_content_quality_empty(self):
    """Test content quality validation with empty content."""
    content = "   "  # Only whitespace
    
    issues = validate_response_content_quality(content)
    
    assert len(issues) >= 1
    assert any("empty or contains only whitespace" in issue for issue in issues)
  
  def test_validate_response_content_quality_non_string(self):
    """Test content quality validation with non-string content."""
    content = 123
    
    issues = validate_response_content_quality(content)
    
    assert len(issues) == 1
    assert "Content must be a string, got int" in issues[0]
  
  def test_validate_response_content_quality_contains_errors(self):
    """Test content quality validation with error indicators."""
    content = "This response contains an Error message"
    
    issues = validate_response_content_quality(content)
    
    assert len(issues) == 1
    assert "appears to contain error messages" in issues[0]
  
  def test_validate_response_content_quality_short_without_punctuation(self):
    """Test content quality validation with short content without punctuation."""
    content = "Hello"
    
    issues = validate_response_content_quality(content, min_length=1)
    
    assert len(issues) == 1
    assert "Very short content without proper sentence ending" in issues[0]
  
  def test_validate_response_content_quality_multiple_issues(self):
    """Test content quality validation with multiple issues."""
    content = ""  # Empty content
    
    issues = validate_response_content_quality(content, min_length=5)
    
    # Should have multiple issues
    assert len(issues) >= 2
    issue_text = " ".join(issues)
    assert "too short" in issue_text
    assert "empty or contains only whitespace" in issue_text


class TestValidationErrorContext:
  """Test validation error context and recovery suggestions."""
  
  def test_validation_error_context_preservation(self):
    """Test that validation context is properly preserved."""
    context = {
      "request_id": "req_123",
      "model": "gpt-4",
      "timestamp": "2024-01-01T00:00:00Z",
    }
    
    error = ResponseValidationError(
      message="Context test error",
      provider="openai",
      validation_context=context,
    )
    
    assert error.validation_context == context
    assert error.validation_context["request_id"] == "req_123"
    assert error.validation_context["model"] == "gpt-4"
  
  def test_recovery_suggestions_deduplication(self):
    """Test that recovery suggestions are deduplicated in aggregated errors."""
    error1 = ResponseValidationError(
      message="Error 1",
      recovery_suggestions=["Suggestion A", "Suggestion B"],
    )
    
    error2 = ResponseValidationError(
      message="Error 2",
      recovery_suggestions=["Suggestion B", "Suggestion C"],  # B is duplicate
    )
    
    aggregated = ResponseValidationError.create_aggregated_error([error1, error2])
    
    # Should have 3 unique suggestions
    assert len(aggregated.recovery_suggestions) == 3
    assert "Suggestion A" in aggregated.recovery_suggestions
    assert "Suggestion B" in aggregated.recovery_suggestions
    assert "Suggestion C" in aggregated.recovery_suggestions
  
  def test_field_error_accumulation(self):
    """Test that field errors are properly accumulated in aggregated errors."""
    error1 = ResponseValidationError(
      message="Error 1",
      field_errors={"field1": "Error in field1", "field2": "Error in field2"},
    )
    
    error2 = ResponseValidationError(
      message="Error 2",
      field_errors={"field3": "Error in field3", "field2": "Different error in field2"},
    )
    
    aggregated = ResponseValidationError.create_aggregated_error([error1, error2])
    
    # Should have all field errors, with later ones overriding earlier ones for same field
    assert len(aggregated.field_errors) == 3
    assert aggregated.field_errors["field1"] == "Error in field1"
    assert aggregated.field_errors["field2"] == "Different error in field2"  # Overridden
    assert aggregated.field_errors["field3"] == "Error in field3"