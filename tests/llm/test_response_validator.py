"""Unit tests for ResponseValidator class."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from rationale_benchmark.llm.concurrent.validator import ResponseValidator
from rationale_benchmark.llm.models import ModelResponse
from rationale_benchmark.llm.exceptions import ResponseValidationError


@pytest.fixture
def validator():
  """Create a ResponseValidator instance."""
  return ResponseValidator()


@pytest.fixture
def valid_response():
  """Create a valid ModelResponse."""
  return ModelResponse(
    text="This is a valid response text.",
    model="gpt-4",
    provider="openai",
    timestamp=datetime.now(),
    latency_ms=1500,
    token_count=25,
    finish_reason="stop",
    cost_estimate=0.001,
    metadata={"prompt_tokens": 15, "completion_tokens": 10}
  )


@pytest.fixture
def openai_response():
  """Create an OpenAI-style response."""
  return ModelResponse(
    text="OpenAI response text",
    model="gpt-4",
    provider="openai",
    timestamp=datetime.now(),
    latency_ms=1200,
    token_count=30,
    finish_reason="stop",
    metadata={
      "prompt_tokens": 20,
      "completion_tokens": 10,
      "choice_index": 0
    }
  )


@pytest.fixture
def anthropic_response():
  """Create an Anthropic-style response."""
  return ModelResponse(
    text="Claude response text",
    model="claude-3-opus",
    provider="anthropic",
    timestamp=datetime.now(),
    latency_ms=1800,
    token_count=25,
    finish_reason="end_turn",
    metadata={
      "input_tokens": 15,
      "output_tokens": 10,
      "stop_reason": "end_turn"
    }
  )


@pytest.fixture
def gemini_response():
  """Create a Gemini-style response."""
  return ModelResponse(
    text="Gemini response text",
    model="gemini-pro",
    provider="gemini",
    timestamp=datetime.now(),
    latency_ms=1600,
    token_count=28,
    finish_reason="STOP",
    metadata={
      "prompt_token_count": 18,
      "candidates_token_count": 10,
      "total_token_count": 28
    }
  )


class TestResponseValidator:
  """Test cases for ResponseValidator class."""

  def test_validate_response_with_valid_response(self, validator, valid_response):
    """Test that valid responses pass validation without errors."""
    # Should not raise any exception
    validator.validate_response(valid_response)

  def test_validate_response_with_none_raises_error(self, validator):
    """Test that None response raises validation error."""
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(None)
    
    assert "Response cannot be None" in str(exc_info.value)

  def test_validate_response_with_non_modelresponse_raises_error(self, validator):
    """Test that non-ModelResponse objects raise validation error."""
    invalid_response = {"text": "test", "model": "test"}
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(invalid_response)
    
    assert "Response must be a ModelResponse instance" in str(exc_info.value)

  def test_validate_basic_structure_with_empty_text(self, validator, valid_response):
    """Test that empty text fails basic structure validation."""
    valid_response.text = ""
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    assert "Response text cannot be empty" in str(exc_info.value)

  def test_validate_basic_structure_with_whitespace_only_text(self, validator, valid_response):
    """Test that whitespace-only text fails validation."""
    valid_response.text = "   \n\t  "
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    assert "Response text cannot be empty or whitespace only" in str(exc_info.value)

  def test_validate_basic_structure_with_empty_model(self, validator, valid_response):
    """Test that empty model fails basic structure validation."""
    valid_response.model = ""
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    assert "Model name cannot be empty" in str(exc_info.value)

  def test_validate_basic_structure_with_empty_provider(self, validator, valid_response):
    """Test that empty provider fails basic structure validation."""
    valid_response.provider = ""
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    assert "Provider name cannot be empty" in str(exc_info.value)

  def test_validate_basic_structure_with_negative_latency(self, validator, valid_response):
    """Test that negative latency fails validation."""
    valid_response.latency_ms = -100
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    assert "Latency cannot be negative" in str(exc_info.value)

  def test_validate_basic_structure_with_negative_token_count(self, validator, valid_response):
    """Test that negative token count fails validation."""
    valid_response.token_count = -5
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    assert "Token count cannot be negative" in str(exc_info.value)

  def test_validate_basic_structure_with_negative_cost(self, validator, valid_response):
    """Test that negative cost estimate fails validation."""
    valid_response.cost_estimate = -0.001
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    assert "Cost estimate cannot be negative" in str(exc_info.value)

  def test_validate_content_quality_with_very_short_text(self, validator, valid_response):
    """Test content quality validation with very short text."""
    valid_response.text = "Hi"
    
    # Should pass but may log a warning (implementation detail)
    validator.validate_response(valid_response)

  def test_validate_content_quality_with_repetitive_text(self, validator, valid_response):
    """Test content quality validation with repetitive text."""
    valid_response.text = "test " * 100  # Very repetitive
    
    # Should pass but may log a warning (implementation detail)
    validator.validate_response(valid_response)

  def test_validate_metadata_with_invalid_metadata_type(self, validator, valid_response):
    """Test that non-dict metadata fails validation."""
    valid_response.metadata = "invalid metadata"
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    assert "Metadata must be a dictionary" in str(exc_info.value)

  def test_validate_openai_specific_response(self, validator, openai_response):
    """Test OpenAI-specific validation logic."""
    # Should pass validation
    validator.validate_response(openai_response)

  def test_validate_openai_response_missing_prompt_tokens(self, validator, openai_response):
    """Test OpenAI response validation with missing prompt_tokens."""
    del openai_response.metadata["prompt_tokens"]
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(openai_response)
    
    assert "OpenAI response missing required metadata field 'prompt_tokens'" in str(exc_info.value)

  def test_validate_openai_response_missing_completion_tokens(self, validator, openai_response):
    """Test OpenAI response validation with missing completion_tokens."""
    del openai_response.metadata["completion_tokens"]
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(openai_response)
    
    assert "OpenAI response missing required metadata field 'completion_tokens'" in str(exc_info.value)

  def test_validate_openai_response_token_count_mismatch(self, validator, openai_response):
    """Test OpenAI response validation with token count mismatch."""
    openai_response.metadata["prompt_tokens"] = 10
    openai_response.metadata["completion_tokens"] = 15
    openai_response.token_count = 30  # Should be 25
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(openai_response)
    
    assert "OpenAI token count mismatch" in str(exc_info.value)

  def test_validate_anthropic_specific_response(self, validator, anthropic_response):
    """Test Anthropic-specific validation logic."""
    # Should pass validation
    validator.validate_response(anthropic_response)

  def test_validate_anthropic_response_missing_input_tokens(self, validator, anthropic_response):
    """Test Anthropic response validation with missing input_tokens."""
    del anthropic_response.metadata["input_tokens"]
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(anthropic_response)
    
    assert "Anthropic response missing required metadata field 'input_tokens'" in str(exc_info.value)

  def test_validate_anthropic_response_missing_output_tokens(self, validator, anthropic_response):
    """Test Anthropic response validation with missing output_tokens."""
    del anthropic_response.metadata["output_tokens"]
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(anthropic_response)
    
    assert "Anthropic response missing required metadata field 'output_tokens'" in str(exc_info.value)

  def test_validate_gemini_specific_response(self, validator, gemini_response):
    """Test Gemini-specific validation logic."""
    # Should pass validation
    validator.validate_response(gemini_response)

  def test_validate_gemini_response_missing_total_token_count(self, validator, gemini_response):
    """Test Gemini response validation with missing total_token_count."""
    del gemini_response.metadata["total_token_count"]
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(gemini_response)
    
    assert "Gemini response missing required metadata field 'total_token_count'" in str(exc_info.value)

  def test_validate_unknown_provider_response(self, validator, valid_response):
    """Test validation of response from unknown provider."""
    valid_response.provider = "unknown_provider"
    
    # Should pass basic validation but skip provider-specific checks
    validator.validate_response(valid_response)

  def test_validate_response_with_multiple_errors(self, validator, valid_response):
    """Test that validation reports the first error encountered."""
    valid_response.text = ""
    valid_response.model = ""
    valid_response.latency_ms = -100
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    # Should report the first error (empty text)
    assert "Response text cannot be empty" in str(exc_info.value)

  def test_validate_response_with_none_optional_fields(self, validator):
    """Test validation with None values for optional fields."""
    response = ModelResponse(
      text="Valid response text",
      model="test-model",
      provider="test-provider",
      timestamp=datetime.now(),
      latency_ms=1000,
      token_count=None,  # Optional
      finish_reason=None,  # Optional
      cost_estimate=None,  # Optional
      metadata={}
    )
    
    # Should pass validation
    validator.validate_response(response)

  def test_validate_response_logs_validation_success(self, validator, valid_response, caplog):
    """Test that successful validation is logged at debug level."""
    import logging
    caplog.set_level(logging.DEBUG)
    
    validator.validate_response(valid_response)
    
    assert "Response validation passed" in caplog.text
    assert valid_response.provider in caplog.text
    assert valid_response.model in caplog.text

  def test_validate_response_includes_provider_in_error(self, validator, valid_response):
    """Test that validation errors include provider information."""
    valid_response.text = ""
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    assert exc_info.value.provider == valid_response.provider

  def test_validate_response_includes_response_data_in_error(self, validator, valid_response):
    """Test that validation errors include response data for debugging."""
    valid_response.text = ""
    
    with pytest.raises(ResponseValidationError) as exc_info:
      validator.validate_response(valid_response)
    
    assert exc_info.value.response_data == valid_response

  def test_validate_response_with_edge_case_text_lengths(self, validator, valid_response):
    """Test validation with edge case text lengths."""
    # Single character
    valid_response.text = "A"
    validator.validate_response(valid_response)
    
    # Very long text
    valid_response.text = "A" * 10000
    validator.validate_response(valid_response)

  def test_validate_response_with_special_characters(self, validator, valid_response):
    """Test validation with special characters in text."""
    valid_response.text = "Response with émojis 🚀 and special chars: @#$%^&*()"
    validator.validate_response(valid_response)

  def test_validate_response_with_unicode_text(self, validator, valid_response):
    """Test validation with Unicode text."""
    valid_response.text = "Unicode text: 你好世界 🌍 Здравствуй мир"
    validator.validate_response(valid_response)

  def test_validate_response_performance_with_large_metadata(self, validator, valid_response):
    """Test validation performance with large metadata."""
    # Create large metadata dictionary with required fields for OpenAI
    large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
    large_metadata.update({
      "prompt_tokens": 15,
      "completion_tokens": 10
    })
    valid_response.metadata = large_metadata
    
    # Should still validate quickly
    validator.validate_response(valid_response)