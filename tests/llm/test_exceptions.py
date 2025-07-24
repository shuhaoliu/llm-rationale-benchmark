"""Unit tests for LLM custom exception classes."""

import pytest

from rationale_benchmark.llm.exceptions import (
  AuthenticationError,
  ConfigurationError,
  ConversationHistoryError,
  LLMConnectorError,
  LLMError,
  ModelNotFoundError,
  NetworkError,
  ProviderError,
  RateLimitError,
  ResponseValidationError,
  StreamingNotSupportedError,
  TimeoutError,
)


class TestLLMError:
  """Test cases for LLMError base exception class."""

  def test_llm_error_creation_with_message_only(self):
    """Test that LLMError can be created with message only."""
    # Arrange & Act
    error = LLMError("Test error message")

    # Assert
    assert str(error) == "Test error message"
    assert error.cause is None

  def test_llm_error_creation_with_message_and_cause(self):
    """Test that LLMError can be created with message and cause."""
    # Arrange
    original_error = ValueError("Original error")

    # Act
    error = LLMError("Test error message", cause=original_error)

    # Assert
    assert str(error) == "Test error message"
    assert error.cause is original_error

  def test_llm_error_inheritance(self):
    """Test that LLMError inherits from Exception."""
    # Arrange & Act
    error = LLMError("Test error")

    # Assert
    assert isinstance(error, Exception)
    assert isinstance(error, LLMError)

  def test_llm_error_with_empty_message(self):
    """Test that LLMError can be created with empty message."""
    # Arrange & Act
    error = LLMError("")

    # Assert
    assert str(error) == ""
    assert error.cause is None

  def test_llm_error_cause_attribute_access(self):
    """Test that cause attribute can be accessed and modified."""
    # Arrange
    error = LLMError("Test error")
    original_error = RuntimeError("Runtime error")

    # Act
    error.cause = original_error

    # Assert
    assert error.cause is original_error


class TestLLMConnectorError:
  """Test cases for LLMConnectorError exception class."""

  def test_llm_connector_error_creation_with_message_only(self):
    """Test that LLMConnectorError can be created with message only."""
    # Arrange & Act
    error = LLMConnectorError("Connector error message")

    # Assert
    assert str(error) == "Connector error message"
    assert error.cause is None

  def test_llm_connector_error_creation_with_message_and_cause(self):
    """Test that LLMConnectorError can be created with message and cause."""
    # Arrange
    original_error = ConnectionError("Connection failed")

    # Act
    error = LLMConnectorError("Connector error", cause=original_error)

    # Assert
    assert str(error) == "Connector error"
    assert error.cause is original_error

  def test_llm_connector_error_inheritance(self):
    """Test that LLMConnectorError inherits from LLMError."""
    # Arrange & Act
    error = LLMConnectorError("Test error")

    # Assert
    assert isinstance(error, LLMError)
    assert isinstance(error, LLMConnectorError)
    assert isinstance(error, Exception)


class TestConfigurationError:
  """Test cases for ConfigurationError exception class."""

  def test_configuration_error_creation_with_message_only(self):
    """Test that ConfigurationError can be created with message only."""
    # Arrange & Act
    error = ConfigurationError("Configuration is invalid")

    # Assert
    assert str(error) == "Configuration is invalid"
    assert error.config_file is None
    assert error.field is None

  def test_configuration_error_creation_with_all_parameters(self):
    """Test that ConfigurationError can be created with all parameters."""
    # Arrange & Act
    error = ConfigurationError(
      "Invalid configuration",
      config_file="config.yaml",
      field="providers.openai.api_key"
    )

    # Assert
    assert str(error) == "Invalid configuration"
    assert error.config_file == "config.yaml"
    assert error.field == "providers.openai.api_key"

  def test_configuration_error_creation_with_config_file_only(self):
    """Test that ConfigurationError can be created with config_file only."""
    # Arrange & Act
    error = ConfigurationError("Config error", config_file="test.yaml")

    # Assert
    assert str(error) == "Config error"
    assert error.config_file == "test.yaml"
    assert error.field is None

  def test_configuration_error_creation_with_field_only(self):
    """Test that ConfigurationError can be created with field only."""
    # Arrange & Act
    error = ConfigurationError("Field error", field="timeout")

    # Assert
    assert str(error) == "Field error"
    assert error.config_file is None
    assert error.field == "timeout"

  def test_configuration_error_inheritance(self):
    """Test that ConfigurationError inherits from LLMError."""
    # Arrange & Act
    error = ConfigurationError("Test error")

    # Assert
    assert isinstance(error, LLMError)
    assert isinstance(error, ConfigurationError)
    assert isinstance(error, Exception)

  def test_configuration_error_attributes_can_be_modified(self):
    """Test that ConfigurationError attributes can be modified after creation."""
    # Arrange
    error = ConfigurationError("Test error")

    # Act
    error.config_file = "modified.yaml"
    error.field = "modified.field"

    # Assert
    assert error.config_file == "modified.yaml"
    assert error.field == "modified.field"


class TestProviderError:
  """Test cases for ProviderError exception class."""

  def test_provider_error_creation_with_provider_and_message(self):
    """Test that ProviderError can be created with provider and message."""
    # Arrange & Act
    error = ProviderError("openai", "API request failed")

    # Assert
    assert str(error) == "[openai] API request failed"
    assert error.provider == "openai"
    assert error.cause is None

  def test_provider_error_creation_with_all_parameters(self):
    """Test that ProviderError can be created with all parameters."""
    # Arrange
    original_error = ConnectionError("Network timeout")

    # Act
    error = ProviderError("anthropic", "Request failed", cause=original_error)

    # Assert
    assert str(error) == "[anthropic] Request failed"
    assert error.provider == "anthropic"
    assert error.cause is original_error

  def test_provider_error_with_empty_provider(self):
    """Test that ProviderError handles empty provider name."""
    # Arrange & Act
    error = ProviderError("", "Error message")

    # Assert
    assert str(error) == "[] Error message"
    assert error.provider == ""

  def test_provider_error_with_empty_message(self):
    """Test that ProviderError handles empty message."""
    # Arrange & Act
    error = ProviderError("openai", "")

    # Assert
    assert str(error) == "[openai] "
    assert error.provider == "openai"

  def test_provider_error_inheritance(self):
    """Test that ProviderError inherits from LLMError."""
    # Arrange & Act
    error = ProviderError("test", "Test error")

    # Assert
    assert isinstance(error, LLMError)
    assert isinstance(error, ProviderError)
    assert isinstance(error, Exception)

  def test_provider_error_provider_attribute_access(self):
    """Test that provider attribute can be accessed and modified."""
    # Arrange
    error = ProviderError("openai", "Test error")

    # Act
    error.provider = "anthropic"

    # Assert
    assert error.provider == "anthropic"
    # Note: The string representation doesn't change automatically
    assert str(error) == "[openai] Test error"


class TestResponseValidationError:
  """Test cases for ResponseValidationError exception class."""

  def test_response_validation_error_creation_with_message_only(self):
    """Test that ResponseValidationError can be created with message only."""
    # Arrange & Act
    error = ResponseValidationError("Response validation failed")

    # Assert
    assert str(error) == "Response validation failed"
    assert error.provider is None
    assert error.response_data is None

  def test_response_validation_error_creation_with_all_parameters(self):
    """Test that ResponseValidationError can be created with all parameters."""
    # Arrange
    response_data = {"choices": [], "model": "gpt-4"}

    # Act
    error = ResponseValidationError(
      "Missing required field",
      provider="openai",
      response_data=response_data
    )

    # Assert
    assert str(error) == "Missing required field"
    assert error.provider == "openai"
    assert error.response_data == response_data

  def test_response_validation_error_with_provider_only(self):
    """Test that ResponseValidationError can be created with provider only."""
    # Arrange & Act
    error = ResponseValidationError("Validation error", provider="anthropic")

    # Assert
    assert str(error) == "Validation error"
    assert error.provider == "anthropic"
    assert error.response_data is None

  def test_response_validation_error_with_response_data_only(self):
    """Test that ResponseValidationError can be created with response_data only."""
    # Arrange
    response_data = {"error": "invalid_format"}

    # Act
    error = ResponseValidationError("Invalid format", response_data=response_data)

    # Assert
    assert str(error) == "Invalid format"
    assert error.provider is None
    assert error.response_data == response_data

  def test_response_validation_error_inheritance(self):
    """Test that ResponseValidationError inherits from LLMError."""
    # Arrange & Act
    error = ResponseValidationError("Test error")

    # Assert
    assert isinstance(error, LLMError)
    assert isinstance(error, ResponseValidationError)
    assert isinstance(error, Exception)

  def test_response_validation_error_attributes_can_be_modified(self):
    """Test that ResponseValidationError attributes can be modified after creation."""
    # Arrange
    error = ResponseValidationError("Test error")
    new_data = {"new": "data"}

    # Act
    error.provider = "gemini"
    error.response_data = new_data

    # Assert
    assert error.provider == "gemini"
    assert error.response_data == new_data


class TestStreamingNotSupportedError:
  """Test cases for StreamingNotSupportedError exception class."""

  def test_streaming_not_supported_error_creation_with_message_only(self):
    """Test that StreamingNotSupportedError can be created with message only."""
    # Arrange & Act
    error = StreamingNotSupportedError("Streaming is not supported")

    # Assert
    assert str(error) == "Streaming is not supported"
    assert error.blocked_params == []

  def test_streaming_not_supported_error_creation_with_blocked_params(self):
    """Test that StreamingNotSupportedError can be created with blocked_params."""
    # Arrange
    blocked_params = ["stream", "streaming", "stream_options"]

    # Act
    error = StreamingNotSupportedError(
      "Streaming parameters detected",
      blocked_params=blocked_params
    )

    # Assert
    assert str(error) == "Streaming parameters detected"
    assert error.blocked_params == blocked_params

  def test_streaming_not_supported_error_with_empty_blocked_params(self):
    """Test that StreamingNotSupportedError handles empty blocked_params list."""
    # Arrange & Act
    error = StreamingNotSupportedError("No streaming", blocked_params=[])

    # Assert
    assert str(error) == "No streaming"
    assert error.blocked_params == []

  def test_streaming_not_supported_error_with_none_blocked_params(self):
    """Test that StreamingNotSupportedError handles None blocked_params."""
    # Arrange & Act
    error = StreamingNotSupportedError("No streaming", blocked_params=None)

    # Assert
    assert str(error) == "No streaming"
    assert error.blocked_params == []

  def test_streaming_not_supported_error_inheritance(self):
    """Test that StreamingNotSupportedError inherits from LLMError."""
    # Arrange & Act
    error = StreamingNotSupportedError("Test error")

    # Assert
    assert isinstance(error, LLMError)
    assert isinstance(error, StreamingNotSupportedError)
    assert isinstance(error, Exception)

  def test_streaming_not_supported_error_blocked_params_can_be_modified(self):
    """Test that blocked_params attribute can be modified after creation."""
    # Arrange
    error = StreamingNotSupportedError("Test error")
    new_params = ["new_stream", "new_streaming"]

    # Act
    error.blocked_params = new_params

    # Assert
    assert error.blocked_params == new_params


class TestAuthenticationError:
  """Test cases for AuthenticationError exception class."""

  def test_authentication_error_creation_with_provider_only(self):
    """Test that AuthenticationError can be created with provider only."""
    # Arrange & Act
    error = AuthenticationError("openai")

    # Assert
    assert str(error) == "[openai] Authentication failed"
    assert error.provider == "openai"

  def test_authentication_error_creation_with_custom_message(self):
    """Test that AuthenticationError can be created with custom message."""
    # Arrange & Act
    error = AuthenticationError("anthropic", "Invalid API key")

    # Assert
    assert str(error) == "[anthropic] Invalid API key"
    assert error.provider == "anthropic"

  def test_authentication_error_inheritance(self):
    """Test that AuthenticationError inherits from ProviderError."""
    # Arrange & Act
    error = AuthenticationError("test")

    # Assert
    assert isinstance(error, ProviderError)
    assert isinstance(error, LLMError)
    assert isinstance(error, AuthenticationError)
    assert isinstance(error, Exception)

  def test_authentication_error_with_empty_provider(self):
    """Test that AuthenticationError handles empty provider."""
    # Arrange & Act
    error = AuthenticationError("")

    # Assert
    assert str(error) == "[] Authentication failed"
    assert error.provider == ""

  def test_authentication_error_with_empty_message(self):
    """Test that AuthenticationError handles empty message."""
    # Arrange & Act
    error = AuthenticationError("openai", "")

    # Assert
    assert str(error) == "[openai] "
    assert error.provider == "openai"


class TestRateLimitError:
  """Test cases for RateLimitError exception class."""

  def test_rate_limit_error_creation_with_provider_only(self):
    """Test that RateLimitError can be created with provider only."""
    # Arrange & Act
    error = RateLimitError("openai")

    # Assert
    assert str(error) == "[openai] Rate limit exceeded"
    assert error.provider == "openai"
    assert error.retry_after is None

  def test_rate_limit_error_creation_with_custom_message(self):
    """Test that RateLimitError can be created with custom message."""
    # Arrange & Act
    error = RateLimitError("anthropic", "Too many requests")

    # Assert
    assert str(error) == "[anthropic] Too many requests"
    assert error.provider == "anthropic"
    assert error.retry_after is None

  def test_rate_limit_error_creation_with_retry_after(self):
    """Test that RateLimitError can be created with retry_after."""
    # Arrange & Act
    error = RateLimitError("openai", retry_after=60)

    # Assert
    assert str(error) == "[openai] Rate limit exceeded"
    assert error.provider == "openai"
    assert error.retry_after == 60

  def test_rate_limit_error_creation_with_all_parameters(self):
    """Test that RateLimitError can be created with all parameters."""
    # Arrange & Act
    error = RateLimitError("gemini", "Quota exceeded", retry_after=120)

    # Assert
    assert str(error) == "[gemini] Quota exceeded"
    assert error.provider == "gemini"
    assert error.retry_after == 120

  def test_rate_limit_error_inheritance(self):
    """Test that RateLimitError inherits from ProviderError."""
    # Arrange & Act
    error = RateLimitError("test")

    # Assert
    assert isinstance(error, ProviderError)
    assert isinstance(error, LLMError)
    assert isinstance(error, RateLimitError)
    assert isinstance(error, Exception)

  def test_rate_limit_error_retry_after_can_be_modified(self):
    """Test that retry_after attribute can be modified after creation."""
    # Arrange
    error = RateLimitError("openai")

    # Act
    error.retry_after = 300

    # Assert
    assert error.retry_after == 300

  def test_rate_limit_error_with_zero_retry_after(self):
    """Test that RateLimitError handles zero retry_after."""
    # Arrange & Act
    error = RateLimitError("openai", retry_after=0)

    # Assert
    assert error.retry_after == 0

  def test_rate_limit_error_with_negative_retry_after(self):
    """Test that RateLimitError handles negative retry_after."""
    # Arrange & Act
    error = RateLimitError("openai", retry_after=-1)

    # Assert
    assert error.retry_after == -1


class TestModelNotFoundError:
  """Test cases for ModelNotFoundError exception class."""

  def test_model_not_found_error_creation(self):
    """Test that ModelNotFoundError can be created with provider and model."""
    # Arrange & Act
    error = ModelNotFoundError("openai", "gpt-5")

    # Assert
    assert str(error) == "[openai] Model 'gpt-5' not found or not available"
    assert error.provider == "openai"
    assert error.model == "gpt-5"

  def test_model_not_found_error_with_empty_provider(self):
    """Test that ModelNotFoundError handles empty provider."""
    # Arrange & Act
    error = ModelNotFoundError("", "test-model")

    # Assert
    assert str(error) == "[] Model 'test-model' not found or not available"
    assert error.provider == ""
    assert error.model == "test-model"

  def test_model_not_found_error_with_empty_model(self):
    """Test that ModelNotFoundError handles empty model."""
    # Arrange & Act
    error = ModelNotFoundError("openai", "")

    # Assert
    assert str(error) == "[openai] Model '' not found or not available"
    assert error.provider == "openai"
    assert error.model == ""

  def test_model_not_found_error_inheritance(self):
    """Test that ModelNotFoundError inherits from ProviderError."""
    # Arrange & Act
    error = ModelNotFoundError("test", "test-model")

    # Assert
    assert isinstance(error, ProviderError)
    assert isinstance(error, LLMError)
    assert isinstance(error, ModelNotFoundError)
    assert isinstance(error, Exception)

  def test_model_not_found_error_model_attribute_can_be_modified(self):
    """Test that model attribute can be modified after creation."""
    # Arrange
    error = ModelNotFoundError("openai", "gpt-4")

    # Act
    error.model = "gpt-3.5-turbo"

    # Assert
    assert error.model == "gpt-3.5-turbo"
    # Note: The string representation doesn't change automatically
    assert str(error) == "[openai] Model 'gpt-4' not found or not available"


class TestNetworkError:
  """Test cases for NetworkError exception class."""

  def test_network_error_creation_with_message_only(self):
    """Test that NetworkError can be created with message only."""
    # Arrange & Act
    error = NetworkError("Connection timeout")

    # Assert
    assert str(error) == "Network error: Connection timeout"
    assert error.cause is None

  def test_network_error_creation_with_message_and_cause(self):
    """Test that NetworkError can be created with message and cause."""
    # Arrange
    original_error = ConnectionError("DNS resolution failed")

    # Act
    error = NetworkError("Unable to connect", cause=original_error)

    # Assert
    assert str(error) == "Network error: Unable to connect"
    assert error.cause is original_error

  def test_network_error_with_empty_message(self):
    """Test that NetworkError handles empty message."""
    # Arrange & Act
    error = NetworkError("")

    # Assert
    assert str(error) == "Network error: "

  def test_network_error_inheritance(self):
    """Test that NetworkError inherits from LLMError."""
    # Arrange & Act
    error = NetworkError("Test error")

    # Assert
    assert isinstance(error, LLMError)
    assert isinstance(error, NetworkError)
    assert isinstance(error, Exception)

  def test_network_error_cause_can_be_modified(self):
    """Test that cause attribute can be modified after creation."""
    # Arrange
    error = NetworkError("Test error")
    new_cause = TimeoutError("Request timeout")

    # Act
    error.cause = new_cause

    # Assert
    assert error.cause is new_cause


class TestConversationHistoryError:
  """Test cases for ConversationHistoryError exception class."""

  def test_conversation_history_error_creation_with_message_only(self):
    """Test that ConversationHistoryError can be created with message only."""
    # Arrange & Act
    error = ConversationHistoryError("Invalid conversation history")

    # Assert
    assert str(error) == "Invalid conversation history"
    assert error.conversation_data is None
    assert error.validation_errors == []
    assert error.message_index is None
    assert error.field_name is None

  def test_conversation_history_error_creation_with_all_parameters(self):
    """Test that ConversationHistoryError can be created with all parameters."""
    # Arrange
    conversation_data = [{"role": "user", "content": "test"}]
    validation_errors = ["Missing 'role' field", "Invalid content type"]

    # Act
    error = ConversationHistoryError(
      "Conversation validation failed",
      conversation_data=conversation_data,
      validation_errors=validation_errors,
      message_index=1,
      field_name="content"
    )

    # Assert
    assert str(error) == "Conversation validation failed"
    assert error.conversation_data == conversation_data
    assert error.validation_errors == validation_errors
    assert error.message_index == 1
    assert error.field_name == "content"

  def test_conversation_history_error_with_conversation_data_only(self):
    """Test that ConversationHistoryError can be created with conversation_data only."""
    # Arrange
    conversation_data = [{"invalid": "data"}]

    # Act
    error = ConversationHistoryError("Invalid data", conversation_data=conversation_data)

    # Assert
    assert str(error) == "Invalid data"
    assert error.conversation_data == conversation_data
    assert error.validation_errors == []
    assert error.message_index is None
    assert error.field_name is None

  def test_conversation_history_error_with_validation_errors_only(self):
    """Test that ConversationHistoryError can be created with validation_errors only."""
    # Arrange
    validation_errors = ["Error 1", "Error 2", "Error 3"]

    # Act
    error = ConversationHistoryError("Multiple errors", validation_errors=validation_errors)

    # Assert
    assert str(error) == "Multiple errors"
    assert error.conversation_data is None
    assert error.validation_errors == validation_errors
    assert error.message_index is None
    assert error.field_name is None

  def test_conversation_history_error_with_message_index_only(self):
    """Test that ConversationHistoryError can be created with message_index only."""
    # Arrange & Act
    error = ConversationHistoryError("Message error", message_index=5)

    # Assert
    assert str(error) == "Message error"
    assert error.conversation_data is None
    assert error.validation_errors == []
    assert error.message_index == 5
    assert error.field_name is None

  def test_conversation_history_error_with_field_name_only(self):
    """Test that ConversationHistoryError can be created with field_name only."""
    # Arrange & Act
    error = ConversationHistoryError("Field error", field_name="role")

    # Assert
    assert str(error) == "Field error"
    assert error.conversation_data is None
    assert error.validation_errors == []
    assert error.message_index is None
    assert error.field_name == "role"

  def test_conversation_history_error_with_empty_validation_errors(self):
    """Test that ConversationHistoryError handles empty validation_errors list."""
    # Arrange & Act
    error = ConversationHistoryError("Test error", validation_errors=[])

    # Assert
    assert error.validation_errors == []

  def test_conversation_history_error_with_none_validation_errors(self):
    """Test that ConversationHistoryError handles None validation_errors."""
    # Arrange & Act
    error = ConversationHistoryError("Test error", validation_errors=None)

    # Assert
    assert error.validation_errors == []

  def test_conversation_history_error_inheritance(self):
    """Test that ConversationHistoryError inherits from LLMConnectorError."""
    # Arrange & Act
    error = ConversationHistoryError("Test error")

    # Assert
    assert isinstance(error, LLMConnectorError)
    assert isinstance(error, LLMError)
    assert isinstance(error, ConversationHistoryError)
    assert isinstance(error, Exception)

  def test_conversation_history_error_attributes_can_be_modified(self):
    """Test that ConversationHistoryError attributes can be modified after creation."""
    # Arrange
    error = ConversationHistoryError("Test error")
    new_data = [{"role": "assistant", "content": "response"}]
    new_errors = ["New error"]

    # Act
    error.conversation_data = new_data
    error.validation_errors = new_errors
    error.message_index = 10
    error.field_name = "new_field"

    # Assert
    assert error.conversation_data == new_data
    assert error.validation_errors == new_errors
    assert error.message_index == 10
    assert error.field_name == "new_field"

  def test_conversation_history_error_with_complex_conversation_data(self):
    """Test ConversationHistoryError with complex conversation data structures."""
    # Arrange
    complex_data = [
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": "Hello", "metadata": {"timestamp": "2024-01-01"}},
      {"role": "assistant", "content": "Hi there!", "tokens": 5}
    ]

    # Act
    error = ConversationHistoryError("Complex data error", conversation_data=complex_data)

    # Assert
    assert error.conversation_data == complex_data
    assert len(error.conversation_data) == 3
    assert error.conversation_data[0]["role"] == "system"
    assert error.conversation_data[1]["metadata"]["timestamp"] == "2024-01-01"
    assert error.conversation_data[2]["tokens"] == 5

  def test_conversation_history_error_with_zero_message_index(self):
    """Test that ConversationHistoryError handles zero message_index."""
    # Arrange & Act
    error = ConversationHistoryError("Index error", message_index=0)

    # Assert
    assert error.message_index == 0

  def test_conversation_history_error_with_negative_message_index(self):
    """Test that ConversationHistoryError handles negative message_index."""
    # Arrange & Act
    error = ConversationHistoryError("Index error", message_index=-1)

    # Assert
    assert error.message_index == -1


class TestTimeoutError:
  """Test cases for TimeoutError exception class."""

  def test_timeout_error_creation_with_default_message(self):
    """Test that TimeoutError can be created with default message."""
    # Arrange & Act
    error = TimeoutError()

    # Assert
    assert str(error) == "Request timed out"
    assert error.timeout_seconds is None

  def test_timeout_error_creation_with_custom_message(self):
    """Test that TimeoutError can be created with custom message."""
    # Arrange & Act
    error = TimeoutError("Connection timeout occurred")

    # Assert
    assert str(error) == "Connection timeout occurred"
    assert error.timeout_seconds is None

  def test_timeout_error_creation_with_timeout_seconds(self):
    """Test that TimeoutError can be created with timeout_seconds."""
    # Arrange & Act
    error = TimeoutError(timeout_seconds=30)

    # Assert
    assert str(error) == "Request timed out"
    assert error.timeout_seconds == 30

  def test_timeout_error_creation_with_all_parameters(self):
    """Test that TimeoutError can be created with all parameters."""
    # Arrange & Act
    error = TimeoutError("Custom timeout message", timeout_seconds=60)

    # Assert
    assert str(error) == "Custom timeout message"
    assert error.timeout_seconds == 60

  def test_timeout_error_with_zero_timeout_seconds(self):
    """Test that TimeoutError handles zero timeout_seconds."""
    # Arrange & Act
    error = TimeoutError(timeout_seconds=0)

    # Assert
    assert error.timeout_seconds == 0

  def test_timeout_error_with_negative_timeout_seconds(self):
    """Test that TimeoutError handles negative timeout_seconds."""
    # Arrange & Act
    error = TimeoutError(timeout_seconds=-1)

    # Assert
    assert error.timeout_seconds == -1

  def test_timeout_error_inheritance(self):
    """Test that TimeoutError inherits from LLMError."""
    # Arrange & Act
    error = TimeoutError()

    # Assert
    assert isinstance(error, LLMError)
    assert isinstance(error, TimeoutError)
    assert isinstance(error, Exception)

  def test_timeout_error_timeout_seconds_can_be_modified(self):
    """Test that timeout_seconds attribute can be modified after creation."""
    # Arrange
    error = TimeoutError()

    # Act
    error.timeout_seconds = 120

    # Assert
    assert error.timeout_seconds == 120


class TestExceptionErrorPropagation:
  """Test cases for exception error propagation patterns."""

  def test_exception_chaining_with_cause(self):
    """Test that exceptions can be properly chained with cause."""
    # Arrange
    original_error = ValueError("Original validation error")
    config_error = ConfigurationError("Config validation failed")
    config_error.cause = original_error

    # Act
    provider_error = ProviderError("openai", "Provider setup failed", cause=config_error)

    # Assert
    assert provider_error.cause is config_error
    assert config_error.cause is original_error
    assert str(provider_error) == "[openai] Provider setup failed"

  def test_exception_propagation_through_inheritance_chain(self):
    """Test that exceptions propagate correctly through inheritance chain."""
    # Arrange & Act
    auth_error = AuthenticationError("openai", "Invalid API key")

    # Assert - Can be caught at any level of the inheritance hierarchy
    with pytest.raises(AuthenticationError):
      raise auth_error

    with pytest.raises(ProviderError):
      raise auth_error

    with pytest.raises(LLMError):
      raise auth_error

    with pytest.raises(Exception):
      raise auth_error

  def test_multiple_exception_types_can_be_caught_together(self):
    """Test that multiple related exception types can be caught together."""
    # Arrange
    errors = [
      AuthenticationError("openai", "Auth failed"),
      RateLimitError("anthropic", "Rate limit"),
      ModelNotFoundError("gemini", "model-x"),
      NetworkError("Connection failed")
    ]

    # Act & Assert - All provider errors can be caught together
    for error in errors[:3]:  # First 3 are ProviderError subclasses
      with pytest.raises(ProviderError):
        raise error

    # All errors can be caught as LLMError
    for error in errors:
      with pytest.raises(LLMError):
        raise error

  def test_exception_context_information_preservation(self):
    """Test that exception context information is preserved during propagation."""
    # Arrange
    conversation_data = [{"role": "user", "content": "test"}]
    validation_errors = ["Missing field", "Invalid type"]

    # Act
    error = ConversationHistoryError(
      "Validation failed",
      conversation_data=conversation_data,
      validation_errors=validation_errors,
      message_index=1,
      field_name="content"
    )

    # Assert - Context information is preserved when caught
    try:
      raise error
    except ConversationHistoryError as caught_error:
      assert caught_error.conversation_data == conversation_data
      assert caught_error.validation_errors == validation_errors
      assert caught_error.message_index == 1
      assert caught_error.field_name == "content"

  def test_exception_cause_chain_traversal(self):
    """Test that exception cause chains can be traversed."""
    # Arrange
    root_cause = ConnectionError("Network failure")
    network_error = NetworkError("Connection timeout", cause=root_cause)
    provider_error = ProviderError("openai", "Request failed", cause=network_error)

    # Act & Assert - Can traverse the cause chain
    assert provider_error.cause is network_error
    assert network_error.cause is root_cause
    assert root_cause.__cause__ is None  # Built-in exception doesn't have our cause attribute

    # Can find root cause by traversing chain
    current_error = provider_error
    while hasattr(current_error, 'cause') and current_error.cause is not None:
      current_error = current_error.cause
    
    assert current_error is root_cause

  def test_exception_handling_with_specific_error_recovery(self):
    """Test exception handling patterns with specific error recovery."""
    # Arrange
    def simulate_provider_operation(should_fail_with):
      """Simulate a provider operation that can fail in different ways."""
      if should_fail_with == "auth":
        raise AuthenticationError("openai", "Invalid API key")
      elif should_fail_with == "rate_limit":
        raise RateLimitError("openai", "Rate exceeded", retry_after=60)
      elif should_fail_with == "model":
        raise ModelNotFoundError("openai", "gpt-5")
      elif should_fail_with == "network":
        raise NetworkError("Connection failed")
      else:
        return "Success"

    # Act & Assert - Different error types can be handled specifically
    
    # Test authentication error handling
    try:
      simulate_provider_operation("auth")
      assert False, "Should have raised AuthenticationError"
    except AuthenticationError as e:
      assert e.provider == "openai"
      assert "Invalid API key" in str(e)

    # Test rate limit error handling with retry information
    try:
      simulate_provider_operation("rate_limit")
      assert False, "Should have raised RateLimitError"
    except RateLimitError as e:
      assert e.provider == "openai"
      assert e.retry_after == 60

    # Test model not found error handling
    try:
      simulate_provider_operation("model")
      assert False, "Should have raised ModelNotFoundError"
    except ModelNotFoundError as e:
      assert e.provider == "openai"
      assert e.model == "gpt-5"

    # Test network error handling
    try:
      simulate_provider_operation("network")
      assert False, "Should have raised NetworkError"
    except NetworkError as e:
      assert "Connection failed" in str(e)

    # Test successful operation
    result = simulate_provider_operation("success")
    assert result == "Success"