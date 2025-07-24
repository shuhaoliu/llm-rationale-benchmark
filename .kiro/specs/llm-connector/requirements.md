# Requirements Document

## Introduction

This document outlines the requirements for implementing a large language model (LLM) connector feature that enables the rationale benchmark tool to connect to multiple LLM providers through a unified interface. The connector will support configuration-driven provider setup, environment-based authentication, parameter validation, and common LLM parameters across different providers including OpenAI, Anthropic, Gemini, and OpenAI-compatible providers like OpenRouter.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to configure multiple LLM providers through a single YAML configuration file, so that I can easily manage different provider settings and specify which configuration to use per CLI command.

#### Acceptance Criteria

1. WHEN the system loads LLM configuration THEN it SHALL read a single YAML file from the `config/llms/` directory
2. WHEN no configuration file is specified THEN the system SHALL use `default-llms.yaml` as the default configuration
3. WHEN a CLI command specifies a configuration file THEN the system SHALL use exactly that configuration file
4. WHEN a configuration file is malformed THEN the system SHALL raise a descriptive validation error
5. WHEN environment variables are referenced in configuration THEN the system SHALL resolve them at runtime
6. IF a provider configuration is missing required fields THEN the system SHALL raise a configuration validation error

### Requirement 2

**User Story:** As a developer, I want to support multiple LLM providers (OpenAI, Anthropic, Gemini, OpenRouter), so that I can benchmark across different model ecosystems.

#### Acceptance Criteria

1. WHEN configuring OpenAI provider THEN the system SHALL support GPT-4, GPT-3.5-turbo, and other OpenAI models
2. WHEN configuring Anthropic provider THEN the system SHALL support Claude models (Opus, Sonnet, Haiku)
3. WHEN configuring Gemini provider THEN the system SHALL support Google's Gemini models
4. WHEN configuring OpenAI-compatible providers THEN the system SHALL support custom base URLs and authentication
5. WHEN a provider is configured THEN the system SHALL validate the provider type against supported providers
6. IF an unsupported provider is specified THEN the system SHALL raise a provider validation error

### Requirement 3

**User Story:** As a developer, I want to validate environment variables for API keys, so that I can ensure proper authentication before making API calls.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL check for required environment variables for each configured provider
2. WHEN an API key environment variable is missing THEN the system SHALL raise a clear authentication error
3. WHEN an API key format is invalid THEN the system SHALL provide a descriptive error message
4. WHEN environment variables are present THEN the system SHALL validate their format without logging sensitive values
5. IF multiple providers are configured THEN the system SHALL validate all required environment variables

### Requirement 4

**User Story:** As a developer, I want to support common LLM parameters like temperature and system prompt, so that I can control model behavior consistently across providers.

#### Acceptance Criteria

1. WHEN configuring a model THEN the system SHALL support temperature parameter (0.0 to 2.0)
2. WHEN configuring a model THEN the system SHALL support max_tokens parameter
3. WHEN configuring a model THEN the system SHALL support system_prompt parameter
4. WHEN configuring a model THEN the system SHALL support timeout and retry parameters
5. WHEN a parameter value is invalid THEN the system SHALL raise a parameter validation error
6. IF provider-specific parameters are needed THEN the system SHALL support them through a provider_specific section

### Requirement 5

**User Story:** As a developer, I want to validate configuration files at startup, so that I can catch configuration errors early before attempting to make API calls.

#### Acceptance Criteria

1. WHEN the application starts THEN it SHALL validate the specified configuration file
2. WHEN configuration validation fails THEN the system SHALL provide specific error messages indicating the file and field
3. WHEN required configuration sections are missing THEN the system SHALL list all missing sections
4. WHEN configuration is loaded THEN the system SHALL validate the complete configuration structure
5. IF configuration is valid THEN the system SHALL proceed with provider initialization

### Requirement 6

**User Story:** As a developer, I want to list available models for each provider, so that I can verify connectivity and see what models are accessible.

#### Acceptance Criteria

1. WHEN requesting available models THEN the system SHALL query each configured provider
2. WHEN a provider is unreachable THEN the system SHALL report the connectivity issue without failing other providers
3. WHEN API authentication fails THEN the system SHALL provide a clear authentication error message
4. WHEN models are successfully retrieved THEN the system SHALL display them grouped by provider
5. IF no models are available for a provider THEN the system SHALL indicate this clearly

### Requirement 7

**User Story:** As a developer, I want to authenticate with all providers using API keys from environment variables, so that I can have a consistent authentication approach across all services.

#### Acceptance Criteria

1. WHEN connecting to any provider THEN the system SHALL use API keys configured in environment variables
2. WHEN an API key environment variable is missing THEN the system SHALL raise a clear authentication error
3. WHEN authentication fails THEN the system SHALL provide a descriptive error message

### Requirement 8

**User Story:** As a developer, I want to support concurrent querying across multiple LLMs with sequential requests per LLM, so that I can efficiently benchmark multiple models while respecting per-provider rate limits and ensuring proper response ordering.

#### Acceptance Criteria

1. WHEN multiple LLMs are configured THEN the system SHALL query them concurrently using separate async tasks with dedicated request queues per provider
2. WHEN making requests to the same LLM provider THEN the system SHALL process them sequentially in a dedicated FIFO queue
3. WHEN a request is in progress for an LLM THEN the system SHALL wait for the complete response before sending the next request to that same LLM
4. WHEN concurrent requests are made to different LLMs THEN the system SHALL process them in parallel without blocking each other
5. WHEN an LLM provider request completes THEN the system SHALL immediately process the next queued request for that provider if available
6. IF an LLM provider fails THEN the system SHALL continue processing requests for other LLMs without interruption
7. WHEN processing multiple requests THEN the system SHALL maintain strict request order within each provider's queue using asyncio.Queue
8. WHEN all concurrent requests complete THEN the system SHALL return responses in the original request order regardless of completion timing
9. WHEN a provider queue is empty THEN the system SHALL keep the queue processing task alive for a configurable timeout period
10. WHEN implementing request queues THEN the system SHALL use asyncio.Future objects to coordinate between request submission and response delivery

### Requirement 9

**User Story:** As a developer, I want structured output validation from all LLM providers, so that I can ensure consistent and reliable response processing with comprehensive validation.

#### Acceptance Criteria

1. WHEN receiving responses from any LLM provider THEN the system SHALL validate the complete response structure before parsing
2. WHEN a response is malformed or incomplete THEN the system SHALL raise a ResponseValidationError with specific field information
3. WHEN parsing provider responses THEN the system SHALL verify all required fields are present and have correct data types
4. WHEN response validation fails THEN the system SHALL provide detailed error information about missing, invalid, or malformed fields
5. IF a provider returns an unexpected response format THEN the system SHALL log the complete response structure and raise a detailed exception
6. WHEN validating OpenAI responses THEN the system SHALL check for choices, model, usage, and object fields with proper nested structure
7. WHEN validating Anthropic responses THEN the system SHALL check for content, model, role, stop_reason, and usage fields
8. WHEN validating any response THEN the system SHALL ensure text content is non-empty and properly formatted
9. WHEN response validation passes THEN the system SHALL log successful validation at debug level
10. WHEN creating ModelResponse objects THEN the system SHALL validate all required fields are populated with correct types

### Requirement 10

**User Story:** As a developer, I want to support configuration discovery and listing, so that I can see what configurations are available and validate my setup.

#### Acceptance Criteria

1. WHEN listing configurations THEN the system SHALL discover all YAML files in the config/llms directory
2. WHEN displaying configurations THEN the system SHALL show available providers and models for each
3. WHEN configuration files have errors THEN the system SHALL indicate which files have issues
4. WHEN no configurations exist THEN the system SHALL provide guidance on creating configuration files
5. IF configuration directory doesn't exist THEN the system SHALL create it with example files

### Requirement 11

**User Story:** As a developer, I want to explicitly disable streaming responses, so that I can ensure consistent response handling and avoid complexity of stream processing.

#### Acceptance Criteria

1. WHEN making requests to any LLM provider THEN the system SHALL never use streaming response modes under any circumstances
2. WHEN providers support streaming THEN the system SHALL explicitly set stream=false or equivalent in request parameters
3. WHEN receiving responses THEN the system SHALL wait for the complete response before processing or returning results
4. WHEN a provider only supports streaming THEN the system SHALL raise a StreamingNotSupportedError with clear guidance
5. WHEN streaming parameters are found in configuration THEN the system SHALL remove them and log a warning message with parameter names
6. WHEN preparing requests THEN the system SHALL filter out all streaming-related parameters from provider_specific sections including stream, streaming, stream_options
7. IF any streaming parameter is detected in a request THEN the system SHALL block it and provide a clear error message
8. WHEN initializing providers THEN the system SHALL validate that no streaming configuration is present
9. WHEN validating requests THEN the system SHALL ensure no streaming parameters are included in any request payload
10. WHEN documenting the system THEN streaming limitations SHALL be clearly stated in all provider documentation

### Requirement 12

**User Story:** As a developer, I want to support conversation history for maintaining context across multiple questions in the same questionnaire, so that I can enable more natural and contextual interactions with LLMs while providing convenient history management.

#### Acceptance Criteria

1. WHEN making a request THEN the system SHALL accept an optional conversation_history parameter as a list of message dictionaries
2. WHEN conversation history is provided THEN the system SHALL include all previous messages in the request to maintain full context
3. WHEN receiving a response THEN the system SHALL return an updated conversation history with the latest user prompt and assistant response appended
4. WHEN conversation history is empty or None THEN the system SHALL treat the request as a new conversation and return history with just the current exchange
5. WHEN conversation history contains system prompts THEN the system SHALL preserve them at the beginning of the conversation across all requests
6. WHEN conversation history becomes too long for the model's context window THEN the system SHALL provide automatic truncation options while preserving system prompts
7. IF conversation history format is invalid THEN the system SHALL raise a ConversationHistoryError with specific validation details
8. WHEN using conversation history THEN the system SHALL maintain proper message ordering with roles (system, user, assistant) and validate role consistency
9. WHEN conversation history is updated THEN the system SHALL ensure thread safety for concurrent requests using the same conversation
10. WHEN persisting conversation history THEN the system SHALL support serialization to and from JSON format for easy file storage
11. WHEN a conversation history list is passed THEN the system SHALL validate each message has required 'role' and 'content' fields
12. WHEN returning updated conversation history THEN the system SHALL include metadata about token usage and truncation if applied
13. WHEN multiple requests use the same conversation history THEN the system SHALL provide options for conversation branching or merging
14. WHEN conversation history exceeds provider limits THEN the system SHALL implement intelligent truncation preserving recent context and system prompts

### Requirement 13

**User Story:** As a developer, I want comprehensive error handling and logging, so that I can troubleshoot configuration and connectivity issues effectively.

#### Acceptance Criteria

1. WHEN errors occur THEN the system SHALL log them with appropriate severity levels
2. WHEN API calls fail THEN the system SHALL log request context without exposing sensitive data
3. WHEN configuration errors occur THEN the system SHALL provide actionable error messages
4. WHEN debugging is enabled THEN the system SHALL provide detailed execution information
5. IF multiple errors occur THEN the system SHALL collect and report all errors together