# LLM Connector Design

## Purpose
- Provide a unified, configuration-driven interface for collecting questionnaire responses from multiple LLM providers.
- Maintain consistent conversation semantics (history tracking, retries, structured outputs) regardless of provider implementation.
- Integrate cleanly with the CLI and benchmark runner defined in `docs/architecture.md`.

## Scope
- Configuration parsing, validation, and environment resolution for connectors stored under `config/llms/`.
- Stateful `LLMConversation` API offering `ask()` with verification/retries and `archive()` for immutable transcripts.
- Provider-agnostic factory that instantiates conversations using a supplied configuration file, selected `provider/model` identifier, and system prompt.
- Provider adapters for OpenAI, OpenAI-compatible, Anthropic, and Gemini endpoints.
- Explicit error handling policy covering validation, retryable transport errors, and non-recoverable failures.

Out of scope: UI integrations, persistence beyond in-memory transcripts, advanced tooling for prompt templating, and dedicated streaming UX (connectors still auto-switch to streaming when providers require it).

## Architectural Overview
- **Configuration Layer**  
  Parses YAML files into a validated mapping of `LLMConnectorConfig` objects keyed by `"{provider}/{model_id}"`, resolving environment variable references and applying defaults for optional parameters (temperature, timeout, retry policy, output mode).
- **Factory Layer**  
  `LLMConversationFactory` consumes a configuration mapping, a `"{provider}/{model_id}"` selector, and an optional system prompt, then produces a ready-to-use `LLMConversation` while wiring the appropriate provider adapter and enforcing response mode (JSON vs. free text).
- **Conversation Layer**  
  `LLMConversation` encapsulates the current transcript, request parameters, retry orchestration, and validation hook invocation.
- **Provider Layer**  
  A `BaseProviderClient` abstract class defines the low-level API for sending messages, handling provider-specific payloads, structured output hints, and response extraction. Concrete adapters satisfy provider-specific expectations.

The CLI resolves config paths, desired `provider/model` selector, and system prompts, then delegates to the factory. Benchmark runners interact exclusively with the `LLMConversation` interface.

## Core Components

### Configuration Model
- Configuration files declare multiple named entries under `models`, each keyed by `"{provider}/{model_id}"` and expanding to an `LLMConnectorConfig`. Shared defaults (e.g., retry policy) can live at the document root and are merged into each entry unless overridden.
- `LLMConnectorConfig` (pydantic model) captures:
  - `provider`: enum (`openai`, `openai_compatible`, `anthropic`, `gemini`).
  - `endpoint`, `api_key`, `model`, `timeout_seconds`, `retry` (max attempts, backoff).
  - `response_format`: enum (`json`, `text`) with `json` as default.
  - Optional fields: `temperature`, `top_p`, `max_tokens`, `metadata`, `provider_specific`.
  - `system_prompt` to seed conversation context; CLI may override.
- Loader resolves YAML via `PyYAML`, expands `${ENV_VAR}` references, validates ranges (e.g., `0.0 <= temperature <= 2.0`) and required keys per provider before returning any configs. Invalid entries (malformed keys, unsupported provider/model tuples, or failing schema constraints) abort the load and raise `ConfigurationError` summarizing all offending settings.

### Factory
- `LLMConversationFactory.create_from_config(path: Path, target_model: str, system_prompt: str | None) -> LLMConversation`.
- Steps:
  1. Read configuration file and run full schema validation across every declared model entry, raising `ConfigurationError` immediately if any setting is invalid.
  2. Resolve `target_model`, which must be provided in `"{provider}/{model_id}"` format and match one of the validated entries; missing or mistyped identifiers raise `ConfigurationError`.
  3. Merge CLI-provided system prompt (falls back to config/system default).
  4. Instantiate provider adapter using `ProviderRegistry`.
  5. Construct `LLMConversation` with initial system message, retry policy, and response mode derived from the selected model.
- Supports caching of provider clients keyed by config hash to avoid redundant HTTP session creation when reusing configs.

### Conversation
- `LLMConversation` maintains:
  - `config`: immutable connector settings.
  - `history`: list of `ConversationTurn` records (`role`, `content`, `timestamp`, optional `verification_errors`).
  - `state`: enum (`active`, `archived`).
- `ask(question: str, validator: Callable[[LLMResponse], bool] | None = None, *, max_attempts: int | None = None) -> LLMResponse`
  - Validates active state; raises `ConversationArchivedError` if archived.
  - Appends user question to history before calling provider.
  - Invokes provider adapter with full history, ensuring structured output instructions when `response_format == json` (e.g., OpenAI `response_format={"type": "json_object"}` or Anthropic tool use).
  - Detects reasoning-mode capabilities from the selected config; when a provider (e.g., Qwen3 reasoning variants) only supports streaming responses, `ask()` forces a streaming invocation and buffers emitted chunks into a single response before returning to the caller.
  - Executes retry loop (default from config, override via `max_attempts`) on transient errors (HTTP 5xx, timeouts) and failed `validator`. Retries add delay via exponential backoff (configurable jitter).
  - Captures each failed validation reason in turn metadata; surfaces `ValidationFailedError` after exhausting attempts without a valid response.
- `archive() -> LLMConversationArchive`
  - Transitions state to archived; subsequent `ask()` calls raise.
  - Returns dataclass with `config_snapshot`, ordered `questions`, `answers`, `timestamps`, and aggregated metadata (e.g., restart counts).
  - Archive object provides `to_dict()` / `to_json()` helpers for persistence.

## Provider Integrations
- `BaseProviderClient` defines async `generate(messages: list[Message], response_format: ResponseFormat) -> LLMResponse`.
- Common behaviors:
  - Normalize messages to provider-specific structure (OpenAI chat, Anthropic Claude messages, Gemini content parts).
  - Translate standardized parameters (`temperature`, `top_p`, `max_tokens`).
  - Respect timeout via aiohttp session with per-request deadlines.
  - Auto-detect when a reasoning-capable model requires streaming, switching to streaming APIs and assembling a buffered response so downstream consumers can keep the non-streaming contract.
  - Surface `RateLimitError`, `AuthenticationError`, `ProviderError`.
- Provider-specific notes:
  - **OpenAI**: uses the official Responses API for non-compatible endpoints, supports function-level JSON output via `response_format={"type": "json_object"}`, and authenticates with bearer headers.
  - **OpenAI-compatible**: same payload shape but configurable base URL and optional extra headers.
  - **Anthropic**: maps system prompt and conversation history to Claude message format; uses headers `x-api-key` and `anthropic-version`.
  - **Gemini**: handles streaming-to-buffer conversion; leverages generative language API; structured output via `jsonSchema` hints when supported.
- Provider registry:
  - `ProviderRegistry.register(provider_type: ProviderType, factory: Callable[[LLMConnectorConfig], BaseProviderClient])`.
  - Enables dependency injection for testing (e.g., mock providers).

## Error Handling and Retries
- Configuration/validation errors (invalid settings or unknown `provider/model` selections) raise immediately and stop factory creation.
- Authentication failures: raise `AuthenticationError` without retries; log redacted provider context.
- Connection issues, timeouts, 5xx responses: retry with exponential backoff (configurable `initial_delay`, `max_delay`, `multiplier`, `jitter`), honoring global `max_attempts`.
- Provider rate limits: treat as retryable if provider returns standard rate-limit status; include `Retry-After` if provided.
- Validator failures: count toward retry budget; include validator message in turn metadata.
- All errors captured via structured logging (`structlog`) with conversation id, provider, attempt number.

## Structured vs. Unstructured Output
- `response_format` controls guidance prompts and downstream parsing.
- JSON mode:
  - Injects explicit instructions in system prompt and provider-specific controls to enforce JSON schema.
  - `validator` receives parsed Python object; parsing errors treated as validation failure.
- Text mode:
  - Returns raw strings; validator operates on text.
  - Archive stores text payload and optionally derived metadata (e.g., parse attempt warnings).

## Conversation Archiving and Persistence
- Archive dataclass includes:
  - `config_snapshot`: sanitized copy of configuration (keys masked).
  - `system_prompt`, `response_format`.
  - `turns`: ordered list with question, answer, timestamps, number of retries.
  - `verification_summary`: counts of retries and validator failures.
- Supports serialization hooks for benchmark runner to emit final JSON artifacts aligning with `docs/architecture.md` expectations.

## Testing Strategy
- Unit tests under `tests/llm/` mirroring module layout.
- Configuration loader tests:
  - Valid/invalid YAML fixtures.
  - Environment variable substitution.
  - Provider-specific required fields.
  - Multi-model documents ensuring valid selection by `"{provider}/{model_id}"` and proper errors for unknown identifiers.
- Conversation tests:
  - Retry logic with mocked provider raising transient errors then succeeding.
  - Validator failure path leading to `ValidationFailedError`.
  - Archive immutability and snapshot content.
  - Streaming fallback buffering when a reasoning model refuses non-streaming responses.
- Provider adapter tests (mocked HTTP):
  - Ensure payload shape per provider.
  - Structured vs. text response handling.
- Integration smoke tests via `pytest-asyncio` with in-memory fake provider to assert transcript behavior.

## Observability and Instrumentation
- Expose debug logging hooks for sent/received payload metadata (without sensitive content).
- Record metrics (future extension) for attempt counts, latency, validator failure rates via pluggable collector interface.

## Future Extensions
- Rich streaming UX for partial outputs (live token display, incremental validators, CLI progress rendering).
- Persistent transcript storage (e.g., writing archives to disk or DB).
- Prompt templating / variable injection at factory layer.
- Multi-turn batch execution and concurrency controls for benchmark runs.
