# Data Models and Integration Interfaces

## Questionnaire Data Models
```python
@dataclass
class Question:
  id: str
  type: str  # "scenario", "choice", "open"
  prompt: str
  options: Optional[list[str]] = None
  expected_reasoning: Optional[str] = None
  bias_type: Optional[str] = None

@dataclass
class Section:
  name: str
  questions: list[Question]

@dataclass
class Questionnaire:
  name: str
  description: str
  sections: list[Section]
```

## LLM Response Model
```python
@dataclass
class LLMResponse:
  model: str
  question_id: str
  response_text: str
  reasoning: Optional[str] = None
  confidence: Optional[float] = None
  timestamp: datetime
  latency_ms: int
```

## LLM Integration Guidelines

### Abstract Provider Interface
All LLM providers must implement a common interface to ensure consistency:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class LLMProvider(ABC):
  """Abstract base class for LLM providers."""
  
  @abstractmethod
  async def generate_response(
    self, 
    prompt: str, 
    model: str,
    **kwargs
  ) -> LLMResponse:
    """Generate response from LLM model."""
    pass
  
  @abstractmethod
  def list_available_models(self) -> List[str]:
    """Return list of available models for this provider."""
    pass
  
  @abstractmethod
  def validate_config(self) -> bool:
    """Validate provider configuration."""
    pass
```

### Provider Configuration Schema
```yaml
providers:
  provider_name:
    api_key: "${PROVIDER_API_KEY}"
    base_url: "https://api.provider.com"  # Optional
    timeout: 30  # seconds
    max_retries: 3
    models:
      - "model-name-1"
      - "model-name-2"
    default_params:
      temperature: 0.7
      max_tokens: 1000
```

### Supported Providers

#### OpenAI Integration
- Support GPT-4, GPT-3.5-turbo, and future models
- Handle rate limiting (RPM and TPM limits)
- Implement proper error handling for API errors
- Support streaming responses for long outputs

#### Anthropic Integration
- Support Claude models (Opus, Sonnet, Haiku)
- Handle Anthropic-specific rate limits
- Implement proper message formatting
- Support system prompts appropriately

#### Local Model Integration
- Support local inference servers (Ollama, vLLM, etc.)
- Handle connection errors gracefully
- Support custom model names and parameters
- Implement health checks for local servers

#### Custom Provider Support
- Allow easy addition of new providers
- Provide templates for new provider implementations
- Support provider-specific parameters
- Document the provider integration process

### Request Standardization
```python
@dataclass
class LLMRequest:
  prompt: str
  model: str
  temperature: float = 0.7
  max_tokens: int = 1000
  system_prompt: Optional[str] = None
  stop_sequences: Optional[List[str]] = None
  provider_specific: Dict[str, Any] = field(default_factory=dict)
```

### Response Standardization
```python
@dataclass
class LLMResponse:
  text: str
  model: str
  provider: str
  timestamp: datetime
  latency_ms: int
  token_count: Optional[int] = None
  finish_reason: Optional[str] = None
  cost_estimate: Optional[float] = None
  metadata: Dict[str, Any] = field(default_factory=dict)
```

### Error Handling for Providers

#### API Error Categories
- **Authentication Errors**: Invalid API keys, expired tokens
- **Rate Limit Errors**: Too many requests, quota exceeded
- **Model Errors**: Model not found, model overloaded
- **Network Errors**: Connection timeouts, DNS failures
- **Validation Errors**: Invalid parameters, malformed requests

#### Retry Strategy
```python
async def make_request_with_retry(
  provider: LLMProvider,
  request: LLMRequest,
  max_retries: int = 3
) -> LLMResponse:
  """Make LLM request with exponential backoff retry."""
  for attempt in range(max_retries + 1):
    try:
      return await provider.generate_response(request)
    except RateLimitError:
      if attempt == max_retries:
        raise
      wait_time = 2 ** attempt
      await asyncio.sleep(wait_time)
    except (NetworkError, TemporaryError):
      if attempt == max_retries:
        raise
      await asyncio.sleep(1)
```

#### Error Logging
- Log all API errors with context
- Include request parameters (excluding sensitive data)
- Log retry attempts and backoff times
- Provide actionable error messages to users

### Rate Limiting and Quotas

#### Rate Limit Management
- Track requests per minute/hour for each provider
- Implement token bucket or sliding window algorithms
- Queue requests when approaching limits
- Provide rate limit status in responses

#### Cost Tracking
- Estimate costs for each request when possible
- Track token usage across models
- Provide cost summaries in benchmark results
- Warn users about high-cost operations

#### Concurrent Request Handling
```python
import asyncio
from asyncio import Semaphore

class RateLimitedProvider:
  def __init__(self, provider: LLMProvider, max_concurrent: int = 5):
    self.provider = provider
    self.semaphore = Semaphore(max_concurrent)
  
  async def generate_response(self, request: LLMRequest) -> LLMResponse:
    async with self.semaphore:
      return await self.provider.generate_response(request)
```

### Prompt Engineering

#### Prompt Templates
Create reusable prompt templates for different question types:

```python
SCENARIO_PROMPT_TEMPLATE = """
You are participating in a reasoning benchmark. Please read the following scenario carefully and provide your response.

Scenario: {scenario}

Please provide:
1. Your decision or choice
2. Your reasoning process
3. Any assumptions you made

Response:
"""

CHOICE_PROMPT_TEMPLATE = """
Please select the best option from the choices below and explain your reasoning.

Question: {question}

Options:
{options}

Your choice and reasoning:
"""
```

#### Prompt Validation
- Validate prompt length against model limits
- Check for potentially harmful content
- Ensure prompts are clear and unambiguous
- Test prompts across different models

### Model-Specific Considerations

#### Context Length Limits
- Track context length for each model
- Truncate or summarize long prompts when needed
- Warn users about context length issues
- Implement sliding window for long conversations

#### Model Capabilities
- Document model-specific features and limitations
- Handle models that do not support certain parameters
- Provide fallback options for unsupported features
- Test compatibility across model versions

#### Response Parsing
- Handle different response formats across models
- Extract structured data from free-form responses
- Implement robust parsing with error handling
- Validate response completeness

### Testing LLM Integrations

#### Test-First Development
- Always create unit tests and mock tests before implementing LLM provider code
- Write tests for error handling scenarios before implementation
- Check `pyproject.toml` for existing test dependencies before adding new ones with `uv add --dev`

#### Mock Responses
Create realistic mock responses for testing:

```python
@pytest.fixture
def mock_openai_response():
  return LLMResponse(
    text="This is a test response from GPT-4.",
    model="gpt-4",
    provider="openai",
    timestamp=datetime.now(),
    latency_ms=1500,
    token_count=25,
    finish_reason="stop"
  )
```

#### Integration Tests
- Test against real APIs in CI/CD while respecting rate limits
- Use dedicated test API keys
- Validate response formats and error handling
- Test timeout and retry mechanisms

#### Performance Testing
- Measure response times across providers
- Test concurrent request handling
- Monitor memory usage during batch processing
- Validate rate limiting effectiveness

### Security Considerations

#### API Key Management
- Never log API keys or tokens
- Use environment variables for sensitive data
- Implement key rotation procedures
- Monitor for key exposure in logs or errors

#### Request Sanitization
- Sanitize user inputs before sending to APIs
- Prevent prompt injection attacks
- Validate request parameters
- Limit request sizes and frequencies

#### Response Handling
- Sanitize responses before processing
- Handle potentially harmful content appropriately
- Log security-relevant events
- Implement content filtering when required

### Monitoring and Observability

#### Metrics Collection
- Track request/response times by provider and model
- Monitor error rates and types
- Measure token usage and costs
- Track rate limit utilization

#### Health Checks
- Implement provider health checks
- Monitor API endpoint availability
- Check model availability and performance
- Alert on service degradation

#### Logging Standards
```python
import structlog

logger = structlog.get_logger()

async def log_llm_request(request: LLMRequest, response: LLMResponse):
  logger.info(
    "llm_request_completed",
    provider=response.provider,
    model=response.model,
    latency_ms=response.latency_ms,
    token_count=response.token_count,
    success=True
  )
```
