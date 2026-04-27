# Data Models and Integration Interfaces

This reference consolidates the core interfaces shared across the CLI,
questionnaire loader, LLM connector layer, and benchmark runner. It aligns with
`docs/01-llm/design.md`, `docs/01-llm/configs.md`, and
`docs/02-questionnaire/design.md`.

## Questionnaire Domain Models

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any

class QuestionType(str, Enum):
  RATING_5 = "rating-5"
  RATING_7 = "rating-7"
  RATING_11 = "rating-11"
  CHOICE = "choice"

@dataclass
class ScoringRule:
  total: int
  weights: dict[str, int]  # normalized for both rating and choice questions

@dataclass
class Question:
  id: str
  type: QuestionType
  prompt: str
  options: dict[str, str] | None
  scoring: ScoringRule

@dataclass
class Section:
  name: str
  instructions: str | None
  questions: list[Question]

@dataclass
class Questionnaire:
  id: str
  name: str
  description: str | None
  version: int | None
  metadata: dict[str, str | int]
  system_prompt: str
  default_population: int
  sections: list[Section]
```

### Scoring Outputs

```python
@dataclass
class QuestionScore:
  question_id: str
  awarded: int
  total: int

@dataclass
class SectionScore:
  section_name: str
  questions: list[QuestionScore]

@dataclass
class QuestionnaireScore:
  questionnaire_id: str
  sections: list[SectionScore]

  @property
  def awarded(self) -> int:
    return sum(q.awarded for section in self.sections for q in section.questions)

  @property
  def total(self) -> int:
    return sum(q.total for section in self.sections for q in section.questions)
```

## LLM Configuration Models

The connector layer loads YAML into validated configuration objects.

```python
from dataclasses import dataclass
from typing import Literal, Mapping

ProviderKey = Literal[
  "openai",
  "anthropic",
  "gemini",
  "openai_compatible"
]

@dataclass(frozen=True)
class RetryPolicy:
  max_attempts: int = 3
  backoff_seconds: float = 1.0
  backoff_multiplier: float = 2.0
  jitter: float = 0.1

@dataclass(frozen=True)
class LLMConnectorConfig:
  provider: ProviderKey
  model: str
  api_key: str
  endpoint: str | None
  base_url: str | None
  timeout_seconds: int
  temperature: float | None
  top_p: float | None
  max_tokens: int | None
  system_prompt: str | None
  response_format: Literal["json", "text"]
  default_params: Mapping[str, Any]
  provider_specific: Mapping[str, Any]
  metadata: Mapping[str, Any]
  retry: RetryPolicy
```

The configuration loader surfaces a mapping of `"{provider}/{model}"` selectors
to `LLMConnectorConfig` instances. Provider keys correspond to the registry
documented in `docs/01-llm/configs.md`.

## Conversation Interfaces

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Sequence

@dataclass
class ConversationTurn:
  role: Literal["system", "user", "assistant"]
  content: str
  timestamp: datetime
  metadata: dict[str, Any]

@dataclass
class LLMRequest:
  prompt: str
  model: str
  system_prompt: str | None = None
  temperature: float | None = None
  max_tokens: int | None = None
  stop_sequences: Sequence[str] | None = None
  provider_specific: dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResponse:
  text: str
  model: str
  provider: str
  timestamp: datetime
  latency_ms: int
  token_count: int | None = None
  finish_reason: str | None = None
  cost_estimate: float | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
```

```python
class LLMConversation:
  """High-level conversation interface exposed to benchmark runners."""

  def ask(
    self,
    prompt: str,
    validator: Callable[[LLMResponse], bool] | None = None,
    *,
    max_attempts: int | None = None
  ) -> LLMResponse: ...

  def archive(self) -> list[ConversationTurn]:
    """Freeze the conversation and return the immutable transcript."""
    ...

  @property
  def is_archived(self) -> bool: ...
```

### Conversation Factory

```python
class LLMConversationFactory:
  """Creates conversations from validated configuration mappings."""

  def __init__(
    self,
    configs: Mapping[str, LLMConnectorConfig],
    provider_registry: ProviderRegistry
  ): ...

  def create(
    self,
    target: str,
    *,
    system_prompt: str | None = None
  ) -> LLMConversation: ...
```

### Provider Interfaces

```python
from abc import ABC, abstractmethod

class BaseProviderClient(ABC):
  """Low-level provider adapter. See docs/01-llm/design.md."""

  @abstractmethod
  async def generate(self, request: LLMRequest) -> LLMResponse:
    """Execute a completion request and return a normalized response."""

  @abstractmethod
  def list_models(self) -> list[str]:
    """Expose available model identifiers for validation."""

  @abstractmethod
  def validate_config(self, config: LLMConnectorConfig) -> None:
    """Raise ConfigurationError when mandatory provider fields are missing."""
```

```python
class ProviderRegistry:
  """Maps provider keys to concrete BaseProviderClient implementations."""

  def register(self, key: str, client: BaseProviderClient) -> None: ...
  def get(self, key: str) -> BaseProviderClient: ...
```

Provider adapters handle provider-specific payload shapes, authentication, and
response parsing while adhering to the normalized request/response interfaces.

## Benchmark Result Models

```python
@dataclass
class QuestionResult:
  questionnaire_id: str
  section_name: str
  question_id: str
  model: str
  population_index: int
  response_text: str
  reasoning: str | None
  score: QuestionScore
  latency_ms: int
  metadata: dict[str, Any]

@dataclass
class ModelBenchmarkResult:
  model: str
  population_index: int
  questionnaire_scores: list[QuestionnaireScore]
  questions: list[QuestionResult]
  section_transcripts: dict[str, list[ConversationTurn]]

@dataclass
class BenchmarkSummary:
  questionnaires_run: int
  total_population: int
  total_questions: int
  models_tested: int
  average_scores_by_questionnaire: dict[str, float]
  average_scores_by_model: dict[str, float]
  cost_estimates: dict[str, float]
```

The CLI serializes these structures into the JSON report described in
`docs/architecture.md`.

## Error Types

```python
class ConfigurationError(Exception):
  """Raised for LLM configuration issues (missing env vars, invalid schema)."""

class QuestionnaireConfigError(Exception):
  """Raised for questionnaire validation failures."""

class ConversationArchivedError(Exception):
  """Raised when attempting to reuse an archived conversation."""
```

All subsystems should propagate these domain-specific exceptions so the CLI can
render actionable error messages and exit codes.
