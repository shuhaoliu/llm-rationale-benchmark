"""Base provider client abstractions used by the conversation layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, List

from .config.connector_models import LLMConnectorConfig
from .exceptions import (
  ProviderError,
  StreamingNotSupportedError,
)


@dataclass
class ProviderResponse:
  """Container describing a provider response."""

  content: str
  raw: Any = None
  finish_reason: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)


class BaseProviderClient(ABC):
  """Abstract provider client used by :class:`LLMConversation`."""

  def __init__(self, config: LLMConnectorConfig):
    self.config = config

  def supports_streaming(self) -> bool:
    """Return ``True`` when the client can stream responses."""

    return False

  def requires_streaming(self) -> bool:
    """Return ``True`` if the provider only answers via streaming APIs."""

    return False

  def generate(
    self,
    messages: List[dict[str, str]],
    *,
    output_schema: dict[str, Any],
    stream: bool = False,
  ) -> ProviderResponse:
    """Generate a response from the provider."""

    if stream:
      if not self.supports_streaming():
        raise StreamingNotSupportedError(
          f"{self.config.provider.value} does not support streaming"
        )
      chunks = list(
        self.stream_generate(
          messages,
          output_schema=output_schema,
        )
      )
      content = "".join(chunks)
      return ProviderResponse(content=content, raw={"chunks": chunks})

    return self._generate(messages, output_schema=output_schema)

  def stream_generate(
    self,
    messages: List[dict[str, str]],
    *,
    output_schema: dict[str, Any],
  ) -> Iterable[str]:
    """Return streaming chunks. Implemented by subclasses when supported."""

    raise StreamingNotSupportedError(
      f"{self.config.provider.value} does not support streaming"
    )

  @abstractmethod
  def _generate(
    self,
    messages: List[dict[str, str]],
    *,
    output_schema: dict[str, Any],
  ) -> ProviderResponse:
    """Perform a non-streaming call against the provider."""


class StaticResponseProvider(BaseProviderClient):
  """Provider used in tests returning predefined responses."""

  def __init__(self, config: LLMConnectorConfig, responses: Iterable[str]):
    super().__init__(config)
    self._responses = list(responses)
    self._index = 0

  def _generate(
    self,
    messages: List[dict[str, str]],
    *,
    output_schema: dict[str, Any],
  ) -> ProviderResponse:
    try:
      content = self._responses[self._index]
    except IndexError as exc:  # pragma: no cover - guard rail
      raise ProviderError(
        self.config.provider.value,
        "StaticResponseProvider exhausted configured responses",
      ) from exc
    self._index += 1
    return ProviderResponse(content=content, raw={"messages": messages})
