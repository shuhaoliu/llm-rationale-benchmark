"""OpenAI Chat Completions provider client."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config.connector_models import LLMConnectorConfig, ResponseFormat
from ..exceptions import (
  AuthenticationError,
  ModelNotFoundError,
  ProviderError,
  RetryableProviderError,
)
from ..provider_client import ProviderResponse
from .base import JSONHTTPProvider


class OpenAIChatClient(JSONHTTPProvider):
  """Client for the OpenAI Chat Completions API."""

  def __init__(self, config: LLMConnectorConfig):
    super().__init__(config)
    base = config.base_url or config.endpoint or "https://api.openai.com/v1"
    self.base_url = base.rstrip("/")
    self.headers = self._build_headers()

  def _build_headers(self) -> Dict[str, str]:
    headers = {
      "Authorization": f"Bearer {self._require_api_key()}",
      "Content-Type": "application/json",
    }
    custom_headers = self.config.provider_specific.get("headers", {})
    if isinstance(custom_headers, dict):
      for key, value in custom_headers.items():
        headers[str(key)] = str(value)
    return headers

  def _generate(
    self,
    messages: List[dict[str, str]],
    *,
    response_format: ResponseFormat,
  ) -> ProviderResponse:
    payload = self._build_payload(messages, response_format)
    url = f"{self.base_url}/chat/completions"
    data, headers = self._post_json(url, self.headers, payload)

    try:
      choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
      raise ProviderError(
        self.config.provider.value,
        "Invalid response payload received from OpenAI",
        cause=exc,
      ) from exc

    content = self._extract_content(choice, response_format)
    metadata = {
      "usage": data.get("usage", {}),
      "id": data.get("id"),
      "headers": headers,
    }

    return ProviderResponse(
      content=content,
      raw=data,
      finish_reason=choice.get("finish_reason"),
      metadata=metadata,
    )

  def _build_payload(
    self,
    messages: List[dict[str, str]],
    response_format: ResponseFormat,
  ) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
      "model": self.config.model,
      "messages": messages,
    }
    payload.update(self.config.default_params)
    payload.update(self.config.provider_specific.get("request_overrides", {}))

    if self.config.temperature is not None:
      payload.setdefault("temperature", self.config.temperature)
    if self.config.top_p is not None:
      payload.setdefault("top_p", self.config.top_p)
    if self.config.max_tokens is not None:
      payload.setdefault("max_tokens", self.config.max_tokens)

    if response_format is ResponseFormat.JSON:
      payload.setdefault("response_format", {"type": "json_object"})

    return payload

  def _extract_content(
    self,
    choice: Dict[str, Any],
    response_format: ResponseFormat,
  ) -> str:
    message = choice.get("message", {})
    content = message.get("content")
    if isinstance(content, list):
      content = "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict)
      )
    if not isinstance(content, str):
      raise ProviderError(
        self.config.provider.value,
        "Missing content in OpenAI response",
      )
    if response_format is ResponseFormat.JSON:
      return content.strip()
    return content

  def _handle_http_error(
    self,
    exc,
    payload: dict[str, Any],
  ) -> RetryableProviderError:
    status = getattr(exc, "code", 0)
    message = self._extract_error_message(payload) or str(status)
    provider = self.config.provider.value

    if status == 401:
      raise AuthenticationError(provider, message)

    if status == 404:
      raise ModelNotFoundError(provider, self.config.model)

    if status == 429 or status >= 500:
      retry_after = None
      headers = getattr(exc, "headers", {}) or {}
      if "retry-after" in headers:
        try:
          retry_after = int(headers["retry-after"])
        except ValueError:
          retry_after = None
      return RetryableProviderError(
        provider,
        message,
        retry_after=retry_after,
      )

    raise ProviderError(provider, message)

