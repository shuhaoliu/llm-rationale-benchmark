"""Google Gemini provider client."""

from __future__ import annotations

import urllib.parse
from typing import Any, Dict, List

from ..config.connector_models import LLMConnectorConfig, ResponseFormat
from ..exceptions import (
  AuthenticationError,
  ProviderError,
  RetryableProviderError,
)
from ..provider_client import ProviderResponse
from .base import JSONHTTPProvider


class GeminiClient(JSONHTTPProvider):
  """Client for Google's Generative Language (Gemini) API."""

  def __init__(self, config: LLMConnectorConfig):
    super().__init__(config)
    base = config.base_url or config.endpoint or "https://generativelanguage.googleapis.com"
    self.base_url = base.rstrip("/")
    self.headers = {
      "Content-Type": "application/json",
    }

  def _generate(
    self,
    messages: List[dict[str, str]],
    *,
    response_format: ResponseFormat,
  ) -> ProviderResponse:
    payload = self._build_payload(messages, response_format)
    url = self._build_request_url()
    data, headers = self._post_json(url, self.headers, payload)

    candidates = data.get("candidates", [])
    if not candidates:
      raise ProviderError(
        self.config.provider.value,
        "Gemini response did not include candidates",
      )
    content = candidates[0].get("content", {})
    text = self._extract_text(content)

    metadata = {
      "safety_ratings": candidates[0].get("safetyRatings", []),
      "headers": headers,
    }

    return ProviderResponse(
      content=text,
      raw=data,
      finish_reason=candidates[0].get("finishReason"),
      metadata=metadata,
    )

  def _build_request_url(self) -> str:
    api_key = self._require_api_key()
    query = urllib.parse.urlencode({"key": api_key})
    return f"{self.base_url}/v1beta/models/{self.config.model}:generateContent?{query}"

  def _build_payload(
    self,
    messages: List[dict[str, str]],
    response_format: ResponseFormat,
  ) -> Dict[str, Any]:
    contents: List[Dict[str, Any]] = []
    for message in messages:
      role = message.get("role")
      content = message.get("content", "")
      if role == "system":
        continue
      contents.append(
        {
          "role": "model" if role == "assistant" else "user",
          "parts": [{"text": content}],
        }
      )

    payload: Dict[str, Any] = {"contents": contents}

    generation_config: Dict[str, Any] = {}
    if self.config.temperature is not None:
      generation_config.setdefault("temperature", self.config.temperature)
    if self.config.top_p is not None:
      generation_config.setdefault("topP", self.config.top_p)

    max_tokens = self.config.max_tokens or self.config.default_params.get("max_tokens")
    if isinstance(max_tokens, int):
      generation_config.setdefault("maxOutputTokens", max_tokens)

    if self.config.default_params:
      generation_config.update(self.config.default_params)

    overrides = self.config.provider_specific.get("generation_config")
    if isinstance(overrides, dict):
      generation_config.update(overrides)

    if generation_config:
      payload["generationConfig"] = generation_config

    if response_format is ResponseFormat.JSON:
      payload.setdefault("responseMimeType", "application/json")

    return payload

  def _extract_text(self, content: Any) -> str:
    if not isinstance(content, dict):
      raise ProviderError(
        self.config.provider.value,
        "Gemini response is missing content",
      )
    parts = content.get("parts", [])
    chunks: list[str] = []
    for part in parts:
      text = part.get("text") if isinstance(part, dict) else None
      if isinstance(text, str):
        chunks.append(text)
    return "".join(chunks)

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

