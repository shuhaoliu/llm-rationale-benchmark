"""Anthropic Messages API provider client."""

from __future__ import annotations

from typing import Any, Dict, List

from ..config.connector_models import LLMConnectorConfig
from ..exceptions import (
  AuthenticationError,
  ProviderError,
  RetryableProviderError,
)
from ..provider_client import ProviderResponse
from .base import JSONHTTPProvider


class AnthropicClient(JSONHTTPProvider):
  """Client for Anthropic's Messages API."""

  def __init__(self, config: LLMConnectorConfig):
    super().__init__(config)
    base = config.base_url or config.endpoint or "https://api.anthropic.com"
    self.base_url = base.rstrip("/")
    self.headers = self._build_headers()

  def supports_streaming(self) -> bool:
    return False

  def _build_headers(self) -> Dict[str, str]:
    headers = {
      "x-api-key": self._require_api_key(),
      "anthropic-version": self._anthropic_version(),
      "content-type": "application/json",
    }
    custom_headers = self.config.provider_specific.get("headers", {})
    if isinstance(custom_headers, dict):
      for key, value in custom_headers.items():
        headers[str(key)] = str(value)
    return headers

  def _anthropic_version(self) -> str:
    version = self.config.provider_specific.get("version")
    if isinstance(version, str) and version.strip():
      return version
    return "2023-06-01"

  def _generate(
    self,
    messages: List[dict[str, str]],
    *,
    output_schema: dict[str, Any],
  ) -> ProviderResponse:
    payload = self._build_payload(messages, output_schema)
    url = f"{self.base_url}/v1/messages"
    data, headers = self._post_json(url, self.headers, payload)

    content_blocks = data.get("content", [])
    text = self._extract_text(content_blocks)
    metadata = {
      "usage": data.get("usage", {}),
      "id": data.get("id"),
      "headers": headers,
    }

    return ProviderResponse(
      content=text,
      raw=data,
      finish_reason=data.get("stop_reason"),
      metadata=metadata,
    )

  def _build_payload(
    self,
    messages: List[dict[str, str]],
    output_schema: dict[str, Any],
  ) -> Dict[str, Any]:
    system_prompt, converted = self._convert_messages(messages)

    payload: Dict[str, Any] = {
      "model": self.config.model,
      "messages": converted,
      "max_tokens": self._resolve_max_tokens(),
    }
    if system_prompt:
      payload["system"] = system_prompt

    payload.update(self.config.default_params)
    overrides = self.config.provider_specific.get("request_overrides")
    if isinstance(overrides, dict):
      payload.update(overrides)

    if self.config.temperature is not None:
      payload.setdefault("temperature", self.config.temperature)
    if self.config.top_p is not None:
      payload.setdefault("top_p", self.config.top_p)

    payload["output_config"] = {
      "format": {
        "type": "json_schema",
        "schema": output_schema,
      }
    }

    return payload

  def _convert_messages(
    self,
    messages: List[dict[str, str]],
  ) -> tuple[str | None, List[Dict[str, Any]]]:
    system_prompt: str | None = None
    converted: List[Dict[str, Any]] = []

    for message in messages:
      role = message.get("role")
      content = message.get("content", "")
      if role == "system":
        system_prompt = (
          content
          if system_prompt is None
          else f"{system_prompt}\n{content}"
        )
        continue

      converted.append(
        {
          "role": "assistant" if role == "assistant" else "user",
          "content": [
            {
              "type": "text",
              "text": content,
            }
          ],
        }
      )

    return system_prompt, converted

  def _resolve_max_tokens(self) -> int:
    if self.config.max_tokens is not None:
      return self.config.max_tokens
    value = self.config.default_params.get("max_tokens")
    if isinstance(value, int):
      return value
    return 1024

  def _extract_text(self, blocks: Any) -> str:
    if not isinstance(blocks, list):
      raise ProviderError(
        self.config.provider.value,
        "Anthropic response did not include content blocks",
      )
    chunks: list[str] = []
    for block in blocks:
      if isinstance(block, dict) and block.get("type") == "text":
        text = block.get("text")
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
