"""Aliyun provider client."""

from __future__ import annotations

from ..config.connector_models import LLMConnectorConfig
from .openai_compatible import OpenAICompatibleClient


class AliyunClient(OpenAICompatibleClient):
  """Client for Aliyun's OpenAI-compatible chat endpoint."""

  def __init__(self, config: LLMConnectorConfig):
    super().__init__(config)

  def _uses_native_structured_output(self) -> bool:
    return self.config.default_params.get("enable_thinking") is False
