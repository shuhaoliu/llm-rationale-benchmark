"""Load and validate connector configurations from YAML documents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from ..exceptions import ConfigurationError
from .connector_models import (
  LLMConnectorConfig,
  format_validation_error,
)


class ConnectorConfigLoader:
  """Load connector configurations keyed by ``provider/model`` selectors."""

  def load(self, path: Path) -> Dict[str, LLMConnectorConfig]:
    """Return validated connector configurations from the supplied YAML file."""

    try:
      raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
      raise ConfigurationError(
        f"Configuration file not found: {path}",
        config_file=str(path),
      ) from exc
    except OSError as exc:
      raise ConfigurationError(
        f"Failed to read configuration file: {exc}",
        config_file=str(path),
      ) from exc

    try:
      document = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
      raise ConfigurationError(
        f"Invalid YAML in configuration file: {exc}",
        config_file=str(path),
      ) from exc

    if not isinstance(document, dict):
      raise ConfigurationError(
        "Configuration document must be a mapping", config_file=str(path)
      )

    resolved = self._resolve_environment(document)
    defaults = resolved.get("defaults", {})
    models_section = resolved.get("models")

    if defaults and not isinstance(defaults, dict):
      raise ConfigurationError(
        "defaults must be a mapping when provided",
        config_file=str(path),
        field="defaults",
      )

    if not isinstance(models_section, dict) or not models_section:
      raise ConfigurationError(
        "models section must be a non-empty mapping",
        config_file=str(path),
        field="models",
      )

    defaults = dict(defaults)
    configurations: Dict[str, LLMConnectorConfig] = {}
    errors: list[str] = []

    for key, entry in models_section.items():
      if not isinstance(key, str) or "/" not in key:
        errors.append(
          f"Model selector '{key}' must use the 'provider/model' format"
        )
        continue

      provider_name, model_id = key.split("/", 1)
      if not provider_name or not model_id:
        errors.append(
          f"Model selector '{key}' must include both provider and model"
        )
        continue

      if entry is None:
        entry = {}

      if not isinstance(entry, dict):
        errors.append(
          f"Configuration for '{key}' must be a mapping"
        )
        continue

      merged = self._deep_merge(defaults, entry)
      merged.setdefault("provider", provider_name)
      merged.setdefault("model", model_id)

      try:
        config = LLMConnectorConfig.model_validate(merged)
      except Exception as exc:  # pragma: no cover - handled via ValidationError
        if hasattr(exc, "errors"):
          message = format_validation_error(exc)  # type: ignore[arg-type]
        else:
          message = str(exc)
        errors.append(f"{key}: {message}")
        continue

      if config.provider.value != provider_name:
        errors.append(
          f"{key}: provider entry '{config.provider.value}' does not match key"
        )
        continue

      configurations[key] = config

    if errors:
      message = "Configuration validation failed:\n" + "\n".join(errors)
      raise ConfigurationError(message, config_file=str(path))

    return configurations

  def _resolve_environment(self, value: Any) -> Any:
    """Recursively resolve ``${VAR}`` placeholders against ``os.environ``."""

    if isinstance(value, str):
      result = value
      start = 0
      while True:
        start = result.find("${", start)
        if start == -1:
          break
        end = result.find("}", start)
        if end == -1:
          break
        name = result[start + 2 : end]
        if not name:
          start = end + 1
          continue
        env_value = os.getenv(name)
        if env_value is None:
          raise ConfigurationError(
            f"Environment variable '{name}' referenced but not set"
          )
        result = result[:start] + env_value + result[end + 1 :]
        start = start + len(env_value)
      return result

    if isinstance(value, dict):
      return {k: self._resolve_environment(v) for k, v in value.items()}

    if isinstance(value, list):
      return [self._resolve_environment(item) for item in value]

    return value

  def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``overlay`` merged on top of ``base`` (non-destructive)."""

    merged: Dict[str, Any] = {}
    for key, value in base.items():
      merged[key] = value

    for key, value in overlay.items():
      if (
        key in merged
        and isinstance(merged[key], dict)
        and isinstance(value, dict)
      ):
        merged[key] = self._deep_merge(merged[key], value)
      else:
        merged[key] = value

    return merged

