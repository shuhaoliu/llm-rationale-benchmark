from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import click

from .errors import QuestionnaireConfigError
from .loader import load_questionnaire_file


def _expand_candidates(
  explicit_paths: Iterable[Path],
  default_dir: Path,
) -> list[Path]:
  paths: list[Path] = []
  sources = list(explicit_paths) or [default_dir]
  for source in sources:
    path = Path(source)
    if path.is_dir():
      candidates = sorted(path.glob("*.yaml"))
      paths.extend(candidate.resolve() for candidate in candidates)
    else:
      paths.append(path.resolve())
  # Preserve order while removing duplicates
  unique: dict[str, Path] = {}
  for path in paths:
    unique.setdefault(str(path), path)
  return list(unique.values())


def _format_error(path: Path, error: QuestionnaireConfigError) -> str:
  parts = [f"{path}: {error.message}"]
  if error.location:
    parts.append(f"location={error.location}")
  if error.line_number is not None:
    parts.append(f"line={error.line_number}")
  return " | ".join(parts)


@click.command()
@click.argument(
  "paths",
  nargs=-1,
  type=click.Path(path_type=Path),
)
@click.option(
  "--questionnaire-dir",
  "questionnaire_dir",
  type=click.Path(path_type=Path, file_okay=False),
  default=Path("config/questionnaires"),
  show_default=True,
  help="Directory searched when no explicit paths are provided.",
)
@click.option(
  "--json",
  "json_output",
  is_flag=True,
  help="Emit validation results as JSON.",
)
def cli(
  paths: tuple[Path, ...],
  questionnaire_dir: Path,
  json_output: bool,
) -> None:
  """Validate questionnaire YAML files using the project schema."""
  file_paths = _expand_candidates(paths, questionnaire_dir)
  if not file_paths:
    raise click.ClickException("No questionnaire files found to validate.")

  results: list[dict[str, object]] = []
  exit_code = 0

  for file_path in file_paths:
    try:
      questionnaire = load_questionnaire_file(
        file_path,
        enforce_id_match=True,
      )
    except QuestionnaireConfigError as error:
      exit_code = 1
      results.append(
        {
          "path": str(file_path),
          "status": "error",
          "message": error.message,
          "location": error.location,
          "line_number": error.line_number,
        }
      )
      if not json_output:
        click.secho(_format_error(file_path, error), fg="red", err=True)
      continue
    except Exception as error:  # pragma: no cover
      exit_code = 1
      results.append(
        {
          "path": str(file_path),
          "status": "error",
          "message": str(error),
          "error_type": error.__class__.__name__,
        }
      )
      if not json_output:
        click.secho(
          f"{file_path}: unexpected error {error}",
          fg="red",
          err=True,
        )
      continue

    results.append(
      {
        "path": str(file_path),
        "status": "ok",
        "questionnaire_id": questionnaire.id,
        "questionnaire_name": questionnaire.name,
      }
    )
    if not json_output:
      click.secho(
        f"{file_path}: OK (id={questionnaire.id})",
        fg="green",
      )

  if json_output:
    payload = {
      "status": "ok" if exit_code == 0 else "error",
      "results": results,
    }
    click.echo(json.dumps(payload, indent=2))

  if exit_code != 0:
    raise click.exceptions.Exit(exit_code)


def main() -> None:
  """Entry point that handles interrupts gracefully."""
  try:
    cli.main(standalone_mode=False)
  except KeyboardInterrupt:
    click.secho("Validation interrupted by user.", fg="yellow", err=True)
    raise SystemExit(1)
  except click.exceptions.Exit as exc:
    raise SystemExit(exc.exit_code)
  except click.ClickException as exc:  # pragma: no cover
    exc.show()
    raise SystemExit(exc.exit_code)


if __name__ == "__main__":  # pragma: no cover
  main()
