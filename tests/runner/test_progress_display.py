from __future__ import annotations

from io import StringIO

from rationale_benchmark.runner.progress_display import (
  QueryProgressDisplay,
  QueryProgressSnapshot,
  QuerySectionStatus,
  render_query_tables,
)


def test_render_query_tables_compacts_inactive_population_rows() -> None:
  snapshot = QueryProgressSnapshot(
    llm_ids=["openai/gpt-4"],
    population_size=4,
    section_count=3,
    statuses={
      "openai/gpt-4": [
        [
          QuerySectionStatus.COMPLETED,
          QuerySectionStatus.COMPLETED,
          QuerySectionStatus.COMPLETED,
        ],
        [
          QuerySectionStatus.IN_PROGRESS,
          QuerySectionStatus.NOT_STARTED,
          QuerySectionStatus.COMPLETED,
        ],
        [
          QuerySectionStatus.NOT_STARTED,
          QuerySectionStatus.NOT_STARTED,
          QuerySectionStatus.NOT_STARTED,
        ],
        [
          QuerySectionStatus.NOT_STARTED,
          QuerySectionStatus.NOT_STARTED,
          QuerySectionStatus.NOT_STARTED,
        ],
      ]
    },
  )

  output = render_query_tables(snapshot)

  assert "openai/gpt-4" in output
  assert "General" not in output
  assert "section" not in output.lower()
  assert "|" not in output
  assert "✅ completed: 1" in output
  assert "⚪ not started: 2" in output
  assert "1 🟡 ⚪ ✅" in output
  assert "✅" in output
  assert "🟡" in output
  assert "⚪" in output


def test_render_query_tables_displays_error_state_and_interrupt_message() -> None:
  snapshot = QueryProgressSnapshot(
    llm_ids=["openai/gpt-4"],
    population_size=2,
    section_count=2,
    statuses={
      "openai/gpt-4": [
        [QuerySectionStatus.ERROR, QuerySectionStatus.ERROR],
        [QuerySectionStatus.COMPLETED, QuerySectionStatus.ERROR],
      ]
    },
    message=(
      "Interrupted: all remaining populations and sections are marked as "
      "error because they were not queried."
    ),
  )

  output = render_query_tables(snapshot)

  assert "❌ error: 1" in output
  assert "1 ✅ ❌" in output
  assert "Interrupted: all remaining populations and sections are marked as error" in (
    output
  )


def test_progress_display_repaints_in_place_when_interactive() -> None:
  stream = StringIO()
  display = QueryProgressDisplay(stream=stream, interactive=True)

  display.start(
    llm_ids=["openai/gpt-4"],
    population_size=1,
    section_count=2,
  )
  display.mark_section_started("openai/gpt-4", 0, 1)
  display.mark_section_error("openai/gpt-4", 0, 1)
  display.mark_unfinished_error(
    "Interrupted: all remaining populations and sections are marked as "
    "error because they were not queried."
  )

  output = stream.getvalue()

  assert "\033[H\033[J" not in output
  assert "\033[" in output
  assert "🟡" in output
  assert "❌" in output
  assert "Interrupted:" in output
