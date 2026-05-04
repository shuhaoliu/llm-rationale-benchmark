"""Terminal progress display for questionnaire query execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from sys import stderr
from typing import Protocol, TextIO


class QuerySectionStatus(Enum):
  """Visual state for one population member and questionnaire section."""

  NOT_STARTED = "⚪"
  IN_PROGRESS = "🟡"
  COMPLETED = "✅"


@dataclass(frozen=True)
class QueryProgressSnapshot:
  """Complete visual state for the active query process."""

  llm_ids: list[str]
  population_size: int
  section_count: int
  statuses: dict[str, list[list[QuerySectionStatus]]]

  @classmethod
  def empty(
    cls,
    *,
    llm_ids: list[str],
    population_size: int,
    section_count: int,
  ) -> QueryProgressSnapshot:
    """Create an all-not-started progress snapshot."""
    return cls(
      llm_ids=list(llm_ids),
      population_size=population_size,
      section_count=section_count,
      statuses={
        llm_id: [
          [QuerySectionStatus.NOT_STARTED for _ in range(section_count)]
          for _ in range(population_size)
        ]
        for llm_id in llm_ids
      },
    )


class QueryProgressDisplayProtocol(Protocol):
  """Display callbacks consumed by the benchmark runner."""

  def start(
    self,
    *,
    llm_ids: list[str],
    population_size: int,
    section_count: int,
  ) -> None: ...

  def mark_section_started(
    self,
    llm_id: str,
    population_index: int,
    section_index: int,
  ) -> None: ...

  def mark_section_completed(
    self,
    llm_id: str,
    population_index: int,
    section_index: int,
  ) -> None: ...

  def stop(self) -> None: ...


def render_query_tables(snapshot: QueryProgressSnapshot) -> str:
  """Render one compact emoji status table per LLM."""
  tables = []
  for llm_id in snapshot.llm_ids:
    completed_count = 0
    not_started_count = 0
    active_rows = []
    for population_index, statuses in enumerate(snapshot.statuses[llm_id]):
      if _all_status(statuses, QuerySectionStatus.COMPLETED):
        completed_count += 1
      elif _all_status(statuses, QuerySectionStatus.NOT_STARTED):
        not_started_count += 1
      else:
        cells = " ".join(status.value for status in statuses)
        active_rows.append(f"{population_index} {cells}")

    rows = [f"LLM: {llm_id}"]
    if completed_count:
      rows.append(f"✅ completed: {completed_count}")
    rows.extend(active_rows)
    if not_started_count:
      rows.append(f"⚪ not started: {not_started_count}")
    tables.append("\n".join(rows))
  return "\n\n".join(tables)


def _all_status(
  statuses: list[QuerySectionStatus],
  status: QuerySectionStatus,
) -> bool:
  return all(current_status is status for current_status in statuses)


class QueryProgressDisplay:
  """Repainting terminal display for query progress."""

  def __init__(
    self,
    *,
    stream: TextIO = stderr,
    interactive: bool = True,
  ) -> None:
    self._stream = stream
    self._interactive = interactive
    self._snapshot: QueryProgressSnapshot | None = None
    self._has_rendered = False
    self._last_render_line_count = 0

  def start(
    self,
    *,
    llm_ids: list[str],
    population_size: int,
    section_count: int,
  ) -> None:
    self._snapshot = QueryProgressSnapshot.empty(
      llm_ids=llm_ids,
      population_size=population_size,
      section_count=section_count,
    )
    self._render()

  def mark_section_started(
    self,
    llm_id: str,
    population_index: int,
    section_index: int,
  ) -> None:
    self._set_status(
      llm_id,
      population_index,
      section_index,
      QuerySectionStatus.IN_PROGRESS,
    )

  def mark_section_completed(
    self,
    llm_id: str,
    population_index: int,
    section_index: int,
  ) -> None:
    self._set_status(
      llm_id,
      population_index,
      section_index,
      QuerySectionStatus.COMPLETED,
    )

  def stop(self) -> None:
    if self._interactive and self._has_rendered:
      self._stream.write("\n")
      self._stream.flush()

  def _set_status(
    self,
    llm_id: str,
    population_index: int,
    section_index: int,
    status: QuerySectionStatus,
  ) -> None:
    if self._snapshot is None:
      return
    self._snapshot.statuses[llm_id][population_index][section_index] = status
    self._render()

  def _render(self) -> None:
    if self._snapshot is None:
      return
    if self._interactive and self._has_rendered:
      self._stream.write(f"\033[{self._last_render_line_count}F\033[J")
    rendered = render_query_tables(self._snapshot)
    self._stream.write(rendered + "\n")
    if not self._interactive:
      self._stream.write("\n")
    self._stream.flush()
    self._has_rendered = True
    self._last_render_line_count = len(rendered.splitlines())


__all__ = [
  "QueryProgressDisplay",
  "QueryProgressDisplayProtocol",
  "QueryProgressSnapshot",
  "QuerySectionStatus",
  "render_query_tables",
]
