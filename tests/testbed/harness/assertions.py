"""Assertion helpers for testbed scenarios.

Every helper opens the Aim repo read-only, queries for the expected
condition, and raises AssertionError with a diagnostic message on
mismatch. Helpers do NOT clean up the repo — teardown is the fixture's
job.

The read side uses ``aim.Repo(path, read_only=True)`` against the on-disk
repo the aim server is writing to. Scenarios that need to verify what
actually landed (memtable-flushed vs. buffered) use these helpers rather
than the aim server's HTTP surface.
"""
from __future__ import annotations

from pathlib import Path

__all__ = [
    "assert_run_name",
    "assert_run_tag",
    "assert_metric_landed",
    "assert_metric_count",
    "assert_metric_values",
    "assert_schema_finalized_event",
    "assert_run_closed",
    "assert_no_run_exists",
    "get_run_tags",
    "get_metric_series",
]


def assert_run_name(repo_path: Path, run_hash: str, expected_name: str) -> None:
    """Assert that ``repo_path``'s run ``run_hash`` has ``run.name == expected_name``.

    Non-obvious contract: reads ``run.name`` not ``run.props.name`` — after
    schema-finalize, callers hit a subtle attribute vs. property fallback
    that changes across Aim versions. Helper canonicalizes to the value the
    dashboard actually renders.
    """
    raise NotImplementedError


def assert_run_tag(repo_path: Path, run_hash: str, tag: str, expected_value: str) -> None:
    """Assert that ``run[tag] == expected_value``. Raises if tag missing."""
    raise NotImplementedError


def assert_metric_landed(repo_path: Path, run_hash: str, metric_name: str) -> None:
    """Assert that at least one value has been written under ``metric_name``.

    Passes iff the metric appears in the run's metrics enumeration AND its
    series has at least one (step, value) point that survived memtable flush.
    """
    raise NotImplementedError


def assert_metric_count(
    repo_path: Path,
    run_hash: str,
    metric_name: str,
    expected_count: int,
) -> None:
    """Assert exactly ``expected_count`` values under ``metric_name``."""
    raise NotImplementedError


def assert_metric_values(
    repo_path: Path,
    run_hash: str,
    metric_name: str,
    expected: list[tuple[int, float]],
) -> None:
    """Assert (step, value) pairs match ``expected`` exactly, in order."""
    raise NotImplementedError


def assert_schema_finalized_event(
    stats_jsonl_path: Path,
    expected_metric_names: list[str] | None = None,
) -> None:
    """Assert the callback's stats jsonl contains a ``schema_finalized`` event.

    If ``expected_metric_names`` is provided, verifies the event's
    metric_names field is a superset. Used by schema-finalize scenarios
    in test_core.py to verify the finalize side-band emitted correctly.
    """
    raise NotImplementedError


def assert_run_closed(repo_path: Path, run_hash: str) -> None:
    """Assert the run is closed (has a non-null ``end_time``)."""
    raise NotImplementedError


def assert_no_run_exists(repo_path: Path, run_hash: str) -> None:
    """Assert no run with the given hash exists in the repo."""
    raise NotImplementedError


def get_run_tags(repo_path: Path, run_hash: str) -> dict[str, str]:
    """Return all tags on the run as a dict. Used for tag-fidelity checks."""
    raise NotImplementedError


def get_metric_series(
    repo_path: Path,
    run_hash: str,
    metric_name: str,
) -> list[tuple[int, float]]:
    """Return the (step, value) pairs for ``metric_name`` in order."""
    raise NotImplementedError
