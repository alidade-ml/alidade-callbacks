"""Assertion helpers for testbed scenarios.

Every helper opens the Aim repo, indexes the target run so its writes
are visible (this is the read-visibility step astrolabe's sync sidecar
performs), then queries for the expected condition. Raises
AssertionError with a diagnostic on mismatch. Helpers do NOT clean up
the repo — teardown is the fixture's job.

Read-visibility discipline (learned during Stage 3 debug):

The aim server accepts writes via its tracking API and stores them
under ``seqs/chunks/<run_hash>/`` on disk. Those writes are NOT
visible to a fresh ``Repo.get_run(hash).metrics()`` call by default —
the meta index (``meta/index/``, ``run_metadata.sqlite``) must be
populated first. Astrolabe's ``sync.py`` does this via:

    repo.request_props(run_hash, read_only=False, created_at=...)
    RepoIndexManager.get_index_manager(repo).index(run_hash)

Every helper here that queries by hash calls ``_index_run()`` first.
The call is idempotent — after the first call for a hash it's near
no-op.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

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


def _open_repo(repo_path: Path) -> Any:
    """Open the on-disk Aim repo at the bind-mounted host path.

    Reads from the same directory the aim server writes to via bind
    mount. Indexing (``_index_run``) is required before writes become
    visible — see module docstring.
    """
    import aim

    return aim.Repo(str(repo_path))


def _index_run(repo: Any, run_hash: str) -> None:
    """Populate the meta index for ``run_hash`` so its writes are visible.

    Mirrors astrolabe's sync sidecar sequence. Idempotent — after the
    first call for a hash, subsequent calls are near no-op.
    """
    from aim.sdk.index_manager import RepoIndexManager

    try:
        repo.request_props(
            run_hash,
            read_only=False,
            created_at=datetime.datetime.now(datetime.UTC),
        )
    except Exception:
        # "already exists" is the common case — harmless
        pass
    try:
        RepoIndexManager.get_index_manager(repo).index(run_hash)
    except Exception:
        pass


def _get_run(repo_path: Path, run_hash: str) -> Any:
    repo = _open_repo(repo_path)
    _index_run(repo, run_hash)
    run = repo.get_run(run_hash)
    if run is None:
        raise AssertionError(f"run {run_hash!r} not found in {repo_path}")
    return run


def assert_run_name(repo_path: Path, run_hash: str, expected_name: str) -> None:
    """Assert that ``repo_path``'s run ``run_hash`` has ``run.name == expected_name``.

    Non-obvious contract: reads ``run.name`` not ``run.props.name`` — after
    schema-finalize, callers hit a subtle attribute vs. property fallback
    that changes across Aim versions. Helper canonicalizes to the value the
    dashboard actually renders.
    """
    run = _get_run(repo_path, run_hash)
    actual = run.name
    if actual != expected_name:
        raise AssertionError(
            f"run {run_hash!r}: expected name={expected_name!r}, got {actual!r}"
        )


def assert_run_tag(repo_path: Path, run_hash: str, tag: str, expected_value: str) -> None:
    """Assert that ``run[tag] == expected_value``. Raises if tag missing."""
    run = _get_run(repo_path, run_hash)
    actual = run.get(tag, default=None)
    if actual is None:
        raise AssertionError(
            f"run {run_hash!r}: tag {tag!r} missing (present: {list(run.dataframe().columns)})"
        )
    if actual != expected_value:
        raise AssertionError(
            f"run {run_hash!r}: tag {tag!r}: expected {expected_value!r}, got {actual!r}"
        )


def assert_metric_landed(repo_path: Path, run_hash: str, metric_name: str) -> None:
    """Assert that at least one value has been written under ``metric_name``.

    Passes iff the metric appears in the run's metrics enumeration AND its
    series has at least one (step, value) point that survived memtable flush.
    """
    series = get_metric_series(repo_path, run_hash, metric_name)
    if not series:
        raise AssertionError(
            f"run {run_hash!r}: metric {metric_name!r} has no values (or not enumerated)"
        )


def assert_metric_count(
    repo_path: Path,
    run_hash: str,
    metric_name: str,
    expected_count: int,
) -> None:
    """Assert exactly ``expected_count`` values under ``metric_name``."""
    series = get_metric_series(repo_path, run_hash, metric_name)
    if len(series) != expected_count:
        raise AssertionError(
            f"run {run_hash!r}: metric {metric_name!r}: expected {expected_count} values, got {len(series)}"
        )


def assert_metric_values(
    repo_path: Path,
    run_hash: str,
    metric_name: str,
    expected: list[tuple[int, float]],
) -> None:
    """Assert (step, value) pairs match ``expected`` exactly, in order."""
    actual = get_metric_series(repo_path, run_hash, metric_name)
    if actual != expected:
        raise AssertionError(
            f"run {run_hash!r}: metric {metric_name!r}:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )


def assert_schema_finalized_event(
    stats_jsonl_path: Path,
    expected_metric_names: list[str] | None = None,
) -> None:
    """Assert the callback's stats jsonl contains a ``schema_finalized`` event.

    If ``expected_metric_names`` is provided, verifies the event's
    metric_names field is a superset. Used by schema-finalize scenarios
    in test_core.py to verify the finalize side-band emitted correctly.
    """
    events = _read_stats(stats_jsonl_path)
    finalize_events = [e for e in events if e.get("kind") == "schema_finalized"]
    if not finalize_events:
        raise AssertionError(
            f"stats jsonl {stats_jsonl_path} has no schema_finalized event "
            f"(events seen: {[e.get('kind') for e in events]})"
        )
    if expected_metric_names is not None:
        seen_names = set()
        for event in finalize_events:
            seen_names.update(event.get("new_metric_names", []))
        missing = set(expected_metric_names) - seen_names
        if missing:
            raise AssertionError(
                f"schema_finalized events missing expected metric names: {missing} "
                f"(saw: {sorted(seen_names)})"
            )


def assert_run_closed(repo_path: Path, run_hash: str) -> None:
    """Assert the run is closed (has a non-null ``end_time``)."""
    run = _get_run(repo_path, run_hash)
    end_time = getattr(run, "end_time", None)
    if end_time is None:
        raise AssertionError(f"run {run_hash!r}: end_time is None (run not closed)")


def assert_no_run_exists(repo_path: Path, run_hash: str) -> None:
    """Assert no run with the given hash exists in the repo."""
    repo = _open_repo(repo_path)
    run = repo.get_run(run_hash)
    if run is not None:
        raise AssertionError(f"run {run_hash!r} unexpectedly exists in {repo_path}")


def get_run_tags(repo_path: Path, run_hash: str) -> dict[str, str]:
    """Return all tags on the run as a dict. Used for tag-fidelity checks."""
    run = _get_run(repo_path, run_hash)
    # aim.Run exposes tags via .props / .dataframe(); iterate to build a dict.
    tags: dict[str, str] = {}
    for key in run.dataframe().columns:
        value = run.get(key, default=None)
        if value is not None and isinstance(value, str):
            tags[key] = value
    return tags


def get_metric_series(
    repo_path: Path,
    run_hash: str,
    metric_name: str,
) -> list[tuple[int, float]]:
    """Return the (step, value) pairs for ``metric_name`` in order.

    Returns [] if the metric doesn't exist on the run (rather than
    raising) — that shape is what memtable-invisibility scenarios rely
    on to distinguish "not-yet-flushed" from "hard error."

    Uses the metric's dataframe to extract steps + values — the
    ``.steps`` / ``.values`` TreeArrayView attributes on Aim's
    ``SequenceV2Data`` don't survive ``list()`` reliably; the dataframe
    is the tested read path.
    """
    repo = _open_repo(repo_path)
    _index_run(repo, run_hash)
    run = repo.get_run(run_hash)
    if run is None:
        return []
    metric = None
    for seq in run.metrics():
        if seq.name == metric_name:
            metric = seq
            break
    if metric is None:
        return []
    df = metric.dataframe()
    return [(int(row.step), float(row.value)) for row in df.itertuples()]


def _read_stats(stats_jsonl_path: Path) -> list[dict]:
    """Read + parse the stats jsonl. Returns [] if the file doesn't exist."""
    if not stats_jsonl_path.exists():
        return []
    events: list[dict] = []
    with stats_jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events
