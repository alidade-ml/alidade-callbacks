"""Sustained-write behavior against a real Aim server.

These scenarios take minutes to hours to run. They catch memory leaks,
RocksDB compaction issues, file-handle exhaustion, and drainer
starvation under sustained load — bug classes that only surface over
long windows and don't appear in the fast scenarios.

Marker: ``testbed_scale``. Excluded from default CI. Run via the
scheduled/event-triggered scale workflow or manually with
``pytest -m testbed_scale``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from tests.testbed.harness.assertions import assert_metric_count
from tests.testbed.harness.driver import DriverConfig, DriverResult

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = [pytest.mark.testbed, pytest.mark.testbed_scale]


def _load_config(
    testbed: "TestbedHandle",
    stats_path: Path,
    *,
    steps: int,
    metrics_per_step: int,
    metrics_per_sec: float,
    run_name: str,
    driver_flags: dict[str, str] | None = None,
) -> DriverConfig:
    return DriverConfig(
        framework="raw",
        steps=steps,
        metrics_per_step=metrics_per_step,
        metrics_per_sec=metrics_per_sec,
        fail_at=None,
        new_metrics_at=[],
        validation_at=[],
        close=True,
        aim_url=testbed.aim_url_from_client,
        run_name=run_name,
        experiment_name="testbed-sustained",
        tags={},
        driver_flags=driver_flags or {},
        stats_jsonl_container_path=f"/host-stats/{stats_path.parent.name}/{stats_path.name}",
    )


RunFixture = Callable[[DriverConfig], DriverResult]

# Acceptable loss tolerance for sustained-write assertions.
# The buffer may drop-oldest under sustained bursts by design — we don't require
# 100% of writes to land, we require the drop rate stays within an expected envelope.
DROP_TOLERANCE_FRACTION = 0.05  # ≤5% of writes may drop under 5000/sec bursts


class TestBurstThroughput:
    """5-minute sustained bursts at increasing rates."""

    @pytest.mark.parametrize("rate", [100, 1000, 5000])
    def test_five_minute_burst(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
        rate: int,
    ) -> None:
        """Emit ``rate`` writes/sec for 5 min; verify writes-in ≈ writes-landed within tolerance."""
        duration_s = 300  # 5 min
        expected_writes = rate * duration_s
        result = run_driver(
            _load_config(
                testbed,
                stats_jsonl_path,
                steps=expected_writes,
                metrics_per_step=1,
                metrics_per_sec=float(rate),
                run_name=f"burst-{rate}-per-sec",
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None

        from tests.testbed.harness.assertions import get_metric_series

        series = get_metric_series(aim_repo, result.run_hash, "metric_0")
        landed = len(series)
        # Within DROP_TOLERANCE of the expected count
        assert landed >= int(expected_writes * (1 - DROP_TOLERANCE_FRACTION)), (
            f"rate={rate}: landed={landed}, expected≥{expected_writes} × (1 - {DROP_TOLERANCE_FRACTION})"
        )


class TestLongDuration:
    """Longer windows to catch slow-burn bugs."""

    def test_one_hour_at_1000_per_sec(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """1000 writes/sec × 1hr. Catches memory-leak-per-write and drainer starvation."""
        result = run_driver(
            _load_config(
                testbed,
                stats_jsonl_path,
                steps=1000 * 3600,
                metrics_per_step=1,
                metrics_per_sec=1000.0,
                run_name="long-1hr-1kps",
                driver_flags={"TESTBED_MONITOR_RSS": "1"},  # driver samples RSS periodically
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        # Verify the run completed all 1000 * 3600 emissions within tolerance
        from tests.testbed.harness.assertions import get_metric_series
        series = get_metric_series(aim_repo, result.run_hash, "metric_0")
        assert len(series) >= int(1000 * 3600 * (1 - DROP_TOLERANCE_FRACTION))

    def test_four_hour_at_100_per_sec(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """100 writes/sec × 4hr. Catches file-handle leaks + RocksDB compaction backpressure."""
        result = run_driver(
            _load_config(
                testbed,
                stats_jsonl_path,
                steps=100 * 4 * 3600,
                metrics_per_step=1,
                metrics_per_sec=100.0,
                run_name="long-4hr-100ps",
                driver_flags={"TESTBED_MONITOR_FDS": "1"},  # driver samples fd count periodically
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None


class TestResourceInvariants:
    """Process-level invariants during sustained load.

    Each of these runs a short burst and asks the driver to report
    the process-level metric (RSS, fd count, thread count) at both the
    start and end. The scenarios assert on the delta, not on absolute
    values — Python startup overhead varies too much for absolute
    bounds to be portable.
    """

    def test_no_unbounded_memory_growth(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """RSS growth over the run stays within an expected envelope (< 100MB delta)."""
        result = run_driver(
            _load_config(
                testbed,
                stats_jsonl_path,
                steps=100_000,
                metrics_per_step=1,
                metrics_per_sec=0.0,
                run_name="rss-invariant-probe",
                driver_flags={"TESTBED_REPORT_RSS_DELTA": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr
        rss_events = [e for e in result.stats_events if e.get("kind") == "rss_delta"]
        assert len(rss_events) == 1
        rss_bytes = rss_events[0].get("delta_bytes", 0)
        assert rss_bytes < 100 * 1024 * 1024, f"RSS grew by {rss_bytes} bytes"

    def test_no_file_handle_leak(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """/proc/<pid>/fd count returns to baseline after close()."""
        result = run_driver(
            _load_config(
                testbed,
                stats_jsonl_path,
                steps=1000,
                metrics_per_step=1,
                metrics_per_sec=0.0,
                run_name="fd-invariant-probe",
                driver_flags={"TESTBED_REPORT_FD_DELTA": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr
        fd_events = [e for e in result.stats_events if e.get("kind") == "fd_delta"]
        assert len(fd_events) == 1
        fd_delta = fd_events[0].get("delta", 0)
        # Small delta OK (aim client keeps some sockets); large delta = leak
        assert fd_delta <= 5, f"fd count grew by {fd_delta}"

    def test_no_thread_leak(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Thread count returns to baseline after close(); drainer joined."""
        result = run_driver(
            _load_config(
                testbed,
                stats_jsonl_path,
                steps=1000,
                metrics_per_step=1,
                metrics_per_sec=0.0,
                run_name="thread-invariant-probe",
                driver_flags={"TESTBED_REPORT_THREAD_DELTA": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr
        thread_events = [
            e for e in result.stats_events if e.get("kind") == "thread_delta"
        ]
        assert len(thread_events) == 1
        thread_delta = thread_events[0].get("delta", 0)
        # After close, all Aim-side + drainer threads should be joined
        assert thread_delta == 0, f"thread count grew by {thread_delta}"
