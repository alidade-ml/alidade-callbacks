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

import pytest

if False:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = [pytest.mark.testbed, pytest.mark.testbed_scale]


class TestBurstThroughput:
    """5-minute sustained bursts at increasing rates."""

    @pytest.mark.parametrize("rate", [100, 1000, 5000])
    def test_five_minute_burst(
        self, testbed: "TestbedHandle", aim_repo: Path, rate: int
    ) -> None:
        """Emit ``rate`` writes/sec for 5 min; verify writes-in ≈ writes-landed within tolerance."""
        raise NotImplementedError


class TestLongDuration:
    """Longer windows to catch slow-burn bugs."""

    def test_one_hour_at_1000_per_sec(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """1000 writes/sec × 1hr. Catches memory-leak-per-write and drainer starvation."""
        raise NotImplementedError

    def test_four_hour_at_100_per_sec(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """100 writes/sec × 4hr. Catches file-handle leaks + RocksDB compaction backpressure."""
        raise NotImplementedError


class TestResourceInvariants:
    """Process-level invariants during sustained load."""

    def test_no_unbounded_memory_growth(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """RSS growth over the run stays within an expected envelope."""
        raise NotImplementedError

    def test_no_file_handle_leak(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """/proc/<pid>/fd count returns to baseline after close()."""
        raise NotImplementedError

    def test_no_thread_leak(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Thread count returns to baseline after close(); drainer joined."""
        raise NotImplementedError
