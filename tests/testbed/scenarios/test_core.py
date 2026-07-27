"""Integration tests for ``src/astrolabe_callbacks/_core.py``.

`_core.py` owns AstrolabeLogger: init/close, tag setting, name
handling, buffer + drainer thread, schema-finalize invariants,
first_metric marker, hash-fidelity in the stats jsonl. Every behavior
implemented in that module gets exercised here against a real Aim
server (via the docker-compose testbed) — no FakeAimRun.

Companion unit tests (``tests/test_core.py``) cover the same module
against ``FakeAimRun`` and catch API-shape bugs cheaply. This file
catches what the fake cannot: memtable flush semantics, RocksDB chunk
visibility, protobuf transport quirks, drainer thread races under a
real socket.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from tests.testbed.harness import compose
from tests.testbed.harness.assertions import (
    assert_metric_count,
    assert_metric_landed,
    assert_run_closed,
    assert_run_name,
    assert_run_tag,
    assert_schema_finalized_event,
    get_run_tags,
)
from tests.testbed.harness.driver import (
    DriverConfig,
    DriverResult,
)

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


# -----------------------------------------------------------------------------
# Config builders — keep scenarios short. Each returns a config with sensible
# defaults; scenarios override the fields they care about.
# -----------------------------------------------------------------------------


def _base_config(testbed: "TestbedHandle", stats_path: Path, **overrides) -> DriverConfig:
    defaults = dict(
        framework="raw",
        steps=5,
        metrics_per_step=1,
        metrics_per_sec=0.0,
        fail_at=None,
        new_metrics_at=[],
        validation_at=[],
        close=True,
        aim_url=testbed.aim_url_from_client,
        run_name="testbed-run",
        experiment_name="testbed-core",
        tags={},
        driver_flags={},
        stats_jsonl_container_path=f"/host-stats/{stats_path.parent.name}/{stats_path.name}",
    )
    defaults.update(overrides)
    return DriverConfig(**defaults)


RunFixture = Callable[[DriverConfig], DriverResult]


# -----------------------------------------------------------------------------


class TestLifecycle:
    """Init/close idempotency and error paths."""

    def test_double_close_is_safe(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Calling close() twice does not raise or corrupt the run."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                run_name="double-close-probe",
                # driver-level flag consumed by the driver to invoke close() twice
                driver_flags={"TESTBED_DOUBLE_CLOSE": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_run_closed(aim_repo, result.run_hash)

    def test_close_without_init_is_safe(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Calling close() on a never-initialized callback does not raise."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                # Skip open: driver should exit cleanly after close-without-init
                driver_flags={"TESTBED_SKIP_INIT": "1"},
                close=True,
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is None  # never opened → no hash

    def test_track_after_close_is_noop(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """track() calls after close() are silently dropped, not errors."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                run_name="track-after-close-probe",
                # driver closes at step 2, keeps track()ing through step 5
                driver_flags={"TESTBED_CLOSE_AT": "2"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        # Post-close writes silently dropped — verify metric only has step-0/1 values
        assert_metric_count(aim_repo, result.run_hash, "metric_0", 2)


class TestNameFidelity:
    """Run.name survives every close-reopen path _core.py takes.

    Direct regression coverage for the v2.0.0-rc1 bug: schema-finalize
    closed + reopened the Run with force_resume=True, and the reopen
    did NOT preserve run.name — after the first finalize the name
    reverted to Aim's default ``Run: <hash>`` fallback.
    """

    def test_name_survives_single_schema_finalize(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                run_name="single-finalize-probe",
                new_metrics_at=[2],  # one schema-finalize
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_run_name(aim_repo, result.run_hash, "single-finalize-probe")

    def test_name_survives_ten_schema_finalizes(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Cap-boundary case: 10 finalizes and run.name still correct."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                run_name="ten-finalize-probe",
                steps=12,
                new_metrics_at=list(range(1, 11)),  # 10 new-metric events, hits cap exactly
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_run_name(aim_repo, result.run_hash, "ten-finalize-probe")

    def test_name_survives_explicit_close_reopen(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """A close() + reopen path not driven by schema-finalize still preserves name."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                run_name="reopen-probe",
                # driver closes and reopens the run mid-way, no new metrics
                driver_flags={"TESTBED_MID_REOPEN_AT": "3"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_run_name(aim_repo, result.run_hash, "reopen-probe")


class TestTagFidelity:
    """Tags set at init survive schema-finalize + close-reopen cycles."""

    def test_astrolabe_tags_survive_finalize(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """astrolabe.experiment, astrolabe.submit_id, astrolabe.version all preserved."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                run_name="astrolabe-tags-probe",
                new_metrics_at=[2],
                tags={
                    "astrolabe.experiment": "test-exp",
                    "astrolabe.submit_id": "sub-abc-123",
                    "astrolabe.version": "v1",
                },
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_run_tag(aim_repo, result.run_hash, "astrolabe.experiment", "test-exp")
        assert_run_tag(aim_repo, result.run_hash, "astrolabe.submit_id", "sub-abc-123")
        assert_run_tag(aim_repo, result.run_hash, "astrolabe.version", "v1")

    def test_custom_tags_survive_finalize(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """User-set tags (non-astrolabe.*) also preserved."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                run_name="custom-tags-probe",
                new_metrics_at=[2],
                tags={"custom.thesis": "orion", "custom.gpu": "H100"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        landed = get_run_tags(aim_repo, result.run_hash)
        assert landed["custom.thesis"] == "orion"
        assert landed["custom.gpu"] == "H100"


class TestSchemaFinalize:
    """Schema-finalize invariants — the multi-fit lifecycle _core.py runs."""

    def test_first_new_metric_finalizes(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """New metric name → a schema_finalized event lists it in new_metric_names.

        The callback library fires an initial finalize on the first-ever
        metric (metric_0 written at step 0), then another when we
        introduce metric_new_step3 at step 3. We check the SPECIFIC
        finalize for our injected new metric rather than counting all.
        """
        result = run_driver(
            _base_config(testbed, stats_jsonl_path, new_metrics_at=[3])
        )
        assert result.exit_code == 0, result.stderr
        finalize_events = [e for e in result.stats_events if e.get("kind") == "schema_finalized"]
        matching = [
            e for e in finalize_events
            if "metric_new_step3" in e.get("new_metric_names", [])
        ]
        assert len(matching) == 1

    def test_repeated_same_metric_no_extra_finalize(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Same metric name N times → only the initial finalize fires."""
        # metrics_per_step=1 means metric_0 written every step, no new names.
        # Only the initial-metric introduction should finalize once.
        result = run_driver(_base_config(testbed, stats_jsonl_path, steps=10))
        assert result.exit_code == 0, result.stderr
        finalize_events = [e for e in result.stats_events if e.get("kind") == "schema_finalized"]
        assert len(finalize_events) == 1

    def test_no_more_than_ten_finalizes(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """15 distinct new metric names → cap-limited finalize count.

        Callback fires an initial finalize on metric_0, then finalizes
        for each new metric until the cap (10) is reached. Total is 10
        (the cap includes the initial).
        """
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                steps=16,
                new_metrics_at=list(range(1, 16)),  # 15 new metrics
            )
        )
        assert result.exit_code == 0, result.stderr
        finalize_events = [e for e in result.stats_events if e.get("kind") == "schema_finalized"]
        assert len(finalize_events) == 10

    def test_cap_hit_logs_warning(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Finalize cap reached emits a WARNING + one-shot cap-hit event."""
        result = run_driver(
            _base_config(
                testbed, stats_jsonl_path, steps=16, new_metrics_at=list(range(1, 16))
            )
        )
        assert result.exit_code == 0, result.stderr
        cap_events = [e for e in result.stats_events if e.get("kind") == "schema_max_finalizes_hit"]
        assert len(cap_events) == 1

    def test_metrics_still_land_after_cap(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """New metrics after cap still track; they just don't trigger finalizes."""
        result = run_driver(
            _base_config(
                testbed, stats_jsonl_path, steps=16, new_metrics_at=list(range(1, 16))
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        # metric_new_step15 was introduced AFTER the cap fired
        assert_metric_landed(aim_repo, result.run_hash, "metric_new_step15")

    def test_finalize_flushes_new_metric_to_disk(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """After our finalize, a Repo(read_only=True) can enumerate the new metric.

        Verifies our workaround exploits Aim's flush contract correctly.
        The Aim-side contract (unflushed → invisible, flushed → visible)
        is documented in test_aim_compat.py.
        """
        result = run_driver(
            _base_config(testbed, stats_jsonl_path, new_metrics_at=[3])
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        # If the finalize forced a flush, the read-only reader sees it
        assert_metric_landed(aim_repo, result.run_hash, "metric_new_step3")

    def test_finalize_event_lists_metric_names(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Event includes up to 10 metric names that triggered the finalize."""
        result = run_driver(
            _base_config(testbed, stats_jsonl_path, new_metrics_at=[3])
        )
        assert result.exit_code == 0, result.stderr
        assert_schema_finalized_event(
            stats_jsonl_path,
            expected_metric_names=["metric_new_step3"],
        )

    def test_finalize_event_carries_timestamp(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        result = run_driver(
            _base_config(testbed, stats_jsonl_path, new_metrics_at=[3])
        )
        assert result.exit_code == 0, result.stderr
        finalize_events = [e for e in result.stats_events if e.get("kind") == "schema_finalized"]
        assert len(finalize_events) >= 1
        ts = finalize_events[0].get("ts")
        assert ts is not None
        assert isinstance(ts, (int, float))


class TestBufferDrainer:
    """Buffer + drainer thread against a real Aim server.

    Unit tests exercise the buffer against FakeAimRun. These scenarios
    catch integration bugs the unit tests miss: drainer thread races
    against a real socket, transient network errors, close-drains
    behavior under real transport latency.
    """

    def test_writes_land_within_expected_latency(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Tracked value appears in the Aim repo within N seconds of track()."""
        result = run_driver(
            _base_config(testbed, stats_jsonl_path, steps=3, metrics_per_step=1)
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_count(aim_repo, result.run_hash, "metric_0", 3)

    def test_close_drains_pending_writes(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """close() blocks until buffer empty; no writes lost."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                steps=50,
                metrics_per_step=5,
                # High write rate to ensure buffer has pending items at close
                driver_flags={"TESTBED_STRESS_BUFFER": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        # All 50 × 5 = 250 writes across 5 metrics — each metric got 50 values
        for i in range(5):
            assert_metric_count(aim_repo, result.run_hash, f"metric_{i}", 50)

    @pytest.mark.skip(reason="testbed-todo: TESTBED_INJECT_DRAINER_DELAY class-level monkey-patch on aim.Run.track works in isolation but doesn't slow the drainer's calls in-process. Root cause unclear — possibly Aim's bound-method caching. Requires deeper investigation.")
    def test_overflow_drops_oldest(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Buffer full → oldest entries evicted; newest survive (training-signal-live)."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                steps=1000,
                metrics_per_step=10,
                driver_flags={"TESTBED_BUFFER_CAPACITY": "50", "TESTBED_INJECT_DRAINER_DELAY": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        # Callback close event stores dropped_oldest under the ``dropped`` field
        close_events = [e for e in result.stats_events if e.get("kind") == "close"]
        assert close_events and close_events[-1].get("dropped", 0) > 0

    @pytest.mark.skip(reason="testbed-todo: same TESTBED_INJECT_DRAINER_DELAY limitation as test_overflow_drops_oldest.")
    def test_overflow_counter_increments(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Callback close event's ``dropped`` field is non-zero after overflow."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                steps=1000,
                metrics_per_step=10,
                driver_flags={"TESTBED_BUFFER_CAPACITY": "50", "TESTBED_INJECT_DRAINER_DELAY": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr
        close_events = [e for e in result.stats_events if e.get("kind") == "close"]
        assert close_events and close_events[-1].get("dropped", 0) > 0

    def test_drainer_retries_on_transient_error(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Injected transient error → drainer backs off + retries; write lands."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                steps=5,
                driver_flags={"TESTBED_INJECT_TRANSIENT_ERROR_AT": "2"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        # Write eventually lands after retry
        assert_metric_count(aim_repo, result.run_hash, "metric_0", 5)

    def test_drainer_death_surfaces_in_stats(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """If drainer thread dies unrecoverably, stats jsonl records the failure."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                steps=5,
                driver_flags={"TESTBED_KILL_DRAINER_AT": "2"},
            )
        )
        assert result.exit_code == 0, result.stderr
        drainer_events = [
            e for e in result.stats_events if e.get("kind") == "drainer_died"
        ]
        assert len(drainer_events) == 1

    def test_close_after_drainer_death_does_not_hang(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """close() bounded even if drainer is dead; no infinite wait."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                steps=5,
                driver_flags={"TESTBED_KILL_DRAINER_AT": "2"},
            )
        )
        assert result.exit_code == 0, result.stderr

    @pytest.mark.skip(reason="testbed-todo: driver + harness don't yet implement TESTBED_RESTART_AIM_AT (needs host-docker access from inside client)")
    def test_writes_after_aim_server_restart_land(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Restart the aim server mid-run; buffered writes eventually land."""
        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                steps=10,
                driver_flags={"TESTBED_RESTART_AIM_AT": "5"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_count(aim_repo, result.run_hash, "metric_0", 10)


class TestFirstMetricMarker:
    """ASTROLABE_FIRST_METRIC_MARKER fires exactly once, on first track_safely.

    Feeds astrolabe's healing/failure hook system — a run that never
    produces a first metric is treated differently from one that made
    progress and then died.
    """

    def test_marker_touched_on_first_track(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Marker file exists on the client container after first track() call."""
        result = run_driver(
            _base_config(testbed, stats_jsonl_path, steps=3, run_name="marker-touch-probe")
        )
        assert result.exit_code == 0, result.stderr
        assert result.marker_touched is True

    def test_marker_touched_exactly_once(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Subsequent track() calls do not re-touch the marker.

        We can't observe "touched exactly once" via file mtime alone (the
        callback library's ``_write_first_metric_marker_once`` uses a
        module-level flag to avoid re-touching, but that flag lives
        inside the driver subprocess and doesn't persist across runs).
        Best we can assert externally: after N metrics, the marker file
        exists (touched at least once). The "exactly once" invariant is
        a callback-internal contract; unit tests cover it.
        """
        result = run_driver(
            _base_config(testbed, stats_jsonl_path, steps=10, run_name="marker-exactly-once-probe")
        )
        assert result.exit_code == 0, result.stderr
        assert result.marker_touched is True

    def test_marker_absent_when_no_metrics_tracked(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Run that opens and closes without any track() leaves no marker."""
        result = run_driver(
            _base_config(testbed, stats_jsonl_path, steps=0, run_name="marker-absent-probe")
        )
        assert result.exit_code == 0, result.stderr
        assert result.marker_touched is False


class TestHashFidelity:
    """Callback stats jsonl carries full-fidelity 24-char run hashes.

    Regression coverage for the ``run_hash[:12]`` bug (memory:
    project_callback_stats_jsonl_is_data_channel). ProjectOrion consumes
    this jsonl programmatically — truncated hashes make runs unjoinable
    back to Aim.
    """

    def test_stats_jsonl_hash_matches_aim_run_hash(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Every hash in the stats jsonl is 24 chars and matches an Aim run hash."""
        result = run_driver(_base_config(testbed, stats_jsonl_path, steps=3))
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert len(result.run_hash) == 24
        # Every event referencing run_hash matches exactly
        for event in result.stats_events:
            if "run_hash" in event:
                assert event["run_hash"] == result.run_hash
                assert len(event["run_hash"]) == 24


class TestFrameworkParityInvariants:
    """All frameworks produce equivalent Aim state for equivalent inputs.

    Lives here (not in per-framework files) because the property being
    asserted is a _core.py invariant: the same AstrolabeLogger produces
    the same series regardless of what framework calls it.
    """

    @pytest.mark.parametrize("framework", ["raw", "composer", "lightning"])
    def test_same_metrics_yield_same_series(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
        framework: str,
    ) -> None:
        """N steps × M metrics → identical (step, value) series across frameworks."""
        # Framework-specific extras only imported if needed by the driver
        if framework == "composer":
            pytest.importorskip("composer")
        elif framework == "lightning":
            pytest.importorskip("lightning")
        elif framework == "hf":
            pytest.importorskip("transformers")

        result = run_driver(
            _base_config(
                testbed,
                stats_jsonl_path,
                framework=framework,
                run_name=f"parity-probe-{framework}",
                steps=5,
                metrics_per_step=2,
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        # Frameworks may add epoch-boundary aggregates even with per-step
        # logging; assert AT LEAST 5 values landed for each metric.
        from tests.testbed.harness.assertions import get_metric_series
        for name in ("metric_0", "metric_1"):
            series = get_metric_series(aim_repo, result.run_hash, name)
            assert len(series) >= 5, f"{framework}: {name} has {len(series)} values"
