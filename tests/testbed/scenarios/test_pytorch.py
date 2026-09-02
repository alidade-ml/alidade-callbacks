"""Integration tests for ``src/alidade_callbacks/pytorch.py``.

Raw PyTorch helper — no framework Trainer, users drive AlidadeLogger
directly via ``AlidadeRun`` (aliased ``Run``). This file verifies
the direct-usage surface works against a real Aim server.

Class names (`TestTraining` / `TestTeardown`) are consistent with the
other framework test files. `TestContextManager` is PyTorch-specific —
no other adapter uses a context-manager entry point, so it stays as a
distinct class rather than being flattened into `TestTeardown`.

No framework extra required — pytorch is always available.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from tests.testbed.harness.assertions import (
    assert_metric_count,
    assert_metric_landed,
    assert_run_closed,
)
from tests.testbed.harness.driver import DriverConfig, DriverResult

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


def _pytorch_config(
    testbed: "TestbedHandle",
    stats_path: Path,
    **overrides,
) -> DriverConfig:
    defaults = dict(
        framework="raw",  # PyTorch adapter is the raw path
        steps=5,
        metrics_per_step=1,
        metrics_per_sec=0.0,
        fail_at=None,
        new_metrics_at=[],
        validation_at=[],
        close=True,
        aim_url=testbed.aim_url_from_client,
        run_name="pytorch-probe",
        experiment_name="testbed-pytorch",
        tags={},
        driver_flags={},
        stats_jsonl_container_path=f"/host-stats/{stats_path.parent.name}/{stats_path.name}",
    )
    defaults.update(overrides)
    return DriverConfig(**defaults)


RunFixture = Callable[[DriverConfig], DriverResult]


class TestTraining:
    """Direct AlidadeRun.track() calls emit metrics."""

    def test_direct_track_emits_metric(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        result = run_driver(_pytorch_config(testbed, stats_jsonl_path, steps=4))
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_count(aim_repo, result.run_hash, "metric_0", 4)


class TestTeardown:
    """Explicit close() finalizes the Aim run cleanly."""

    def test_close_sets_end_time(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        result = run_driver(_pytorch_config(testbed, stats_jsonl_path))
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_run_closed(aim_repo, result.run_hash)

    def test_close_drains_buffer(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Metrics pending in buffer at close still land."""
        result = run_driver(
            _pytorch_config(
                testbed,
                stats_jsonl_path,
                steps=30,
                metrics_per_step=3,
                driver_flags={"TESTBED_STRESS_BUFFER": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        for i in range(3):
            assert_metric_count(aim_repo, result.run_hash, f"metric_{i}", 30)


class TestContextManager:
    """Framework-specific: ``with AlidadeRun(...) as run`` semantics.

    Kept distinct from `TestTeardown` because the context-manager surface
    is a PyTorch-only entry point (other adapters drive the callback
    through their framework's lifecycle, not a with-block).
    """

    def test_context_manager_closes_on_normal_exit(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Context exit closes the run and drains the buffer."""
        result = run_driver(
            _pytorch_config(
                testbed,
                stats_jsonl_path,
                steps=3,
                driver_flags={"TESTBED_USE_CONTEXT_MANAGER": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_run_closed(aim_repo, result.run_hash)
        assert_metric_landed(aim_repo, result.run_hash, "metric_0")

    def test_context_manager_closes_on_exception(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Exception inside the with-block still closes the run cleanly."""
        result = run_driver(
            _pytorch_config(
                testbed,
                stats_jsonl_path,
                steps=5,
                fail_at=2,
                driver_flags={"TESTBED_USE_CONTEXT_MANAGER": "1"},
            )
        )
        # Driver raises SimulatedFailure inside the with-block; context
        # manager cleans up. Exit code is 42 (SimulatedFailure sentinel),
        # not 0 — but the Aim run should still be closed.
        assert result.exit_code == 42, result.stderr
        assert result.run_hash is not None
        assert_run_closed(aim_repo, result.run_hash)
