"""Integration tests for ``src/astrolabe_callbacks/lightning.py``.

Lightning's ``AstrolabeLightningLogger`` translates PyTorch
Lightning's on_* hooks into AstrolabeLogger calls. This file verifies
the adapter drives the underlying Logger correctly against a real Aim
server.

Class names (`TestTraining` / `TestValidation` / `TestTeardown`) are
consistent across all framework test files.

Skips if the ``lightning`` extra is not installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from tests.testbed.harness.assertions import assert_metric_count, assert_run_closed
from tests.testbed.harness.driver import DriverConfig, DriverResult

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


def _lightning_config(
    testbed: "TestbedHandle",
    stats_path: Path,
    **overrides,
) -> DriverConfig:
    defaults = dict(
        framework="lightning",
        steps=5,
        metrics_per_step=1,
        metrics_per_sec=0.0,
        fail_at=None,
        new_metrics_at=[],
        validation_at=[],
        close=True,
        aim_url=testbed.aim_url_from_client,
        run_name="lightning-probe",
        experiment_name="testbed-lightning",
        tags={},
        driver_flags={},
        stats_jsonl_container_path=f"/host-stats/{stats_path.parent.name}/{stats_path.name}",
    )
    defaults.update(overrides)
    return DriverConfig(**defaults)


RunFixture = Callable[[DriverConfig], DriverResult]


class TestTraining:
    """Training-step metric emission via Lightning's on_train_batch_end hook."""

    def test_on_train_batch_end_emits_metric(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        pytest.importorskip("lightning")
        result = run_driver(_lightning_config(testbed, stats_jsonl_path, steps=4))
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_count(aim_repo, result.run_hash, "metric_0", 4)


class TestValidation:
    """Validation metrics routed through Lightning's on_validation_end hook."""

    def test_on_validation_end_emits_val_metric(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """val/* metrics land on the same run as training, not a separate eval run."""
        pytest.importorskip("lightning")
        result = run_driver(
            _lightning_config(testbed, stats_jsonl_path, steps=5, validation_at=[2, 4])
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_count(aim_repo, result.run_hash, "val/loss", 2)


class TestTeardown:
    """Lightning's teardown hook cleanly finalizes the Aim run."""

    def test_teardown_sets_end_time(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        pytest.importorskip("lightning")
        result = run_driver(_lightning_config(testbed, stats_jsonl_path))
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_run_closed(aim_repo, result.run_hash)

    def test_teardown_drains_buffer(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Metrics pending in buffer at teardown time still land."""
        pytest.importorskip("lightning")
        result = run_driver(
            _lightning_config(
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
