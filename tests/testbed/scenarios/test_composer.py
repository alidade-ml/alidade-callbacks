"""Integration tests for ``src/astrolabe_callbacks/composer.py``.

Composer's ``AstrolabeComposerLogger`` translates the Composer trainer's
lifecycle events (batch_end, epoch_end, close, etc.) into
AstrolabeLogger calls. This file verifies the adapter drives the
underlying Logger correctly against a real Aim server. Cross-framework
parity — that all frameworks produce the same series — lives in
test_core.py because that's a Logger property, not an adapter one.

Class names (`TestTraining` / `TestValidation` / `TestTeardown`) are
consistent across all framework test files. Framework-specific quirks,
if any, go in dedicated classes named for the quirk.

Skips if the ``composer`` extra is not installed.
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


def _composer_config(
    testbed: "TestbedHandle",
    stats_path: Path,
    **overrides,
) -> DriverConfig:
    defaults = dict(
        framework="composer",
        steps=5,
        metrics_per_step=1,
        metrics_per_sec=0.0,
        fail_at=None,
        new_metrics_at=[],
        validation_at=[],
        close=True,
        aim_url=testbed.aim_url_from_client,
        run_name="composer-probe",
        experiment_name="testbed-composer",
        tags={},
        driver_flags={},
        stats_jsonl_container_path=f"/host-stats/{stats_path.parent.name}/{stats_path.name}",
    )
    defaults.update(overrides)
    return DriverConfig(**defaults)


RunFixture = Callable[[DriverConfig], DriverResult]


class TestTraining:
    """Training-step metric emission via Composer's batch_end hook."""

    def test_batch_end_emits_metric(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """batch_end call translates Composer's Logger.log_metrics into an Aim track."""
        pytest.importorskip("composer")
        result = run_driver(_composer_config(testbed, stats_jsonl_path, steps=3))
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_count(aim_repo, result.run_hash, "metric_0", 3)

    def test_epoch_end_flushes_new_metrics(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """New metrics that appeared during the epoch are flushed at epoch boundary."""
        pytest.importorskip("composer")
        # Driver structures training as multiple epochs; new metric introduced mid-epoch
        result = run_driver(
            _composer_config(
                testbed,
                stats_jsonl_path,
                steps=6,
                new_metrics_at=[3],
                driver_flags={"TESTBED_COMPOSER_EPOCHS": "2"},  # 3 steps per epoch
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        # After the mid-epoch new metric + epoch_end, the metric is readable
        assert_metric_landed(aim_repo, result.run_hash, "metric_new_step3")


class TestValidation:
    """Validation metrics routed through Composer's eval hooks."""

    def test_eval_batch_end_emits_val_metric(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """eval_batch_end call lands as val/* metric on the training run."""
        pytest.importorskip("composer")
        result = run_driver(
            _composer_config(testbed, stats_jsonl_path, steps=5, validation_at=[2, 4])
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        # Two validation emits — landed under val/ prefix on the training run
        assert_metric_count(aim_repo, result.run_hash, "val/loss", 2)


class TestTeardown:
    """Composer's close hook cleanly finalizes the Aim run."""

    def test_close_hook_sets_end_time(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """After Composer close, Aim run has end_time set."""
        pytest.importorskip("composer")
        result = run_driver(_composer_config(testbed, stats_jsonl_path))
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_run_closed(aim_repo, result.run_hash)

    def test_close_hook_drains_buffer(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Metrics pending in buffer at close time still land."""
        pytest.importorskip("composer")
        result = run_driver(
            _composer_config(
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
