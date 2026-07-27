"""Integration tests for ``src/astrolabe_callbacks/huggingface.py``.

HF Trainer's ``AstrolabeHFTrainerCallback`` translates HuggingFace
Trainer lifecycle events (on_log, on_evaluate, on_train_end, ...) into
AstrolabeLogger calls. This file verifies the adapter drives the
underlying Logger correctly against a real Aim server.

Class names (`TestTraining` / `TestValidation` / `TestTeardown`) are
consistent across all framework test files.

Skips if the ``hf`` extra is not installed.
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


def _hf_config(
    testbed: "TestbedHandle",
    stats_path: Path,
    **overrides,
) -> DriverConfig:
    defaults = dict(
        framework="hf",
        steps=5,
        metrics_per_step=1,
        metrics_per_sec=0.0,
        fail_at=None,
        new_metrics_at=[],
        validation_at=[],
        close=True,
        aim_url=testbed.aim_url_from_client,
        run_name="hf-probe",
        experiment_name="testbed-hf",
        tags={},
        driver_flags={},
        stats_jsonl_container_path=f"/host-stats/{stats_path.parent.name}/{stats_path.name}",
    )
    defaults.update(overrides)
    return DriverConfig(**defaults)


RunFixture = Callable[[DriverConfig], DriverResult]


@pytest.mark.skip(reason="testbed-todo: HF driver doesn't explicitly log ``metric_0``; only HF's built-in ``loss`` is emitted.")
class TestTraining:
    """Training-step metric emission via HF's on_log hook."""

    def test_on_log_emits_metric(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        pytest.importorskip("transformers")
        result = run_driver(_hf_config(testbed, stats_jsonl_path, steps=4))
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_count(aim_repo, result.run_hash, "metric_0", 4)


@pytest.mark.skip(reason="testbed-todo: HF driver doesn't wire an eval dataset; on_evaluate never fires.")
class TestValidation:
    """Validation metrics routed through HF's on_evaluate hook."""

    def test_on_evaluate_emits_val_metric(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        pytest.importorskip("transformers")
        result = run_driver(
            _hf_config(testbed, stats_jsonl_path, steps=5, validation_at=[2, 4])
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_count(aim_repo, result.run_hash, "val/loss", 2)


class TestTeardown:
    """HF's on_train_end hook cleanly finalizes the Aim run."""

    def test_on_train_end_sets_end_time(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        pytest.importorskip("transformers")
        result = run_driver(_hf_config(testbed, stats_jsonl_path))
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_run_closed(aim_repo, result.run_hash)

    @pytest.mark.skip(reason="testbed-todo: HF driver doesn't emit multiple named metrics; scenario expects raw-driver shape.")
    def test_on_train_end_drains_buffer(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """Metrics pending in buffer at on_train_end still land."""
        pytest.importorskip("transformers")
        result = run_driver(
            _hf_config(
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
