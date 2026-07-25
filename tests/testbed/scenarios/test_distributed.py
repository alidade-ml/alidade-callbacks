"""Integration tests for ``src/astrolabe_callbacks/_distributed.py``.

Rank-zero gating: distributed training with N processes must produce
ONE Aim run, not N. `_distributed.py` handles the detection order
(torch.distributed → RANK env → LOCAL_RANK env → single-process
fallback). These scenarios verify the gate holds against real
processes writing to a real Aim server.

Not requiring a distributed framework install: `_distributed.py` reads
env vars, so scenarios manipulate ``RANK`` / ``LOCAL_RANK`` /
``WORLD_SIZE`` in the client container's env to simulate distributed
launchers without spinning up an actual multi-process setup.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from tests.testbed.harness.assertions import assert_metric_landed, assert_no_run_exists
from tests.testbed.harness.driver import DriverConfig, DriverResult

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


def _rank_config(
    testbed: "TestbedHandle",
    stats_path: Path,
    *,
    rank_env: dict[str, str],
    run_name: str,
) -> DriverConfig:
    return DriverConfig(
        framework="raw",
        steps=3,
        metrics_per_step=1,
        metrics_per_sec=0.0,
        fail_at=None,
        new_metrics_at=[],
        validation_at=[],
        close=True,
        aim_url=testbed.aim_url_from_client,
        run_name=run_name,
        experiment_name="testbed-distributed",
        tags={},
        # rank_env keys (RANK, LOCAL_RANK, WORLD_SIZE) are read by the driver
        # before importing torch — see the driver's rank-detection path.
        driver_flags={f"TESTBED_ENV_{k}": v for k, v in rank_env.items()},
        stats_jsonl_container_path=f"/host-stats/{stats_path.name}",
    )


RunFixture = Callable[[DriverConfig], DriverResult]


class TestRankGating:
    """Non-zero ranks skip Aim writes."""

    def test_rank_zero_writes_land(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """RANK=0 simulates the primary process; metrics land in Aim."""
        result = run_driver(
            _rank_config(
                testbed,
                stats_jsonl_path,
                rank_env={"RANK": "0", "WORLD_SIZE": "4"},
                run_name="rank0-probe",
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_landed(aim_repo, result.run_hash, "metric_0")

    def test_rank_nonzero_skips_writes(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """RANK=1 simulates a secondary process; no Aim run is opened."""
        result = run_driver(
            _rank_config(
                testbed,
                stats_jsonl_path,
                rank_env={"RANK": "1", "WORLD_SIZE": "4"},
                run_name="rank1-probe",
            )
        )
        assert result.exit_code == 0, result.stderr
        # No run opened → no run_hash
        assert result.run_hash is None

    def test_local_rank_env_respected_when_global_absent(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """LOCAL_RANK=0 with no RANK still counts as rank-zero."""
        result = run_driver(
            _rank_config(
                testbed,
                stats_jsonl_path,
                rank_env={"LOCAL_RANK": "0"},
                run_name="local-rank0-probe",
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_landed(aim_repo, result.run_hash, "metric_0")

    def test_single_process_fallback_is_rank_zero(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """No env vars, no torch.distributed → treated as rank-zero (writes land)."""
        result = run_driver(
            _rank_config(
                testbed,
                stats_jsonl_path,
                rank_env={},
                run_name="singleproc-probe",
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.run_hash is not None
        assert_metric_landed(aim_repo, result.run_hash, "metric_0")


class TestDetectionOrder:
    """The four-tier detection order: torch.dist > RANK > LOCAL_RANK > fallback."""

    def test_torch_distributed_wins_over_env(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """When torch.distributed reports rank-zero, that wins over conflicting env."""
        # Driver initializes torch.distributed with rank=0, sets env RANK=1
        result = run_driver(
            _rank_config(
                testbed,
                stats_jsonl_path,
                rank_env={"RANK": "1", "WORLD_SIZE": "2", "TESTBED_TORCH_DIST_RANK": "0"},
                run_name="torch-dist-wins-probe",
            )
        )
        assert result.exit_code == 0, result.stderr
        # torch.distributed said rank 0 → writes land
        assert result.run_hash is not None
        assert_metric_landed(aim_repo, result.run_hash, "metric_0")

    def test_rank_env_wins_over_local_rank(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        run_driver: RunFixture,
    ) -> None:
        """RANK=1 + LOCAL_RANK=0 → treated as non-zero rank (RANK wins)."""
        result = run_driver(
            _rank_config(
                testbed,
                stats_jsonl_path,
                rank_env={"RANK": "1", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
                run_name="rank-wins-probe",
            )
        )
        assert result.exit_code == 0, result.stderr
        # RANK=1 → non-zero → no run opened
        assert result.run_hash is None
