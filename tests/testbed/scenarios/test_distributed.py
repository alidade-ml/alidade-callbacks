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

import pytest

if False:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


class TestRankGating:
    """Non-zero ranks skip Aim writes."""

    def test_rank_zero_writes_land(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """RANK=0 simulates the primary process; metrics land in Aim."""
        raise NotImplementedError

    def test_rank_nonzero_skips_writes(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """RANK=1 simulates a secondary process; no Aim run is opened."""
        raise NotImplementedError

    def test_local_rank_env_respected_when_global_absent(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """LOCAL_RANK=0 with no RANK still counts as rank-zero."""
        raise NotImplementedError

    def test_single_process_fallback_is_rank_zero(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """No env vars, no torch.distributed → treated as rank-zero (writes land)."""
        raise NotImplementedError


class TestDetectionOrder:
    """The four-tier detection order in _distributed.py: torch.dist > RANK > LOCAL_RANK > fallback."""

    def test_torch_distributed_wins_over_env(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """When torch.distributed is initialized, its rank wins over env vars."""
        raise NotImplementedError

    def test_rank_env_wins_over_local_rank(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """RANK=1 + LOCAL_RANK=0 → treated as non-zero rank (RANK wins)."""
        raise NotImplementedError
