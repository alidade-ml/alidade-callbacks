"""Integration tests for ``src/astrolabe_callbacks/pytorch.py``.

Raw PyTorch helper — no framework Trainer, users drive AstrolabeLogger
directly via ``track_metric`` or a context manager. This file verifies
the direct-usage surface works against a real Aim server.

Class names (`TestTraining` / `TestTeardown`) are consistent with the
other framework test files. `TestContextManager` is PyTorch-specific —
no other adapter uses a context-manager entry point, so it stays as a
distinct class rather than being flattened into `TestTeardown`.

No framework extra required — pytorch is always available.
"""
from __future__ import annotations

from pathlib import Path

import pytest

if False:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


class TestTraining:
    """Direct AstrolabeLogger.track() calls emit metrics."""

    def test_direct_track_emits_metric(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        raise NotImplementedError


class TestTeardown:
    """Explicit close() finalizes the Aim run cleanly."""

    def test_close_sets_end_time(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        raise NotImplementedError

    def test_close_drains_buffer(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Metrics pending in buffer at close still land."""
        raise NotImplementedError


class TestContextManager:
    """Framework-specific: ``with AstrolabeLogger(...) as logger`` semantics.

    Kept distinct from `TestTeardown` because the context-manager surface
    is a PyTorch-only entry point (other adapters drive the callback
    through their framework's lifecycle, not a with-block).
    """

    def test_context_manager_closes_on_normal_exit(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Context exit closes the run and drains the buffer."""
        raise NotImplementedError

    def test_context_manager_closes_on_exception(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Exception inside the with-block still closes the run cleanly."""
        raise NotImplementedError
