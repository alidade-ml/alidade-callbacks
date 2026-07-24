"""Integration tests for ``src/astrolabe_callbacks/lightning.py``.

Lightning's ``AstrolabeLightningCallback`` translates PyTorch
Lightning's on_* hooks into AstrolabeLogger calls. This file verifies
the adapter drives the underlying Logger correctly against a real Aim
server.

Class names (`TestTraining` / `TestValidation` / `TestTeardown`) are
consistent across all framework test files.

Skips if the ``lightning`` extra is not installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

if False:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


class TestTraining:
    """Training-step metric emission via Lightning's on_train_batch_end hook."""

    def test_on_train_batch_end_emits_metric(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        pytest.importorskip("lightning")
        raise NotImplementedError


class TestValidation:
    """Validation metrics routed through Lightning's on_validation_end hook."""

    def test_on_validation_end_emits_val_metric(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """val/* metrics land on the same run as training, not a separate eval run."""
        pytest.importorskip("lightning")
        raise NotImplementedError


class TestTeardown:
    """Lightning's teardown hook cleanly finalizes the Aim run."""

    def test_teardown_sets_end_time(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        pytest.importorskip("lightning")
        raise NotImplementedError

    def test_teardown_drains_buffer(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Metrics pending in buffer at teardown time still land."""
        pytest.importorskip("lightning")
        raise NotImplementedError
