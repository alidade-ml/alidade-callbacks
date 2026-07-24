"""Integration tests for ``src/astrolabe_callbacks/composer.py``.

Composer's ``AstrolabeCallback`` translates the Composer trainer's
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

import pytest

if False:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


class TestTraining:
    """Training-step metric emission via Composer's batch_end hook."""

    def test_batch_end_emits_metric(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """batch_end call translates Composer's Logger.log_metrics into an Aim track."""
        pytest.importorskip("composer")
        raise NotImplementedError

    def test_epoch_end_flushes_new_metrics(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """New metrics that appeared during the epoch are flushed at epoch boundary."""
        pytest.importorskip("composer")
        raise NotImplementedError


class TestValidation:
    """Validation metrics routed through Composer's eval hooks."""

    def test_eval_batch_end_emits_val_metric(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """eval_batch_end call lands as val/* metric on the training run."""
        pytest.importorskip("composer")
        raise NotImplementedError


class TestTeardown:
    """Composer's close hook cleanly finalizes the Aim run."""

    def test_close_hook_sets_end_time(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """After Composer close, Aim run has end_time set."""
        pytest.importorskip("composer")
        raise NotImplementedError

    def test_close_hook_drains_buffer(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Metrics pending in buffer at close time still land."""
        pytest.importorskip("composer")
        raise NotImplementedError
