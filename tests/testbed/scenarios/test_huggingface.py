"""Integration tests for ``src/astrolabe_callbacks/huggingface.py``.

HF Trainer's ``AstrolabeHFCallback`` translates HuggingFace Trainer
lifecycle events (on_log, on_evaluate, on_train_end, ...) into
AstrolabeLogger calls. This file verifies the adapter drives the
underlying Logger correctly against a real Aim server.

Class names (`TestTraining` / `TestValidation` / `TestTeardown`) are
consistent across all framework test files.

Skips if the ``hf`` extra is not installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

if False:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


class TestTraining:
    """Training-step metric emission via HF's on_log hook."""

    def test_on_log_emits_metric(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        pytest.importorskip("transformers")
        raise NotImplementedError


class TestValidation:
    """Validation metrics routed through HF's on_evaluate hook."""

    def test_on_evaluate_emits_val_metric(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        pytest.importorskip("transformers")
        raise NotImplementedError


class TestTeardown:
    """HF's on_train_end hook cleanly finalizes the Aim run."""

    def test_on_train_end_sets_end_time(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        pytest.importorskip("transformers")
        raise NotImplementedError

    def test_on_train_end_drains_buffer(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Metrics pending in buffer at on_train_end still land."""
        pytest.importorskip("transformers")
        raise NotImplementedError
