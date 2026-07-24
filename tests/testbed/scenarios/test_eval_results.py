"""Integration tests for ``src/astrolabe_callbacks/eval_results.py``.

Module-level eval helpers against real Aim.

``start_eval_run``, ``log_eval_table``, and (post-eval-linkage Milestone 0)
``start_eval_run_from_checkpoint`` are the researcher-facing eval surface.
These scenarios verify each helper lands the right tags, metrics, and
lifecycle events against a real aim server.
"""
from __future__ import annotations

from pathlib import Path

import pytest

if False:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


class TestLogEvalTable:
    """One-shot ``log_eval_table`` — the 80% eval case."""

    def test_writes_all_rows_at_step_zero(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Each row appears as one metric with a single (step=0, value) point."""
        raise NotImplementedError

    def test_sets_identity_tags(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """astrolabe.kind=eval, astrolabe.task_set, astrolabe.model_run_hash all set."""
        raise NotImplementedError

    def test_closes_run(self, testbed: "TestbedHandle", aim_repo: Path) -> None:
        """Helper closes the run before returning (has end_time)."""
        raise NotImplementedError

    def test_rejects_empty_rows(self, testbed: "TestbedHandle", aim_repo: Path) -> None:
        """Empty ``rows`` dict raises before creating an Aim run."""
        raise NotImplementedError

    def test_rejects_non_numeric_score(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Non-float score raises TypeError before creating an Aim run."""
        raise NotImplementedError


class TestStartEvalRun:
    """Lower-level ``start_eval_run`` — streaming / rolling eval case."""

    def test_returns_open_run_with_tags(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Returned aim.Run has identity tags and is open for track()."""
        raise NotImplementedError

    def test_caller_owns_close(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Helper does NOT auto-close; caller must call close()."""
        raise NotImplementedError

    def test_multi_step_tracking_lands(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Successive track() calls at different steps produce a full series."""
        raise NotImplementedError


class TestStartEvalRunFromCheckpoint:
    """Checkpoint-based eval linkage (lands with eval-linkage Milestone 0)."""

    def test_reads_embedded_meta_from_pt(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Loads a .pt checkpoint with embedded astrolabe meta, sets model_run_hash."""
        raise NotImplementedError

    def test_reads_embedded_meta_from_safetensors(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Loads a .safetensors checkpoint via header-only read, sets model_run_hash."""
        raise NotImplementedError

    def test_explicit_model_run_hash_wins(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Explicit ``model_run_hash=`` param overrides embedded meta."""
        raise NotImplementedError

    def test_on_missing_parent_warn_returns_unlinked_run(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """No meta + on_missing_parent='warn' → logs WARNING, returns unlinked run."""
        raise NotImplementedError

    def test_on_missing_parent_raise_raises(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """No meta + on_missing_parent='raise' → MissingParentError."""
        raise NotImplementedError

    def test_returned_run_exposes_linked_attribute(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Returned run has ``.astrolabe_linked: bool`` reflecting linkage state."""
        raise NotImplementedError
