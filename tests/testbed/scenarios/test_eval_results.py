"""Integration tests for ``src/astrolabe_callbacks/eval_results.py``.

Module-level eval helpers against real Aim.

``start_eval_run``, ``log_eval_table``, and (post-eval-linkage Milestone 0)
``start_eval_run_from_checkpoint`` are the researcher-facing eval surface.
These scenarios verify each helper lands the right tags, metrics, and
lifecycle events against a real aim server.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from tests.testbed.harness.assertions import (
    assert_metric_count,
    assert_metric_values,
    assert_run_closed,
    assert_run_tag,
    get_run_tags,
)
from tests.testbed.harness.eval_driver import EvalDriverConfig, EvalDriverResult

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


def _table_config(
    testbed: "TestbedHandle",
    *,
    model_run_hash: str,
    rows: dict[str, tuple[str, float]],
    task_set: str = "glue",
    driver_flags: dict[str, str] | None = None,
) -> EvalDriverConfig:
    return EvalDriverConfig(
        aim_url=testbed.aim_url_from_client,
        task_set=task_set,
        model_run_hash=model_run_hash,
        rows=rows,
        streaming_metrics=[],
        use_from_checkpoint=False,
        checkpoint_path=None,
        on_missing_parent="warn",
        driver_flags=driver_flags or {},
    )


def _streaming_config(
    testbed: "TestbedHandle",
    *,
    model_run_hash: str,
    streaming: list[tuple[str, list[tuple[int, float]]]],
    task_set: str = "cola",
    driver_flags: dict[str, str] | None = None,
) -> EvalDriverConfig:
    return EvalDriverConfig(
        aim_url=testbed.aim_url_from_client,
        task_set=task_set,
        model_run_hash=model_run_hash,
        rows={},
        streaming_metrics=streaming,
        use_from_checkpoint=False,
        checkpoint_path=None,
        on_missing_parent="warn",
        driver_flags=driver_flags or {},
    )


def _checkpoint_config(
    testbed: "TestbedHandle",
    *,
    checkpoint_path: str,
    task_set: str = "cola",
    model_run_hash: str = "",
    on_missing_parent: str = "warn",
    driver_flags: dict[str, str] | None = None,
) -> EvalDriverConfig:
    return EvalDriverConfig(
        aim_url=testbed.aim_url_from_client,
        task_set=task_set,
        model_run_hash=model_run_hash,
        rows={},
        streaming_metrics=[],
        use_from_checkpoint=True,
        checkpoint_path=checkpoint_path,
        on_missing_parent=on_missing_parent,
        driver_flags=driver_flags or {},
    )


RunEvalFixture = Callable[[EvalDriverConfig], EvalDriverResult]
FAKE_PARENT_HASH = "a" * 24


class TestLogEvalTable:
    """One-shot ``log_eval_table`` — the 80% eval case."""

    def test_writes_all_rows_at_step_zero(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Each row appears as one metric with a single (step=0, value) point."""
        result = run_eval_driver(
            _table_config(
                testbed,
                model_run_hash=FAKE_PARENT_HASH,
                rows={
                    "cola": ("matthews", 0.82),
                    "sst2": ("accuracy", 0.94),
                },
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.eval_run_hash is not None
        assert_metric_values(
            aim_repo, result.eval_run_hash, "eval/cola/matthews", [(0, 0.82)]
        )
        assert_metric_values(
            aim_repo, result.eval_run_hash, "eval/sst2/accuracy", [(0, 0.94)]
        )

    def test_sets_identity_tags(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """astrolabe.kind=eval, astrolabe.task_set, astrolabe.model_run_hash all set."""
        result = run_eval_driver(
            _table_config(
                testbed,
                model_run_hash=FAKE_PARENT_HASH,
                task_set="glue",
                rows={"cola": ("matthews", 0.82)},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.eval_run_hash is not None
        assert_run_tag(aim_repo, result.eval_run_hash, "astrolabe.kind", "eval")
        assert_run_tag(aim_repo, result.eval_run_hash, "astrolabe.task_set", "glue")
        assert_run_tag(
            aim_repo, result.eval_run_hash, "astrolabe.model_run_hash", FAKE_PARENT_HASH
        )

    def test_closes_run(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Helper closes the run before returning (has end_time)."""
        result = run_eval_driver(
            _table_config(
                testbed,
                model_run_hash=FAKE_PARENT_HASH,
                rows={"cola": ("matthews", 0.82)},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.eval_run_hash is not None
        assert_run_closed(aim_repo, result.eval_run_hash)

    def test_rejects_empty_rows(
        self,
        testbed: "TestbedHandle",
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Empty ``rows`` dict raises before creating an Aim run."""
        result = run_eval_driver(
            _table_config(testbed, model_run_hash=FAKE_PARENT_HASH, rows={})
        )
        # Non-zero exit; no run created
        assert result.exit_code != 0
        assert result.eval_run_hash is None

    def test_rejects_non_numeric_score(
        self,
        testbed: "TestbedHandle",
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Non-float score raises TypeError before creating an Aim run."""
        result = run_eval_driver(
            _table_config(
                testbed,
                model_run_hash=FAKE_PARENT_HASH,
                # Driver interprets NaN-marker string as instruction to pass a non-numeric
                rows={"cola": ("matthews", 0.0)},
                driver_flags={"TESTBED_EVAL_INJECT_NON_NUMERIC": "1"},
            )
        )
        assert result.exit_code != 0
        assert result.eval_run_hash is None


class TestStartEvalRun:
    """Lower-level ``start_eval_run`` — streaming / rolling eval case."""

    def test_returns_open_run_with_tags(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Returned aim.Run has identity tags and is open for track()."""
        result = run_eval_driver(
            _streaming_config(
                testbed,
                model_run_hash=FAKE_PARENT_HASH,
                task_set="cola",
                streaming=[("eval/cola/matthews", [(1, 0.5)])],
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.eval_run_hash is not None
        tags = get_run_tags(aim_repo, result.eval_run_hash)
        assert tags["astrolabe.kind"] == "eval"
        assert tags["astrolabe.task_set"] == "cola"

    def test_caller_owns_close(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Helper does NOT auto-close; caller must call close().

        Driver deliberately leaves the run open, verifies helper didn't
        force-close, then closes explicitly at the end.
        """
        result = run_eval_driver(
            _streaming_config(
                testbed,
                model_run_hash=FAKE_PARENT_HASH,
                streaming=[("eval/cola/matthews", [(1, 0.5)])],
                driver_flags={"TESTBED_EVAL_VERIFY_NO_AUTOCLOSE": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr

    def test_multi_step_tracking_lands(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Successive track() calls at different steps produce a full series."""
        result = run_eval_driver(
            _streaming_config(
                testbed,
                model_run_hash=FAKE_PARENT_HASH,
                streaming=[
                    ("eval/cola/matthews", [(1000, 0.4), (2000, 0.5), (3000, 0.6)])
                ],
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.eval_run_hash is not None
        assert_metric_values(
            aim_repo,
            result.eval_run_hash,
            "eval/cola/matthews",
            [(1000, 0.4), (2000, 0.5), (3000, 0.6)],
        )


class TestStartEvalRunFromCheckpoint:
    """Checkpoint-based eval linkage (lands with eval-linkage Milestone 0)."""

    def test_reads_embedded_meta_from_pt(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Loads a .pt checkpoint with embedded astrolabe meta, sets model_run_hash."""
        # Driver creates a .pt with embedded meta at TESTBED_CKPT_HASH before eval starts.
        result = run_eval_driver(
            _checkpoint_config(
                testbed,
                checkpoint_path="/tmp/testbed-ckpt.pt",
                driver_flags={"TESTBED_CREATE_PT_CHECKPOINT_WITH_HASH": FAKE_PARENT_HASH},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.linked is True
        assert result.eval_run_hash is not None
        assert_run_tag(
            aim_repo, result.eval_run_hash, "astrolabe.model_run_hash", FAKE_PARENT_HASH
        )

    def test_reads_embedded_meta_from_safetensors(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Loads a .safetensors checkpoint via header-only read, sets model_run_hash."""
        result = run_eval_driver(
            _checkpoint_config(
                testbed,
                checkpoint_path="/tmp/testbed-ckpt.safetensors",
                driver_flags={
                    "TESTBED_CREATE_SAFETENSORS_CHECKPOINT_WITH_HASH": FAKE_PARENT_HASH
                },
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.linked is True
        assert result.eval_run_hash is not None
        assert_run_tag(
            aim_repo, result.eval_run_hash, "astrolabe.model_run_hash", FAKE_PARENT_HASH
        )

    def test_explicit_model_run_hash_wins(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Explicit ``model_run_hash=`` param overrides embedded meta."""
        override = "b" * 24
        result = run_eval_driver(
            _checkpoint_config(
                testbed,
                checkpoint_path="/tmp/testbed-ckpt.pt",
                model_run_hash=override,
                driver_flags={
                    # Embedded hash differs from override; override wins
                    "TESTBED_CREATE_PT_CHECKPOINT_WITH_HASH": FAKE_PARENT_HASH
                },
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.eval_run_hash is not None
        assert_run_tag(
            aim_repo, result.eval_run_hash, "astrolabe.model_run_hash", override
        )

    def test_on_missing_parent_warn_returns_unlinked_run(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """No meta + on_missing_parent='warn' → logs WARNING, returns unlinked run."""
        result = run_eval_driver(
            _checkpoint_config(
                testbed,
                checkpoint_path="/tmp/testbed-ckpt-no-meta.pt",
                on_missing_parent="warn",
                driver_flags={"TESTBED_CREATE_PT_CHECKPOINT_WITHOUT_META": "1"},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.linked is False
        # Run created but no model_run_hash tag
        assert result.eval_run_hash is not None
        tags = get_run_tags(aim_repo, result.eval_run_hash)
        assert "astrolabe.model_run_hash" not in tags

    def test_on_missing_parent_raise_raises(
        self,
        testbed: "TestbedHandle",
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """No meta + on_missing_parent='raise' → MissingParentError."""
        result = run_eval_driver(
            _checkpoint_config(
                testbed,
                checkpoint_path="/tmp/testbed-ckpt-no-meta.pt",
                on_missing_parent="raise",
                driver_flags={"TESTBED_CREATE_PT_CHECKPOINT_WITHOUT_META": "1"},
            )
        )
        # Exit code 43 = MissingParentError sentinel (see EvalDriverResult docstring)
        assert result.exit_code == 43
        assert result.eval_run_hash is None

    def test_returned_run_exposes_linked_attribute(
        self,
        testbed: "TestbedHandle",
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Returned run has ``.astrolabe_linked: bool`` reflecting linkage state."""
        # Linked case
        result_linked = run_eval_driver(
            _checkpoint_config(
                testbed,
                checkpoint_path="/tmp/testbed-ckpt.pt",
                driver_flags={"TESTBED_CREATE_PT_CHECKPOINT_WITH_HASH": FAKE_PARENT_HASH},
            )
        )
        assert result_linked.exit_code == 0
        assert result_linked.linked is True

        # Unlinked case
        result_unlinked = run_eval_driver(
            _checkpoint_config(
                testbed,
                checkpoint_path="/tmp/testbed-ckpt-no-meta.pt",
                on_missing_parent="warn",
                driver_flags={"TESTBED_CREATE_PT_CHECKPOINT_WITHOUT_META": "1"},
            )
        )
        assert result_unlinked.exit_code == 0
        assert result_unlinked.linked is False
