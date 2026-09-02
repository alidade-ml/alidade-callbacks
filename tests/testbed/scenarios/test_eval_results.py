"""Integration tests for ``src/alidade_callbacks/eval_results.py``.

Module-level eval helpers against real Aim.

``start_eval_run``, ``log_eval_table``, and (post-eval-linkage Milestone 0)
``start_eval_run_from_checkpoint`` are the researcher-facing eval surface.
These scenarios verify each helper lands the right tags, metrics, and
lifecycle events against a real aim server.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from tests.testbed.harness.assertions import (
    assert_metric_count,
    assert_metric_values,
    assert_run_closed,
    assert_run_experiment,
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


    def test_second_metric_on_one_task_is_refused(
        self,
        testbed: "TestbedHandle",
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """A task gets one column, so the second metric is refused at the
        call site rather than written and left unreachable."""
        result = run_eval_driver(
            _streaming_config(
                testbed,
                model_run_hash=FAKE_PARENT_HASH,
                task_set="ordering",
                streaming=[
                    ("eval/multi/aaa_first", [(0, 0.111)]),
                    ("eval/multi/zzz_last", [(0, 0.222)]),
                ],
            )
        )
        assert result.exit_code != 0
        assert "eval/multi/aaa_first" in result.stderr
        assert "eval/multi/zzz_last" in result.stderr

    def test_one_metric_each_on_two_tasks_both_land(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """The guard is per task. A mock cannot show that wrapping a real
        ``aim.Run``'s ``track`` still writes through to Aim."""
        result = run_eval_driver(
            _streaming_config(
                testbed,
                model_run_hash=FAKE_PARENT_HASH,
                task_set="ordering",
                streaming=[
                    ("eval/multi/aaa_first", [(0, 0.111)]),
                    ("eval/solo/only", [(0, 0.333)]),
                ],
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.eval_run_hash is not None
        assert_metric_values(
            aim_repo, result.eval_run_hash, "eval/multi/aaa_first", [(0, 0.111)]
        )
        assert_metric_values(
            aim_repo, result.eval_run_hash, "eval/solo/only", [(0, 0.333)]
        )


class TestStartEvalRunFromCheckpoint:
    """Checkpoint-based eval linkage. Unskipped when the helper landed."""

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


class TestSubmitIdentity:
    """Eval runs produced inside an astrolabe submit inherit its identity.

    The engine exports ``AIM_RUN_TAGS`` and ``ALIDADE_EXPERIMENT_NAME``
    into every step env; the training callback reads them and the eval
    helpers used not to. These scenarios run the real helpers against a
    real Aim server with that env present, because filing is a property
    Aim resolves at run-open — a mocked ``Run`` cannot show which
    experiment a run actually landed in.
    """

    SUBMIT_ENV = {
        "ALIDADE_EXPERIMENT_NAME": "latent-bert",
        "AIM_RUN_TAGS": (
            "astrolabe.submit_id=s-testbed-1,astrolabe.version=v3,"
            "astrolabe.user=nathan,astrolabe.experiment=latent-bert"
        ),
    }

    def test_eval_lands_in_the_submitting_experiment(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """The whole point: a submit's evals sit on the submit's page.

        Filed under ``eval/<task_set>`` they land on a page named for the
        benchmark, which no experiment view queries.
        """
        config = _table_config(
            testbed,
            model_run_hash=FAKE_PARENT_HASH,
            task_set="glue",
            rows={"cola": ("matthews", 0.82)},
        )
        result = run_eval_driver(
            replace(config, submit_env=dict(self.SUBMIT_ENV))
        )
        assert result.exit_code == 0, result.stderr
        assert result.eval_run_hash is not None
        assert_run_experiment(aim_repo, result.eval_run_hash, "latent-bert")

    def test_submit_tags_land_on_the_eval_run(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Without these an eval is unattributable — no submitter to filter
        on and no submit to bill its GPU time against."""
        config = _table_config(
            testbed,
            model_run_hash=FAKE_PARENT_HASH,
            task_set="glue",
            rows={"cola": ("matthews", 0.82)},
        )
        result = run_eval_driver(
            replace(config, submit_env=dict(self.SUBMIT_ENV))
        )
        assert result.exit_code == 0, result.stderr
        assert result.eval_run_hash is not None
        tags = get_run_tags(aim_repo, result.eval_run_hash)
        assert tags["astrolabe.submit_id"] == "s-testbed-1"
        assert tags["astrolabe.version"] == "v3"
        assert tags["astrolabe.user"] == "nathan"
        # The discovery tags must survive being written alongside them.
        assert tags["astrolabe.kind"] == "eval"
        assert tags["astrolabe.model_run_hash"] == FAKE_PARENT_HASH

    def test_falls_back_to_the_benchmark_outside_a_submit(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """Ad-hoc use away from astrolabe still has to work — there is no
        experiment to inherit, so the benchmark label is all there is."""
        result = run_eval_driver(
            _table_config(
                testbed,
                model_run_hash=FAKE_PARENT_HASH,
                task_set="mmlu",
                rows={"stem": ("accuracy", 0.61)},
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.eval_run_hash is not None
        assert_run_experiment(aim_repo, result.eval_run_hash, "eval/mmlu")

    def test_an_unlinked_eval_is_filed_the_same_way(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        run_eval_driver: RunEvalFixture,
    ) -> None:
        """A run stamped with its model after the fact has to be
        indistinguishable from one that resolved immediately, or stamping
        moves it between pages."""
        config = _checkpoint_config(
            testbed,
            checkpoint_path="/tmp/unstamped.pt",
            task_set="glue",
            on_missing_parent="warn",
            driver_flags={"TESTBED_CREATE_PT_CHECKPOINT_WITHOUT_META": "1"},
        )
        result = run_eval_driver(
            replace(config, submit_env=dict(self.SUBMIT_ENV))
        )
        assert result.exit_code == 0, result.stderr
        assert result.linked is False
        assert result.eval_run_hash is not None
        assert_run_experiment(aim_repo, result.eval_run_hash, "latent-bert")
        assert_run_tag(
            aim_repo, result.eval_run_hash, "astrolabe.submit_id", "s-testbed-1"
        )
