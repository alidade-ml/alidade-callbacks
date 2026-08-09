"""Tests for astrolabe_callbacks.eval_results — post-training eval helpers.

Contract being verified:

* :func:`log_eval_table` opens an Aim run, applies the three-tag identity
  contract, tracks each row under ``eval/<task>/<metric>`` at ``step=0``,
  and closes the run.
* :func:`start_eval_run` returns an *open* run with the three tags set;
  the caller owns ``close()``.
* Both reject malformed inputs at the call site BEFORE creating any
  Aim run — half-tagged runs would silently confuse astrolabe's
  dashboard.
* The metric path convention ``eval/<task>/<metric>`` must be exact —
  slashes in the task or metric label scramble the dashboard's column
  parsing.

Tests mock ``aim.Run`` at the SDK boundary. Real Aim has
indexing/commit timing quirks that bite read-back tests; mocking the
SDK boundary is the convention across this package's tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from astrolabe_callbacks.eval_results import (
    EvalInputError,
    log_eval_table,
    start_eval_run,
)


# ---------- helpers ----------------------------------------------------


def _make_run_mock(run_hash: str = "abc123") -> MagicMock:
    """Stand-in for ``aim.Run``. Tracks setitem + track + close calls."""
    run = MagicMock()
    run.hash = run_hash
    return run


# ---------- input validation: log_eval_table -------------------------


class TestLogEvalTableValidation:
    """All validation errors MUST fire BEFORE any Aim run is created.

    A half-tagged eval run (e.g., kind set but task_set missing) would
    appear in the dashboard's discovery query but render nothing — silent
    corruption is worse than a noisy crash at the call site.
    """

    @pytest.fixture(autouse=True)
    def _aim_patch(self):
        """Patch aim.Run for every test in this class; assert it wasn't called.

        The fixture is autouse so even tests that expect validation to
        raise can verify Run() was never reached.
        """
        self.aim_run_mock = MagicMock(return_value=_make_run_mock())
        self.aim_url = "aim://test"
        with patch("aim.Run", self.aim_run_mock):
            yield

    def _assert_no_run_created(self):
        assert not self.aim_run_mock.called, (
            "Validation must reject BEFORE creating an Aim run; "
            f"aim.Run was called with {self.aim_run_mock.call_args}"
        )

    def test_rejects_empty_model_run_hash(self):
        with pytest.raises(EvalInputError, match="model_run_hash"):
            log_eval_table(
                model_run_hash="",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_non_string_model_run_hash(self):
        with pytest.raises(EvalInputError, match="model_run_hash"):
            log_eval_table(
                model_run_hash=None,  # type: ignore[arg-type]
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_empty_task_set(self):
        with pytest.raises(EvalInputError, match="task_set"):
            log_eval_table(
                model_run_hash="abc",
                task_set="",
                rows={"cola": ("matthews", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_empty_rows(self):
        with pytest.raises(EvalInputError, match="at least one task"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_non_dict_rows(self):
        with pytest.raises(EvalInputError, match="must be a dict"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows=[("cola", "matthews", 0.5)],  # type: ignore[arg-type]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_empty_task_name(self):
        with pytest.raises(EvalInputError, match="task name"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"": ("matthews", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_slash_in_task_name(self):
        # Metric path is ``eval/<task>/<metric>``. A slash in the task
        # scrambles which segment the dashboard reads as which.
        with pytest.raises(EvalInputError, match="must not contain '/'"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola/sub": ("matthews", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_slash_in_metric_label(self):
        with pytest.raises(EvalInputError, match="must not contain '/'"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews/v2", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_empty_metric_label(self):
        with pytest.raises(EvalInputError, match="metric label"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_non_tuple_row(self):
        with pytest.raises(EvalInputError, match="must be a .* tuple"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": 0.5},  # type: ignore[dict-item]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_three_element_row(self):
        with pytest.raises(EvalInputError, match="must be a .* tuple"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 0.5, "extra")},  # type: ignore[dict-item]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_string_score(self):
        with pytest.raises(EvalInputError, match="must be a number"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", "0.5")},  # type: ignore[dict-item]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_bool_score(self):
        # bool is an int subclass in Python — without the explicit reject,
        # ``rows={"cola": ("accuracy", True)}`` would silently log 1.0.
        with pytest.raises(EvalInputError, match="must be a number"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("accuracy", True)},  # type: ignore[dict-item]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_none_score(self):
        with pytest.raises(EvalInputError, match="must be a number"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", None)},  # type: ignore[dict-item]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()


# ---------- input validation: start_eval_run -------------------------


class TestStartEvalRunValidation:
    @pytest.fixture(autouse=True)
    def _aim_patch(self):
        self.aim_run_mock = MagicMock(return_value=_make_run_mock())
        self.aim_url = "aim://test"
        with patch("aim.Run", self.aim_run_mock):
            yield

    def test_rejects_empty_model_run_hash(self):
        with pytest.raises(EvalInputError, match="model_run_hash"):
            start_eval_run(
                model_run_hash="",
                task_set="glue",
                aim_url=self.aim_url,
            )
        assert not self.aim_run_mock.called

    def test_rejects_empty_task_set(self):
        with pytest.raises(EvalInputError, match="task_set"):
            start_eval_run(
                model_run_hash="abc",
                task_set="",
                aim_url=self.aim_url,
            )
        assert not self.aim_run_mock.called


# ---------- tag contract ---------------------------------------------


class TestTagContract:
    """The three identity tags are the dashboard's only way to discover
    eval runs from the model run page. Missing any of them = invisible run."""

    def test_log_eval_table_sets_all_three_tags(self, tmp_path):
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="model-hash-123",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url="aim://test",
            )
        run.__setitem__.assert_any_call("astrolabe.kind", "eval")
        run.__setitem__.assert_any_call("astrolabe.task_set", "glue")
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", "model-hash-123")

    def test_start_eval_run_sets_all_three_tags(self, tmp_path):
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            start_eval_run(
                model_run_hash="xyz",
                task_set="mmlu",
                aim_url="aim://test",
            )
        run.__setitem__.assert_any_call("astrolabe.kind", "eval")
        run.__setitem__.assert_any_call("astrolabe.task_set", "mmlu")
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", "xyz")

    def test_falls_back_to_eval_task_set_outside_a_submit(self, tmp_path):
        # Ad-hoc use with no astrolabe env around it: there is no
        # experiment to inherit, so the benchmark label is all we have.
        run = _make_run_mock()
        mock_run_factory = MagicMock(return_value=run)
        with patch("aim.Run", mock_run_factory):
            start_eval_run(
                model_run_hash="abc",
                task_set="glue",
                aim_url="aim://test",
            )
        mock_run_factory.assert_called_once()
        kwargs = mock_run_factory.call_args.kwargs
        assert kwargs["experiment"] == "eval/glue"


# ---------- metric path convention ----------------------------------


class TestMetricPaths:
    """The dashboard's table block parses ``eval/<task>/<metric>``.
    Tracking under any other shape leaves the table un-populated."""

    def test_paths_use_eval_task_metric_convention(self, tmp_path):
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={
                    "cola": ("matthews", 0.822),
                    "sst2": ("accuracy", 0.943),
                },
                aim_url="aim://test",
            )
        # Verify both metrics tracked with the right paths.
        run.track.assert_any_call(0.822, name="eval/cola/matthews", step=0)
        run.track.assert_any_call(0.943, name="eval/sst2/accuracy", step=0)

    def test_one_track_call_per_row(self, tmp_path):
        run = _make_run_mock()
        rows = {
            "cola": ("matthews", 0.5),
            "sst2": ("accuracy", 0.6),
            "mnli": ("accuracy_matched", 0.7),
        }
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows=rows,
                aim_url="aim://test",
            )
        assert run.track.call_count == len(rows)

    def test_step_is_zero(self, tmp_path):
        # step=0 marks "post-training one-shot" — the dispatcher uses
        # this to choose the table block over the trace block.
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url="aim://test",
            )
        for call_args in run.track.call_args_list:
            assert call_args.kwargs["step"] == 0

    def test_score_is_cast_to_float(self, tmp_path):
        # Aim's track() expects a numeric — int inputs should reach it
        # as floats so all eval values share a type in storage.
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 1)},  # int score
                aim_url="aim://test",
            )
        ((value,), _) = run.track.call_args
        assert isinstance(value, float)
        assert value == 1.0


# ---------- run lifecycle ----------------------------------------------


class TestRunLifecycle:
    def test_log_eval_table_closes_run(self, tmp_path):
        # Forgetting to close leaves end_time=0 — the dashboard would
        # render this as in-flight forever.
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url="aim://test",
            )
        run.close.assert_called_once()

    def test_log_eval_table_closes_run_even_when_track_raises(self, tmp_path):
        # If Aim's track() blows up mid-loop, we still need to close the
        # run so we don't leak an in-flight tag on the dashboard.
        run = _make_run_mock()
        run.track.side_effect = RuntimeError("aim hiccup")
        with patch("aim.Run", return_value=run):
            with pytest.raises(RuntimeError, match="aim hiccup"):
                log_eval_table(
                    model_run_hash="abc",
                    task_set="glue",
                    rows={"cola": ("matthews", 0.5)},
                    aim_url="aim://test",
                )
        run.close.assert_called_once()

    def test_start_eval_run_does_not_close(self, tmp_path):
        # The lower-level helper hands the caller an OPEN run. Closing
        # here would set end_time=0 immediately.
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            start_eval_run(
                model_run_hash="abc",
                task_set="glue",
                aim_url="aim://test",
            )
        assert not run.close.called

    def test_log_eval_table_returns_run_hash(self, tmp_path):
        run = _make_run_mock(run_hash="b73e9c8d")
        with patch("aim.Run", return_value=run):
            got = log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url="aim://test",
            )
        assert got == "b73e9c8d"


# ---------- happy path summary ----------------------------------------


class TestHappyPath:
    def test_full_glue_table_round_trip(self, tmp_path):
        run = _make_run_mock(run_hash="eval-abc")
        with patch("aim.Run", return_value=run):
            got = log_eval_table(
                model_run_hash="model-hash-123",
                task_set="glue",
                rows={
                    "cola": ("matthews",          0.822),
                    "sst2": ("accuracy",          0.943),
                    "mnli": ("accuracy_matched",  0.864),
                    "avg":  ("mean",              0.876),
                },
                aim_url="aim://test",
            )
        assert got == "eval-abc"
        # All three identity tags set
        run.__setitem__.assert_any_call("astrolabe.kind", "eval")
        run.__setitem__.assert_any_call("astrolabe.task_set", "glue")
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", "model-hash-123")
        # All four rows tracked under the convention path
        run.track.assert_any_call(0.822, name="eval/cola/matthews", step=0)
        run.track.assert_any_call(0.943, name="eval/sst2/accuracy", step=0)
        run.track.assert_any_call(0.864, name="eval/mnli/accuracy_matched", step=0)
        run.track.assert_any_call(0.876, name="eval/avg/mean", step=0)
        # Run closed
        run.close.assert_called_once()


# ---------- start_eval_run_from_checkpoint ---------------------------
#
# Contract re-derived from purpose (this helper and its tests were
# written in one sitting, so the contract is stated rather than read
# off the implementation):
#
# - The eval author names a FILE. The training run comes out of that
#   file. No hash, no tag key, no astrolabe internals at the call site.
# - Resolution order: explicit model_run_hash > the checkpoint's
#   aim_run_hash > on_missing_parent.
# - Resolution is OFFLINE. Aim is never queried to find a parent. A
#   checkpoint carrying a submit but no run is unresolved, not a lookup
#   trigger — guessing among several runs under one submit is the
#   mechanism this helper replaces.
# - An unlinked run is a supported outcome, not an error, and must
#   still be recognizable as an eval so it can be stamped later.
# - Validation fires before any Aim run exists, same as the rest of
#   this module: a half-formed run is worse than a loud failure.

from pathlib import Path  # noqa: E402

from astrolabe_callbacks.checkpoint import export_checkpoint  # noqa: E402
from astrolabe_callbacks.checkpoint import CheckpointMeta  # noqa: E402
from astrolabe_callbacks.eval_results import (  # noqa: E402
    MissingParentError,
    register_external_model,
    start_eval_run_from_checkpoint,
)

ORIGIN = "aaaa1111bbbb2222cccc3333"


def _ckpt(tmp_path: Path, name="model.pt", **meta_fields) -> Path:
    meta = CheckpointMeta(created_at="2026-08-06T00:00:00Z", **meta_fields)
    return export_checkpoint({}, tmp_path / name, fmt="pt", meta=meta)


class TestFromCheckpointValidation:
    """Every one of these must fail before an Aim run is created."""

    @pytest.mark.parametrize("bad", ["nope", "", None, "WARN", 1])
    def test_rejects_unknown_on_missing_parent(self, tmp_path, bad):
        factory = MagicMock()
        with patch("aim.Run", factory):
            with pytest.raises(EvalInputError, match="on_missing_parent"):
                start_eval_run_from_checkpoint(
                    checkpoint={}, task_set="glue", on_missing_parent=bad
                )
        factory.assert_not_called()

    @pytest.mark.parametrize("bad", ["", None, 0, []])
    def test_rejects_empty_task_set(self, tmp_path, bad):
        factory = MagicMock()
        with patch("aim.Run", factory):
            with pytest.raises(EvalInputError, match="task_set"):
                start_eval_run_from_checkpoint(checkpoint={}, task_set=bad)
        factory.assert_not_called()

    def test_rejects_empty_explicit_hash(self, tmp_path):
        """Empty is a caller bug, not a request to fall through."""
        factory = MagicMock()
        with patch("aim.Run", factory):
            with pytest.raises(EvalInputError, match="model_run_hash"):
                start_eval_run_from_checkpoint(
                    checkpoint={}, task_set="glue", model_run_hash=""
                )
        factory.assert_not_called()

    def test_missing_checkpoint_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            start_eval_run_from_checkpoint(
                checkpoint=tmp_path / "nope.pt", task_set="glue"
            )

    def test_unrecognizable_checkpoint_raises(self, tmp_path):
        junk = tmp_path / "junk.pt"
        junk.write_bytes(b"not a checkpoint")
        with pytest.raises(ValueError):
            start_eval_run_from_checkpoint(checkpoint=junk, task_set="glue")


class TestUnresolvedParent:
    def test_raise_mode_raises_missing_parent(self, tmp_path):
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        factory = MagicMock()
        with patch("aim.Run", factory):
            with pytest.raises(MissingParentError):
                start_eval_run_from_checkpoint(
                    checkpoint=plain, task_set="glue", on_missing_parent="raise"
                )
        factory.assert_not_called()

    def test_missing_parent_error_is_not_an_input_error(self, tmp_path):
        """Distinct type so CI can fail on orphaned evals without also
        swallowing malformed arguments."""
        assert not issubclass(MissingParentError, EvalInputError)

    def test_warn_mode_returns_an_unlinked_run(self, tmp_path):
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            result = start_eval_run_from_checkpoint(
                checkpoint=plain, task_set="glue", on_missing_parent="warn"
            )
        assert result.astrolabe_linked is False

    def test_unlinked_run_carries_no_model_run_hash_tag(self, tmp_path):
        """A blank or placeholder hash would be worse than none — the
        dashboard would join on it and render an empty section."""
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            start_eval_run_from_checkpoint(
                checkpoint=plain, task_set="glue", on_missing_parent="warn"
            )
        tags = [c.args[0] for c in run.__setitem__.call_args_list]
        assert "astrolabe.model_run_hash" not in tags

    def test_unlinked_run_is_still_recognizable_as_an_eval(self, tmp_path):
        """Stamping it later only works if kind and task_set are set."""
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            start_eval_run_from_checkpoint(
                checkpoint=plain, task_set="glue", on_missing_parent="warn"
            )
        run.__setitem__.assert_any_call("astrolabe.kind", "eval")
        run.__setitem__.assert_any_call("astrolabe.task_set", "glue")

    def test_submit_without_a_run_hash_stays_unresolved(self, tmp_path):
        """The offline rule. Env gave this checkpoint an identity but no
        run existed in-process. Resolving it would mean asking Aim which
        run under the submit to pick — ambiguous exactly when healing
        fired, and a guess is what this helper replaces."""
        ckpt = _ckpt(tmp_path, submit_id="sub-123", experiment="exp", version="v1")
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            result = start_eval_run_from_checkpoint(
                checkpoint=ckpt, task_set="glue", on_missing_parent="warn"
            )
        assert result.astrolabe_linked is False


class TestResolutionOrder:
    def test_explicit_hash_overrides_the_checkpoint(self, tmp_path):
        ckpt = _ckpt(tmp_path, aim_run_hash=ORIGIN)
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            start_eval_run_from_checkpoint(
                checkpoint=ckpt, task_set="glue", model_run_hash="override-me"
            )
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", "override-me")

    def test_explicit_hash_used_when_checkpoint_has_none(self, tmp_path):
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            result = start_eval_run_from_checkpoint(
                checkpoint=plain, task_set="glue", model_run_hash=ORIGIN
            )
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", ORIGIN)
        assert result.astrolabe_linked is True


class TestFromCheckpointHappyPath:
    def test_links_to_the_run_embedded_in_the_checkpoint(self, tmp_path):
        ckpt = _ckpt(tmp_path, aim_run_hash=ORIGIN, submit_id="sub-1")
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            result = start_eval_run_from_checkpoint(checkpoint=ckpt, task_set="cola")
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", ORIGIN)
        run.__setitem__.assert_any_call("astrolabe.kind", "eval")
        run.__setitem__.assert_any_call("astrolabe.task_set", "cola")
        assert result.astrolabe_linked is True

    def test_a_derived_checkpoint_attributes_to_the_original_training(self, tmp_path):
        """The GLUE-probe shape. Surgery copies the origin run forward
        into aim_run_hash, so the reader needs no chain walking — it
        reads one field and gets the pretrain."""
        derived = _ckpt(
            tmp_path,
            name="disc.pt",
            aim_run_hash=ORIGIN,
            derived_from=ORIGIN,
            derivation_chain_length=1,
        )
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            result = start_eval_run_from_checkpoint(checkpoint=derived, task_set="cola")
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", ORIGIN)
        assert result.astrolabe_linked is True

    def test_accepts_an_already_loaded_state_dict(self, tmp_path):
        """Script-level eval has usually already loaded the file to run
        the model; making it pay a second read would be a papercut."""
        from astrolabe_callbacks.checkpoint import stamp_state_dict

        state = stamp_state_dict({}, CheckpointMeta(aim_run_hash=ORIGIN))
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            result = start_eval_run_from_checkpoint(checkpoint=state, task_set="glue")
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", ORIGIN)
        assert result.astrolabe_linked is True

    def test_falls_back_to_eval_task_set_outside_a_submit(self, tmp_path):
        ckpt = _ckpt(tmp_path, aim_run_hash=ORIGIN)
        factory = MagicMock(return_value=_make_run_mock())
        with patch("aim.Run", factory):
            start_eval_run_from_checkpoint(checkpoint=ckpt, task_set="mmlu")
        assert factory.call_args.kwargs["experiment"] == "eval/mmlu"


class TestSubmitIdentity:
    """An eval script launched as an astrolabe step inherits the submit's
    identity from the environment. Before this, that identity was in the
    process and thrown away: evals could not be filtered by submitter or
    repo, and the GPU time they burned billed to nothing.

    Filing follows the same env, so a submit's evals sit in the submit's
    experiment rather than in a per-benchmark bucket.
    """

    def test_files_under_the_submitting_experiment(self, monkeypatch):
        monkeypatch.setenv("ASTROLABE_EXPERIMENT_NAME", "latent-bert")
        factory = MagicMock(return_value=_make_run_mock())
        with patch("aim.Run", factory):
            start_eval_run(
                model_run_hash="abc", task_set="glue", aim_url="aim://test"
            )
        assert factory.call_args.kwargs["experiment"] == "latent-bert"

    def test_experiment_env_wins_over_the_tag_payload(self, monkeypatch):
        """Both carry an experiment; they disagree only if something
        upstream is inconsistent, and resolve_run_config already treats
        the dedicated env var as authoritative. Matching it keeps one
        answer to 'which experiment am I in' across the library."""
        monkeypatch.setenv("ASTROLABE_EXPERIMENT_NAME", "authoritative")
        monkeypatch.setenv("AIM_RUN_TAGS", "astrolabe.experiment=stale")
        run = _make_run_mock()
        factory = MagicMock(return_value=run)
        with patch("aim.Run", factory):
            start_eval_run(
                model_run_hash="abc", task_set="glue", aim_url="aim://test"
            )
        assert factory.call_args.kwargs["experiment"] == "authoritative"
        run.__setitem__.assert_any_call("astrolabe.experiment", "authoritative")

    def test_blank_tag_values_are_not_written(self, monkeypatch):
        """A tag present but empty is worse than absent — the dashboard
        renders it as a real value, so an empty submitter reads as a
        submitter named nothing rather than an unattributed run."""
        monkeypatch.setenv(
            "AIM_RUN_TAGS", "astrolabe.user=,astrolabe.submit_id=s-1"
        )
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            start_eval_run(
                model_run_hash="abc", task_set="glue", aim_url="aim://test"
            )
        written = [c.args[0] for c in run.__setitem__.call_args_list]
        assert "astrolabe.user" not in written
        assert "astrolabe.submit_id" in written

    def test_malformed_tag_payload_does_not_break_the_eval(self, monkeypatch):
        """parse_aim_run_tags is deliberately forgiving. An eval that has
        already run should still record its scores, identity or not."""
        monkeypatch.setenv("AIM_RUN_TAGS", "garbage-with-no-equals")
        run = _make_run_mock()
        factory = MagicMock(return_value=run)
        with patch("aim.Run", factory):
            start_eval_run(
                model_run_hash="abc", task_set="glue", aim_url="aim://test"
            )
        assert factory.call_args.kwargs["experiment"] == "eval/glue"
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", "abc")

    def test_identity_tags_land_on_the_run(self, monkeypatch):
        monkeypatch.setenv(
            "AIM_RUN_TAGS",
            "astrolabe.submit_id=s-9,astrolabe.version=v2,"
            "astrolabe.user=nathan,astrolabe.gpu_rate_cents_per_hour=250",
        )
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            start_eval_run(
                model_run_hash="abc", task_set="glue", aim_url="aim://test"
            )
        run.__setitem__.assert_any_call("astrolabe.submit_id", "s-9")
        run.__setitem__.assert_any_call("astrolabe.version", "v2")
        run.__setitem__.assert_any_call("astrolabe.user", "nathan")
        run.__setitem__.assert_any_call(
            "astrolabe.gpu_rate_cents_per_hour", "250"
        )

    def test_the_contract_tags_survive_a_colliding_identity(self, monkeypatch):
        """The discovery tags are the only reason the dashboard can see an
        eval run. An unexpected key in the ambient payload shadowing one
        would make the run vanish, so identity is applied first and the
        contract tags overwrite it."""
        monkeypatch.setenv(
            "AIM_RUN_TAGS",
            "astrolabe.kind=metadata,astrolabe.task_set=wrong,"
            "astrolabe.model_run_hash=hijacked",
        )
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            start_eval_run(
                model_run_hash="real-model", task_set="glue", aim_url="aim://test"
            )
        final = dict(c.args for c in run.__setitem__.call_args_list)
        assert final["astrolabe.kind"] == "eval"
        assert final["astrolabe.task_set"] == "glue"
        assert final["astrolabe.model_run_hash"] == "real-model"

    def test_an_unlinked_run_gets_the_same_filing_and_identity(
        self, tmp_path, monkeypatch
    ):
        """A run stamped with its model later must be indistinguishable
        from one that resolved on the first try."""
        monkeypatch.setenv("ASTROLABE_EXPERIMENT_NAME", "latent-bert")
        monkeypatch.setenv("AIM_RUN_TAGS", "astrolabe.submit_id=s-3")
        ckpt = _ckpt(tmp_path)
        run = _make_run_mock()
        factory = MagicMock(return_value=run)
        with patch("aim.Run", factory):
            start_eval_run_from_checkpoint(
                checkpoint=ckpt, task_set="glue", on_missing_parent="warn"
            )
        assert factory.call_args.kwargs["experiment"] == "latent-bert"
        run.__setitem__.assert_any_call("astrolabe.submit_id", "s-3")

    def test_log_eval_table_inherits_it(self, monkeypatch):
        monkeypatch.setenv("ASTROLABE_EXPERIMENT_NAME", "latent-bert")
        monkeypatch.setenv("AIM_RUN_TAGS", "astrolabe.user=nathan")
        run = _make_run_mock()
        factory = MagicMock(return_value=run)
        with patch("aim.Run", factory):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url="aim://test",
            )
        assert factory.call_args.kwargs["experiment"] == "latent-bert"
        run.__setitem__.assert_any_call("astrolabe.user", "nathan")


class TestRaisesByDefault:
    """The worst outcome in this area used to be the default: warn wrote
    an eval run with no model_run_hash, which lands in Aim and is
    invisible to the dashboard forever. An hour of GPU time producing
    numbers nobody can find."""

    def test_default_raises_rather_than_writing_an_orphan(self, tmp_path):
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        factory = MagicMock()
        with patch("aim.Run", factory):
            with pytest.raises(MissingParentError):
                start_eval_run_from_checkpoint(checkpoint=plain, task_set="glue")
        factory.assert_not_called()

    def test_the_error_names_both_ways_out(self, tmp_path):
        """It fires before any scoring, so the message is the whole
        remedy — it has to cover the trained case and the downloaded
        case, not just one."""
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        with patch("aim.Run", MagicMock()):
            with pytest.raises(MissingParentError) as exc:
                start_eval_run_from_checkpoint(checkpoint=plain, task_set="glue")
        assert "model_run_hash" in str(exc.value)
        assert "external_name" in str(exc.value)


class TestExternalName:
    """Scoring a model astrolabe never trained."""

    def test_registers_the_model_and_links_the_eval_to_it(self, tmp_path):
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        entry, evalrun = _make_run_mock("entry-hash"), _make_run_mock("eval-hash")
        with patch("aim.Run", side_effect=[entry, evalrun]):
            result = start_eval_run_from_checkpoint(
                checkpoint=plain, task_set="glue", external_name="roberta-base"
            )
        entry.__setitem__.assert_any_call(
            "astrolabe.kind", "external_checkpoint"
        )
        assert entry.name == "roberta-base"
        evalrun.__setitem__.assert_any_call(
            "astrolabe.model_run_hash", "entry-hash"
        )
        assert result.astrolabe_linked is True

    def test_the_entry_is_closed_so_it_does_not_read_as_in_flight(self, tmp_path):
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        entry, evalrun = _make_run_mock("entry-hash"), _make_run_mock("eval-hash")
        with patch("aim.Run", side_effect=[entry, evalrun]):
            start_eval_run_from_checkpoint(
                checkpoint=plain, task_set="glue", external_name="roberta-base"
            )
        entry.close.assert_called_once()

    def test_provenance_wins_over_the_name(self, tmp_path):
        """A sweep over mixed models passes external_name every time;
        refusing the overlap would break the obvious loop. The file knows
        better than the argument."""
        ckpt = _ckpt(tmp_path, aim_run_hash=ORIGIN)
        run = _make_run_mock()
        factory = MagicMock(return_value=run)
        with patch("aim.Run", factory):
            start_eval_run_from_checkpoint(
                checkpoint=ckpt, task_set="glue", external_name="roberta-base"
            )
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", ORIGIN)
        # One run: the eval. No entry was registered.
        assert factory.call_count == 1

    def test_explicit_hash_wins_over_the_name(self, tmp_path):
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        run = _make_run_mock()
        factory = MagicMock(return_value=run)
        with patch("aim.Run", factory):
            start_eval_run_from_checkpoint(
                checkpoint=plain, task_set="glue",
                model_run_hash="explicit", external_name="roberta-base",
            )
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", "explicit")
        assert factory.call_count == 1

    @pytest.mark.parametrize("bad", ["", "   ", None, 7])
    def test_register_rejects_an_unusable_name(self, bad):
        factory = MagicMock()
        with patch("aim.Run", factory):
            with pytest.raises(EvalInputError, match="name"):
                register_external_model(name=bad)
        factory.assert_not_called()

    def test_never_reads_from_aim(self, tmp_path):
        """The lookup this replaces could not work under local-aim
        transport, where compute sees only its own submit's runs — it
        would find nothing and mint a duplicate silently."""
        entry = _make_run_mock("entry-hash")
        with patch("aim.Run", return_value=entry), patch("aim.Repo") as repo:
            register_external_model(name="roberta-base")
        repo.assert_not_called()

    def test_entry_carries_the_submit_identity(self, monkeypatch):
        """It files under the submitting experiment, so it is one of that
        experiment's own rows — a row with no version would sit outside
        every version group and vanish from the page."""
        monkeypatch.setenv("ASTROLABE_EXPERIMENT_NAME", "latent-bert")
        monkeypatch.setenv("AIM_RUN_TAGS", "astrolabe.version=v3")
        entry = _make_run_mock("entry-hash")
        factory = MagicMock(return_value=entry)
        with patch("aim.Run", factory):
            register_external_model(name="roberta-base")
        assert factory.call_args.kwargs["experiment"] == "latent-bert"
        entry.__setitem__.assert_any_call("astrolabe.version", "v3")


class TestPublicExports:
    """Every eval helper a user is told to import must be importable from
    the package root.

    `register_external_model` shipped in `eval_results.__all__` but was
    never added to the package's re-export list, so the documented
    `from astrolabe_callbacks import register_external_model` raised
    ImportError. Nothing caught it: the module's own tests import from
    `astrolabe_callbacks.eval_results` directly, which worked fine.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "log_eval_table",
            "start_eval_run",
            "start_eval_run_from_checkpoint",
            "register_external_model",
            "EvalInputError",
            "MissingParentError",
        ],
    )
    def test_importable_from_package_root(self, name):
        import astrolabe_callbacks

        assert hasattr(astrolabe_callbacks, name), (
            f"{name} is in eval_results.__all__ but not re-exported from "
            f"astrolabe_callbacks — the documented import fails"
        )
        assert name in astrolabe_callbacks.__all__
