"""Tests for ``AlidadeRun`` (raw-PyTorch context manager)."""

from __future__ import annotations

import pytest

from alidade_callbacks import Run
from alidade_callbacks.pytorch import AlidadeRun


# ----------------------------------------------------------------------
# Run alias
# ----------------------------------------------------------------------


class TestRunAlias:
    def test_run_is_alias_of_alidade_run(self):
        # `from alidade_callbacks import Run` is the documented short
        # form; if we ever rename or wrap AlidadeRun, this guards
        # against silently breaking the alias.
        assert Run is AlidadeRun


# ----------------------------------------------------------------------
# Constructor
# ----------------------------------------------------------------------


class TestConstructor:
    def test_env_experiment_name_wins(self, monkeypatch):
        monkeypatch.setenv("ALIDADE_EXPERIMENT_NAME", "from-env")
        r = AlidadeRun(experiment_name="from-arg")
        assert r._cfg.experiment_name == "from-env"

    def test_env_aim_url_wins(self, monkeypatch):
        monkeypatch.setenv("ALIDADE_AIM_URL", "aim://from-env:9999")
        r = AlidadeRun(aim_url="aim://from-arg:1111")
        assert r._cfg.aim_url == "aim://from-env:9999"

    def test_default_aim_url(self):
        r = AlidadeRun()
        assert r._cfg.aim_url == "aim://localhost:43800"


# ----------------------------------------------------------------------
# Context-manager lifecycle
# ----------------------------------------------------------------------


class TestContextManagerEdgeCases:
    def test_aim_missing_does_not_break_enter_exit(self, monkeypatch):
        # Connection failure → context manager still works, methods
        # all no-op gracefully.
        import sys
        monkeypatch.setitem(sys.modules, "aim", None)
        with AlidadeRun() as run:
            assert run.is_active is False
            run.log_train(loss=0.5, step=1)  # must not raise
            run.log_eval(loss=0.4, step=1)
            run.log("custom", 1.0)
            run.set_tag("k", "v")

    def test_strict_mode_raises_on_connection_failure(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "aim", None)
        monkeypatch.setenv("ALIDADE_CALLBACK_STRICT", "1")
        with pytest.raises(RuntimeError, match="aim not installed"):
            with AlidadeRun():
                pass

    def test_exception_inside_block_marks_failed(self, fake_aim_run):
        try:
            with AlidadeRun() as run:
                run.log_train(loss=0.5, step=1)
                raise ValueError("training crashed")
        except ValueError:
            pass

        assert fake_aim_run[-1].tags["alidade.status"] == "failed"
        assert fake_aim_run[-1].closed is True

    def test_clean_exit_marks_completed(self, fake_aim_run):
        with AlidadeRun() as run:
            run.log_train(loss=0.5, step=1)
        assert fake_aim_run[-1].tags["alidade.status"] == "completed"

    def test_methods_after_exit_are_noops(self, fake_aim_run):
        with AlidadeRun() as run:
            run.log_train(loss=0.5, step=1)

        tracked_count_before = len(fake_aim_run[-1].tracked)
        # Methods after exit must be no-ops; no raises and no new
        # tracked entries.
        run.log_train(loss=0.6, step=2)
        run.log_eval(loss=0.4, step=2)
        run.log("custom", 1.0)
        assert len(fake_aim_run[-1].tracked) == tracked_count_before

    def test_non_rank_zero_no_run_opened(self, monkeypatch, fake_aim_run):
        monkeypatch.setenv("RANK", "1")
        with AlidadeRun() as run:
            assert run.is_active is False
            run.log_train(loss=0.5, step=1)
        # No FakeAimRun was constructed at all.
        assert fake_aim_run == []


class TestContextManagerHappyPath:
    def test_basic_flow(self, fake_aim_run):
        with AlidadeRun(experiment_name="exp", tags={"k": "v"}) as run:
            assert run.is_active is True
        # Tags applied
        assert fake_aim_run[-1].tags["k"] == "v"
        # Status written and run closed
        assert fake_aim_run[-1].tags["alidade.status"] == "completed"
        assert fake_aim_run[-1].closed is True

    def test_run_name_propagates(self, fake_aim_run):
        with AlidadeRun(run_name="bert-tiny"):
            pass
        assert fake_aim_run[-1].name == "bert-tiny"


# ----------------------------------------------------------------------
# log_train — train/ namespace + wall_time
# ----------------------------------------------------------------------


class TestLogTrain:
    def test_namespaces_under_train_prefix(self, fake_aim_run):
        with AlidadeRun() as run:
            run.log_train(loss=0.5, accuracy=0.9, step=1)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "train/loss" in names
        assert "train/accuracy" in names

    def test_logs_wall_time_alongside_metrics(self, fake_aim_run):
        with AlidadeRun() as run:
            run.log_train(loss=0.5, step=1)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "wall_time" in names

    def test_step_passed_through(self, fake_aim_run):
        with AlidadeRun() as run:
            run.log_train(loss=0.5, step=42)
        loss_entry = next(
            t for t in fake_aim_run[-1].tracked if t["name"] == "train/loss"
        )
        assert loss_entry["step"] == 42

    def test_step_optional(self, fake_aim_run):
        with AlidadeRun() as run:
            run.log_train(loss=0.5)
        loss_entry = next(
            t for t in fake_aim_run[-1].tracked if t["name"] == "train/loss"
        )
        assert loss_entry["step"] is None

    def test_no_metrics_still_logs_wall_time(self, fake_aim_run):
        # Edge: user calls log_train with no kwargs (just for wall_time).
        with AlidadeRun() as run:
            run.log_train(step=1)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        # Only wall_time gets logged.
        assert names == ["wall_time"]


# ----------------------------------------------------------------------
# log_eval — EVAL_PREFIX namespace
# ----------------------------------------------------------------------


class TestLogEval:
    def test_namespaces_under_eval_prefix(self, fake_aim_run):
        from alidade_callbacks._core import EVAL_METRIC_PREFIX
        with AlidadeRun() as run:
            run.log_eval(loss=0.4, accuracy=0.95, step=10)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert f"{EVAL_METRIC_PREFIX}/loss" in names
        assert f"{EVAL_METRIC_PREFIX}/accuracy" in names

    def test_does_not_log_wall_time(self, fake_aim_run):
        # Eval uses the wall_time anchored at training; no fresh
        # wall_time write on each log_eval (otherwise wall_time would
        # double-log on alternating train/eval).
        with AlidadeRun() as run:
            run.log_eval(loss=0.4, step=10)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "wall_time" not in names


# ----------------------------------------------------------------------
# log — escape hatch
# ----------------------------------------------------------------------


class TestLogEscapeHatch:
    def test_passes_name_through_unchanged(self, fake_aim_run):
        with AlidadeRun() as run:
            run.log("custom/throughput", 42.0, step=1)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "custom/throughput" in names

    def test_context_passed_through(self, fake_aim_run):
        with AlidadeRun() as run:
            run.log("metric", 1.0, step=1, context={"subset": "val"})
        entry = next(
            t for t in fake_aim_run[-1].tracked if t["name"] == "metric"
        )
        assert entry["context"] == {"subset": "val"}


# ----------------------------------------------------------------------
# set_tag — late-binding tags
# ----------------------------------------------------------------------


class TestSetTag:
    def test_writes_to_run(self, fake_aim_run):
        with AlidadeRun() as run:
            run.set_tag("late.tag", "computed_value")
        assert fake_aim_run[-1].tags["late.tag"] == "computed_value"

    def test_no_run_is_noop(self):
        r = AlidadeRun()
        r.set_tag("x", "y")  # not entered yet — must not raise

    def test_strict_mode_raises_on_failure(self, monkeypatch):
        class FailingRun:
            def __init__(self, **kw): pass
            def __setitem__(self, k, v): raise RuntimeError("aim broke")
            def track(self, *a, **kw): pass
            def close(self): pass

        monkeypatch.setattr("aim.Run", FailingRun)
        monkeypatch.setenv("ALIDADE_CALLBACK_STRICT", "1")
        with pytest.raises(RuntimeError, match="aim broke"):
            with AlidadeRun() as run:
                run.set_tag("k", "v")


# ----------------------------------------------------------------------
# pause_eval / resume — wall-time control
# ----------------------------------------------------------------------


class TestPauseResume:
    def test_pause_and_resume_safe_when_not_active(self):
        # Before entering the context, pause/resume must be safe.
        r = AlidadeRun()
        r.pause_eval()
        r.resume()  # no raise

    def test_pause_excludes_eval_time_from_wall_time(self, fake_aim_run):
        import time

        with AlidadeRun() as run:
            run.log_train(loss=0.5, step=1)  # anchor wall_time
            wall_time_1 = next(
                t["value"]
                for t in fake_aim_run[-1].tracked
                if t["name"] == "wall_time"
            )
            run.pause_eval()
            time.sleep(0.05)  # 50ms "in eval"
            run.resume()
            run.log_train(loss=0.4, step=2)
            # The second wall_time should be close to the first (eval
            # time excluded), not 50ms+ later.
            wall_time_2 = next(
                t["value"]
                for t in fake_aim_run[-1].tracked
                if t["name"] == "wall_time" and t["step"] == 2
            )

        # Wall_time 2 - 1 should be small (training time only),
        # not include the 50ms sleep.
        assert (wall_time_2 - wall_time_1) < 0.04


# ----------------------------------------------------------------------
# is_active property
# ----------------------------------------------------------------------


class TestIsActive:
    def test_false_before_entering(self):
        r = AlidadeRun()
        assert r.is_active is False

    def test_true_inside_context(self, fake_aim_run):
        with AlidadeRun() as run:
            assert run.is_active is True

    def test_false_after_exit(self, fake_aim_run):
        with AlidadeRun() as run:
            pass
        assert run.is_active is False
