"""Tests for ``AlidadeHFTrainerCallback``.

HF Trainer's ``on_log`` hook receives a pre-aggregated metrics dict.
We pass through everything; specific keys (``loss``, ``learning_rate``,
``grad_norm``, ``epoch``) get re-namespaced under ``train/`` for
display ergonomics. ``eval_*`` keys get re-namespaced under the
canonical eval prefix.
"""

from __future__ import annotations

import pytest

from alidade_callbacks._core import EVAL_METRIC_PREFIX
from alidade_callbacks.huggingface import (
    AlidadeHFTrainerCallback,
    _normalize_log_key,
)


# ----------------------------------------------------------------------
# Helpers: minimal HF stand-ins
# ----------------------------------------------------------------------


class _FakeTrainerArgs:
    def __init__(self, run_name: str | None = None, output_dir: str | None = None):
        self.run_name = run_name
        self.output_dir = output_dir


class _FakeState:
    def __init__(self, global_step: int = 0):
        self.global_step = global_step


class _FakeControl:
    pass


# ----------------------------------------------------------------------
# Key normalizer — edge cases
# ----------------------------------------------------------------------


class TestNormalizeLogKeyEdgeCases:
    def test_empty_eval_suffix(self):
        # "eval_" with empty suffix → None (skipped to avoid empty
        # metric names that Aim would refuse).
        assert _normalize_log_key("eval_") is None

    def test_passes_through_unknown_train_keys(self):
        # User-added custom training metric. Must NOT be auto-prefixed
        # — the user's name is theirs, not ours to rewrite.
        assert _normalize_log_key("my_custom_metric") == "my_custom_metric"

    def test_passes_through_namespaced_user_metrics(self):
        assert _normalize_log_key("custom/throughput") == "custom/throughput"

    def test_eval_with_underscored_suffix(self):
        # "eval_my_custom" → "eval/my_custom" (whole suffix preserved).
        assert _normalize_log_key("eval_my_custom") == f"{EVAL_METRIC_PREFIX}/my_custom"

    def test_uppercase_eval_not_recognized(self):
        # Case-sensitive. HF emits lowercase.
        assert _normalize_log_key("EVAL_loss") == "EVAL_loss"

    def test_just_eval_no_underscore(self):
        # "eval" alone is not "eval_" so passes through unchanged.
        assert _normalize_log_key("eval") == "eval"


class TestNormalizeLogKeyHappyPath:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("loss", "train/loss"),
            ("learning_rate", "train/lr"),
            ("grad_norm", "train/grad_norm"),
            ("epoch", "train/epoch"),
            ("eval_loss", f"{EVAL_METRIC_PREFIX}/loss"),
            ("eval_accuracy", f"{EVAL_METRIC_PREFIX}/accuracy"),
            ("eval_f1", f"{EVAL_METRIC_PREFIX}/f1"),
            ("custom", "custom"),  # user-defined passes through
        ],
    )
    def test_canonical_renames(self, raw, expected):
        assert _normalize_log_key(raw) == expected


# ----------------------------------------------------------------------
# Constructor + on_train_begin
# ----------------------------------------------------------------------


class TestOnTrainBegin:
    def test_opens_run_with_explicit_name(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback(run_name="my-run")
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        assert fake_aim_run[-1].name == "my-run"

    def test_run_name_from_args(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(
            _FakeTrainerArgs(run_name="hf-args-run-name"),
            _FakeState(),
            _FakeControl(),
        )
        assert fake_aim_run[-1].name == "hf-args-run-name"

    def test_run_name_from_output_dir(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(
            _FakeTrainerArgs(output_dir="/tmp/checkpoints/bert-v3"),
            _FakeState(),
            _FakeControl(),
        )
        # basename of the output_dir.
        assert fake_aim_run[-1].name == "bert-v3"

    def test_run_name_chain_explicit_wins(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback(run_name="explicit")
        cb.on_train_begin(
            _FakeTrainerArgs(run_name="from-args", output_dir="/tmp/from-dir"),
            _FakeState(),
            _FakeControl(),
        )
        assert fake_aim_run[-1].name == "explicit"

    def test_double_on_train_begin_no_op(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        first_run = cb._run
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        assert cb._run is first_run
        assert len(fake_aim_run) == 1

    def test_rank_nonzero_no_run(self, monkeypatch, fake_aim_run):
        monkeypatch.setenv("RANK", "3")
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        assert fake_aim_run == []
        assert cb._run is None


# ----------------------------------------------------------------------
# on_log — primary metric flow
# ----------------------------------------------------------------------


class TestOnLogEdgeCases:
    def test_no_logs_no_op(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_log(_FakeTrainerArgs(), _FakeState(), _FakeControl(), logs=None)
        cb.on_log(_FakeTrainerArgs(), _FakeState(), _FakeControl(), logs={})
        # Only wall_time would be logged if logs were non-empty; here
        # nothing.
        assert fake_aim_run[-1].tracked == []

    def test_skips_non_numeric_values(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(global_step=10), _FakeControl())
        cb.on_log(
            _FakeTrainerArgs(),
            _FakeState(global_step=10),
            _FakeControl(),
            logs={"loss": 0.5, "tag": "experiment-name"},
        )
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "train/loss" in names
        # Strings can't coerce to float → silently skipped.
        assert "tag" not in names

    def test_no_run_is_noop(self):
        cb = AlidadeHFTrainerCallback()
        # on_train_begin not called → _run stays None
        cb.on_log(
            _FakeTrainerArgs(),
            _FakeState(),
            _FakeControl(),
            logs={"loss": 0.5},
        )  # must not raise

    def test_rank_nonzero_is_noop(self, monkeypatch, fake_aim_run):
        monkeypatch.setenv("RANK", "1")
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_log(
            _FakeTrainerArgs(),
            _FakeState(global_step=10),
            _FakeControl(),
            logs={"loss": 0.5},
        )
        assert fake_aim_run == []


class TestOnLogHappyPath:
    def test_passes_through_user_metrics(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(global_step=0), _FakeControl())
        cb.on_step_end(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_log(
            _FakeTrainerArgs(),
            _FakeState(global_step=10),
            _FakeControl(),
            logs={
                "loss": 0.5,
                "learning_rate": 1e-4,
                "my_custom_metric": 99.0,
            },
        )
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "train/loss" in names
        assert "train/lr" in names
        assert "my_custom_metric" in names  # user-named pass-through
        assert "wall_time" in names

    def test_step_carries_through(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_log(
            _FakeTrainerArgs(),
            _FakeState(global_step=42),
            _FakeControl(),
            logs={"loss": 0.5},
        )
        loss_entry = next(
            t for t in fake_aim_run[-1].tracked if t["name"] == "train/loss"
        )
        assert loss_entry["step"] == 42


# ----------------------------------------------------------------------
# on_evaluate — eval-side pass-through
# ----------------------------------------------------------------------


class TestOnEvaluate:
    def test_eval_keys_renamed(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_evaluate(
            _FakeTrainerArgs(),
            _FakeState(global_step=100),
            _FakeControl(),
            metrics={"eval_loss": 0.4, "eval_accuracy": 0.95},
        )
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert f"{EVAL_METRIC_PREFIX}/loss" in names
        assert f"{EVAL_METRIC_PREFIX}/accuracy" in names

    def test_no_metrics_is_noop(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_evaluate(
            _FakeTrainerArgs(), _FakeState(), _FakeControl(), metrics=None
        )
        assert fake_aim_run[-1].tracked == []


# ----------------------------------------------------------------------
# on_train_end
# ----------------------------------------------------------------------


class TestOnTrainEnd:
    def test_closes_with_completed(self, fake_aim_run):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        run = cb._run
        cb.on_train_end(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        assert run.tags["alidade.status"] == "completed"
        assert run.closed is True
        assert cb._run is None

    def test_no_run_is_noop(self):
        cb = AlidadeHFTrainerCallback()
        cb.on_train_end(
            _FakeTrainerArgs(), _FakeState(), _FakeControl()
        )  # must not raise


# ----------------------------------------------------------------------
# Trainer's end-of-training summary
# ----------------------------------------------------------------------

_SUMMARY = {
    "train_runtime": 19.3,
    "train_samples_per_second": 15.5,
    "train_steps_per_second": 15.5,
    "total_flos": 0.0,
    "train_loss": 2.955,
    "epoch": 1.0,
}


def _names_logged(fake_aim_run, logs):
    cb = AlidadeHFTrainerCallback()
    cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
    cb.on_log(_FakeTrainerArgs(), _FakeState(global_step=300), _FakeControl(), logs=logs)
    return {t["name"] for t in fake_aim_run[-1].tracked}


class TestTheTrainingSummaryIsNotLogged:
    def test_none_of_its_totals_become_charts(self, fake_aim_run):
        names = _names_logged(fake_aim_run, dict(_SUMMARY))
        assert not names & set(_SUMMARY) - {"epoch"}, names

    def test_a_train_loss_the_user_logs_still_lands(self, fake_aim_run):
        # Only the summary record is recognised, by its runtime key.
        names = _names_logged(fake_aim_run, {"train_loss": 2.9})
        assert "train_loss" in names

    def test_the_summary_epoch_still_lands(self, fake_aim_run):
        names = _names_logged(fake_aim_run, dict(_SUMMARY))
        assert "train/epoch" in names


# ----------------------------------------------------------------------
# wall_time across an eval pass
# ----------------------------------------------------------------------


class _Control:
    def __init__(self, should_evaluate: bool = False):
        self.should_evaluate = should_evaluate


def _wall_times(run):
    return [(t["step"], t["value"]) for t in run.tracked if t["name"] == "wall_time"]


def _started(clock):
    cb = AlidadeHFTrainerCallback()
    cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
    cb.on_step_end(_FakeTrainerArgs(), _FakeState(global_step=1), _Control())
    clock(10)
    return cb


class TestAnEvalCarriesTheWallTimeOfItsStep:
    def test_eval_metrics_logged_mid_eval_read_the_step_s_time(
        self, fake_aim_run, clock
    ):
        cb = _started(clock)
        cb.on_step_end(_FakeTrainerArgs(), _FakeState(global_step=2), _Control(True))
        clock(60)
        cb.on_log(_FakeTrainerArgs(), _FakeState(global_step=2), _Control(), logs={"eval_loss": 0.4})
        assert _wall_times(fake_aim_run[-1]) == [(2, 10)]

    def test_the_next_step_does_not_count_the_eval(self, fake_aim_run, clock):
        cb = _started(clock)
        cb.on_step_end(_FakeTrainerArgs(), _FakeState(global_step=2), _Control(True))
        clock(60)
        cb.on_evaluate(_FakeTrainerArgs(), _FakeState(global_step=2), _Control(), metrics={"eval_loss": 0.4})
        clock(1)
        cb.on_log(_FakeTrainerArgs(), _FakeState(global_step=3), _Control(), logs={"loss": 0.3})
        assert _wall_times(fake_aim_run[-1]) == [(3, 11)]

    def test_an_epoch_end_eval_is_paused_too(self, fake_aim_run, clock):
        cb = _started(clock)
        cb.on_epoch_end(_FakeTrainerArgs(), _FakeState(global_step=2), _Control(True))
        clock(60)
        cb.on_log(_FakeTrainerArgs(), _FakeState(global_step=2), _Control(), logs={"eval_loss": 0.4})
        assert _wall_times(fake_aim_run[-1]) == [(2, 10)]

    def test_a_pass_that_never_reports_does_not_stop_the_clock(
        self, fake_aim_run, clock
    ):
        cb = _started(clock)
        cb.on_step_end(_FakeTrainerArgs(), _FakeState(global_step=2), _Control(True))
        clock(60)
        cb.on_step_begin(_FakeTrainerArgs(), _FakeState(global_step=2), _Control())
        clock(1)
        cb.on_log(_FakeTrainerArgs(), _FakeState(global_step=3), _Control(), logs={"loss": 0.3})
        assert _wall_times(fake_aim_run[-1]) == [(3, 11)]

    def test_a_step_with_no_eval_due_does_not_pause(self, fake_aim_run, clock):
        cb = _started(clock)
        cb.on_step_end(_FakeTrainerArgs(), _FakeState(global_step=2), _Control(False))
        clock(5)
        cb.on_log(_FakeTrainerArgs(), _FakeState(global_step=2), _Control(), logs={"loss": 0.3})
        assert _wall_times(fake_aim_run[-1]) == [(2, 15)]


def test_a_real_trainer_masks_eval_and_logs_no_summary(fake_aim_run, tmp_path):
    """HF's own hook order, which the hand-driven tests above assume."""
    pytest.importorskip("transformers")
    import time

    import torch
    import torch.nn as nn
    from transformers import Trainer, TrainingArguments

    eval_s = 1.5

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, input_ids=None, labels=None, **kwargs):
            if not self.training:
                time.sleep(eval_s)
            out = self.lin(input_ids.float())
            return {"loss": ((out - labels.float()) ** 2).mean(), "logits": out}

    class Rows(torch.utils.data.Dataset):
        def __init__(self, n):
            self.x, self.y = torch.randn(n, 4), torch.randn(n, 1)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return {"input_ids": self.x[i], "labels": self.y[i]}

    args = TrainingArguments(
        output_dir=str(tmp_path), max_steps=4, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, logging_steps=1, eval_strategy="steps",
        eval_steps=2, disable_tqdm=True, report_to=[], use_cpu=True, save_strategy="no",
    )
    Trainer(
        model=Toy(), args=args, train_dataset=Rows(4), eval_dataset=Rows(1),
        callbacks=[AlidadeHFTrainerCallback()],
    ).train()

    tracked = [t for run in fake_aim_run for t in run.tracked]
    walls = [(t["step"], t["value"]) for t in tracked if t["name"] == "wall_time"]
    names = {t["name"] for t in tracked}
    assert f"{EVAL_METRIC_PREFIX}/loss" in names
    assert not names & {"train_loss", "train_runtime", "total_flos"}, names
    assert [v for _, v in walls] == sorted(v for _, v in walls), walls
    # Per step, not in total: a slow runner's warm-up is training time.
    jumps = [later - earlier for (_, earlier), (_, later) in zip(walls, walls[1:])]
    assert max(jumps) < eval_s / 2, f"an eval pass leaked into {walls}"
