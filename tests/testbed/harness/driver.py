"""Callback-library exerciser — runs inside the ``client`` container.

Dispatches by ``config.framework`` to one of four drivers:

- ``raw``: no framework, no model, no torch. Directly opens
  ``AstrolabeRun`` (from ``pytorch.py``) and drives ``track()`` /
  ``close()`` in a loop. Used by ``test_core.py``, ``test_pytorch.py``,
  ``test_distributed.py`` — those tests are about the callback's
  internal behavior (buffer, drainer, schema-finalize, rank gating),
  not about how a framework calls into it.
- ``composer``: **real** Composer ``Trainer`` fitting a single
  ``nn.Linear(4, 1)`` model on deterministic fake data. The Composer
  event loop drives ``AstrolabeComposerLogger`` for real; batch_end,
  epoch_end, close fire the way they do in production.
- ``lightning``: real Lightning ``Trainer`` with a tiny
  ``LightningModule``. Drives ``AstrolabeLightningLogger`` via the
  actual Lightning event loop.
- ``hf``: real HuggingFace ``Trainer`` with a tiny model and a toy
  dataset. Drives ``AstrolabeHFTrainerCallback`` via the real HF
  event loop.

Why real training for framework paths: the callback contracts are
about "did the framework's real event loop pass the right args to the
callback in the right sequence?" A fake framework that just calls
``callback.batch_end(...)`` misses argument-shape bugs the real
integration would catch. Real training is ~15 lines per framework and
fully deterministic (seeded torch, fixed tiny data, CPU only).

Why the raw path stays exerciser-shaped: there's no framework to
drive. The tests are exercising ``AstrolabeRun.track()`` directly.
Adding a tiny linear layer would be theater — it wouldn't test
anything the direct call doesn't already.

Fault-injection driver_flags (``TESTBED_KILL_DRAINER_AT``,
``TESTBED_INJECT_TRANSIENT_ERROR_AT``, ``TESTBED_BUFFER_CAPACITY``,
etc.) apply on top of whichever driver mode — real training + injected
drainer death at step N, for example.

Invoked as a subprocess by ``compose.exec_in`` from the pytest
harness. Config comes in via env vars (``TESTBED_*``); result comes
out via stdout (``ASTROLABE_RUN_HASH=<hash>``) + parsed stats jsonl.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


__all__ = [
    "DriverConfig",
    "DriverResult",
    "SimulatedFailure",
    "Framework",
    "config_to_env",
    "run_driver",
    "main",
]


Framework = Literal["raw", "composer", "lightning", "hf"]


@dataclass(frozen=True)
class DriverConfig:
    """See module docstring for field semantics."""

    framework: Framework
    steps: int
    metrics_per_step: int
    metrics_per_sec: float
    fail_at: int | None
    new_metrics_at: list[int]
    validation_at: list[int]
    close: bool
    aim_url: str
    run_name: str
    experiment_name: str
    tags: dict[str, str]
    driver_flags: dict[str, str]
    stats_jsonl_container_path: str

    @classmethod
    def from_env(cls) -> "DriverConfig":
        """Build from ``TESTBED_*`` env vars. Missing required → SystemExit(2)."""
        def _req(key: str) -> str:
            v = os.environ.get(key)
            if v is None:
                print(f"missing required env var {key}", file=sys.stderr)
                raise SystemExit(2)
            return v

        return cls(
            framework=os.environ.get("TESTBED_FRAMEWORK", "raw"),  # type: ignore[arg-type]
            steps=int(_req("TESTBED_STEPS")),
            metrics_per_step=int(os.environ.get("TESTBED_METRICS_PER_STEP", "1")),
            metrics_per_sec=float(os.environ.get("TESTBED_METRICS_PER_SEC", "0.0")),
            fail_at=(int(os.environ["TESTBED_FAIL_AT"]) if os.environ.get("TESTBED_HAS_FAIL_AT") == "1" else None),
            new_metrics_at=json.loads(os.environ.get("TESTBED_NEW_METRICS_AT", "[]")),
            validation_at=json.loads(os.environ.get("TESTBED_VALIDATION_AT", "[]")),
            close=os.environ.get("TESTBED_CLOSE", "1") == "1",
            aim_url=_req("TESTBED_AIM_URL"),
            run_name=_req("TESTBED_RUN_NAME"),
            experiment_name=_req("TESTBED_EXPERIMENT_NAME"),
            tags=json.loads(os.environ.get("TESTBED_TAGS", "{}")),
            driver_flags=json.loads(os.environ.get("TESTBED_DRIVER_FLAGS", "{}")),
            stats_jsonl_container_path=_req("TESTBED_STATS_JSONL_CONTAINER_PATH"),
        )


@dataclass(frozen=True)
class DriverResult:
    exit_code: int
    run_hash: str | None
    stats_events: list[dict]
    stdout: str
    stderr: str
    marker_touched: bool = False  # ASTROLABE_FIRST_METRIC_MARKER file existed after run


class SimulatedFailure(RuntimeError):
    """Raised at ``fail_at`` step. Exit code 42."""


def config_to_env(config: DriverConfig) -> dict[str, str]:
    """Serialize ``config`` to a dict suitable for ``compose.exec_in(env=...)``."""
    env = {
        "TESTBED_FRAMEWORK": config.framework,
        "TESTBED_STEPS": str(config.steps),
        "TESTBED_METRICS_PER_STEP": str(config.metrics_per_step),
        "TESTBED_METRICS_PER_SEC": str(config.metrics_per_sec),
        "TESTBED_NEW_METRICS_AT": json.dumps(config.new_metrics_at),
        "TESTBED_VALIDATION_AT": json.dumps(config.validation_at),
        "TESTBED_CLOSE": "1" if config.close else "0",
        "TESTBED_AIM_URL": config.aim_url,
        "TESTBED_RUN_NAME": config.run_name,
        "TESTBED_EXPERIMENT_NAME": config.experiment_name,
        "TESTBED_TAGS": json.dumps(config.tags),
        "TESTBED_DRIVER_FLAGS": json.dumps(config.driver_flags),
        "TESTBED_STATS_JSONL_CONTAINER_PATH": config.stats_jsonl_container_path,
    }
    if config.fail_at is not None:
        env["TESTBED_FAIL_AT"] = str(config.fail_at)
        env["TESTBED_HAS_FAIL_AT"] = "1"

    # Pass callback-library env contract too
    env["ASTROLABE_AIM_URL"] = config.aim_url
    env["ASTROLABE_EXPERIMENT_NAME"] = config.experiment_name
    env["ASTROLABE_CALLBACK_STATS_PATH"] = config.stats_jsonl_container_path
    if config.tags:
        env["AIM_RUN_TAGS"] = ",".join(f"{k}={v}" for k, v in config.tags.items())

    # First-metric marker — always set, unique per invocation. Driver
    # checks the file after training and prints the touched-or-not
    # signal so scenarios can assert on it.
    marker_path = f"/tmp/testbed-first-metric-marker/{config.run_name}.tag"
    env["ASTROLABE_FIRST_METRIC_MARKER"] = marker_path

    # driver_flags cascade into env under their own names so downstream code
    # in the container can read them directly (e.g. rank-detection reads
    # RANK from a driver_flag "TESTBED_ENV_RANK" that maps to env RANK).
    for key, value in config.driver_flags.items():
        if key.startswith("TESTBED_ENV_"):
            env[key[len("TESTBED_ENV_") :]] = value
    return env


# ---------------------------------------------------------------------------
# Driver dispatch
# ---------------------------------------------------------------------------


def run_driver(config: DriverConfig) -> str | None:
    """Execute one driver invocation. Returns the Aim run hash (or None)."""
    _seed_all(int(os.environ.get("TESTBED_SEED", "0")))
    if config.framework == "raw":
        return _run_raw(config)

    # Framework paths: the callback library nulls its ``_run`` on close and
    # doesn't emit run_hash in normal stats events, so we can't read the
    # hash off the logger after trainer.fit() returns. Instead we query
    # the aim server for the run by name (scenarios set unique names).
    if config.framework == "composer":
        _run_composer(config)
    elif config.framework == "lightning":
        _run_lightning(config)
    elif config.framework == "hf":
        _run_hf(config)
    else:
        raise SystemExit(f"unknown framework: {config.framework}")
    return _find_recent_framework_run(config.aim_url, config.experiment_name, config.run_name)


def _find_recent_framework_run(aim_url: str, experiment_name: str, run_name: str) -> str | None:
    """Query the aim server for the most recently created run in ``experiment_name``.

    Framework loggers null their ``_run`` reference on close, so we can't
    read the hash from the logger object. Prefer name-match if a run with
    ``run_name`` exists (Lightning/HF accept run_name); fall back to the
    most recent run in the experiment (Composer's callback doesn't accept
    run_name so the aim run gets a random default name).
    """
    try:
        import aim

        repo = aim.Repo(aim_url)
        candidates = []
        for run_hash in repo.list_all_runs():
            run = aim.Run(run_hash, repo=aim_url, read_only=True)
            if run.name == run_name:
                return run_hash
            if run.experiment == experiment_name:
                candidates.append((run.creation_time, run_hash))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
    except Exception:
        pass
    return None


def _seed_all(seed: int) -> None:
    """Seed random / numpy / torch so framework paths are deterministic."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _metric_name(step: int, index: int, new_metrics_at: list[int]) -> str:
    """Deterministic metric-name schedule. Base metrics ``metric_0..N``; each step in ``new_metrics_at`` adds ``metric_new_step<step>``."""
    return f"metric_{index}"


# ---------------------------------------------------------------------------
# Fault injection — monkey-patches applied post-open, pre-write.
# Each hook is opt-in via a TESTBED_* driver flag.
# ---------------------------------------------------------------------------


def _apply_fault_injection(config: DriverConfig, run) -> None:
    """Wire fault-injection hooks onto the just-opened AstrolabeRun.

    Reads driver_flags and modifies the buffer / drainer as directed.
    Called AFTER ``run.__enter__`` — at that point ``run._run._astrolabe_buffer``
    is the drainer-owning ``_MetricBuffer`` instance the callback lib
    attached in ``_core.open_aim_run``.
    """
    if run._run is None:
        return  # not rank-zero, no buffer to patch
    buffer = getattr(run._run, "_astrolabe_buffer", None)
    if buffer is None:
        return

    # --- Buffer capacity override (TESTBED_BUFFER_CAPACITY=N) -----------
    # Replaces the internal queue with a smaller one so overflow fires
    # under normal load. Must happen before any writes.
    capacity = _int_flag(config, "TESTBED_BUFFER_CAPACITY")
    if capacity is not None:
        import queue as _q

        buffer._queue = _q.Queue(maxsize=capacity)
        # Reset any max-size-derived attrs on the buffer if present
        if hasattr(buffer, "_max_size"):
            buffer._max_size = capacity

    # --- Drainer delay (TESTBED_INJECT_DRAINER_DELAY=1) ------------------
    # Slows the drainer's aim.Run.track call so the queue fills faster
    # than it drains. Needed for TESTBED_BUFFER_CAPACITY overflow to
    # actually manifest under modest write rates. Wrapping track (not
    # the drain loop) means the sleep fires per-drained-item.
    if config.driver_flags.get("TESTBED_INJECT_DRAINER_DELAY") == "1":
        import time as _t
        import aim as _aim

        # Patch at the class level — some Aim versions have
        # __getattribute__ overrides that ignore instance-level track
        # overrides in the tracking-server code path. Class-level
        # setattr always wins for method resolution.
        orig_class_track = _aim.Run.track

        def _slow_class_track(self, *args, **kwargs):
            _t.sleep(0.05)
            return orig_class_track(self, *args, **kwargs)

        _aim.Run.track = _slow_class_track

    # NB: TESTBED_KILL_DRAINER_AT / TESTBED_INJECT_TRANSIENT_ERROR_AT are
    # not implementable via ``aim.Run.track`` monkey-patching — Aim's
    # ``@noexcept`` decorator swallows every exception before the callback
    # library's retry loop or the drainer's outer handler sees it. See
    # RED_FLAGS.md for the details. Scenarios that depend on those flags
    # stay ``@pytest.mark.skip``.


# ---------------------------------------------------------------------------
# Raw driver — direct AstrolabeRun
# ---------------------------------------------------------------------------


def _run_raw(config: DriverConfig) -> str | None:
    """Direct AstrolabeRun exerciser. No torch, no framework."""
    from astrolabe_callbacks.pytorch import AstrolabeRun
    from astrolabe_callbacks._distributed import is_rank_zero

    # Initialize torch.distributed if the scenario asked us to. Must
    # happen BEFORE is_rank_zero() — that's the whole point of
    # test_torch_distributed_wins_over_env (env RANK conflicts with
    # torch.distributed's rank; distributed should win).
    torch_dist_rank = _int_flag(config, "TESTBED_TORCH_DIST_RANK")
    if torch_dist_rank is not None:
        import torch.distributed as _dist
        import os as _os

        # Single-process distributed setup via file init. world_size=1
        # so we don't block waiting for other ranks. rank must be 0
        # then (torch.dist requires 0 ≤ rank < world_size). The scenario
        # sets torch_dist_rank=0 anyway; anything else would be a
        # test-authoring error.
        assert torch_dist_rank == 0, "world_size=1 requires rank=0"
        init_file = "/tmp/testbed-dist-init"
        try:
            _os.remove(init_file)
        except OSError:
            pass
        _dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            world_size=1,
            rank=0,
        )

    # Rank-gating: non-rank-zero returns None (no run opened).
    if not is_rank_zero():
        return None

    # Optional context-manager path. Hash captured OUTSIDE the with so it
    # survives an exception firing inside the block.
    use_ctx = config.driver_flags.get("TESTBED_USE_CONTEXT_MANAGER") == "1"
    if use_ctx:
        captured_hash: str | None = None
        try:
            with AstrolabeRun(
                aim_url=config.aim_url,
                experiment_name=config.experiment_name,
                tags=config.tags,
                run_name=config.run_name,
            ) as run:
                _apply_fault_injection(config, run)
                captured_hash = run._run.hash if run._run is not None else None
                # Emit the hash BEFORE the body — so it lands in stdout
                # even if _drive_raw_body raises SimulatedFailure.
                if captured_hash:
                    print(f"ASTROLABE_RUN_HASH={captured_hash}", flush=True)
                _drive_raw_body(config, run, captured_hash)
            return captured_hash
        except SimulatedFailure:
            # Re-raise; main() catches to emit exit code 42
            raise

    # Standard open/close path
    run = AstrolabeRun(
        aim_url=config.aim_url,
        experiment_name=config.experiment_name,
        tags=config.tags,
        run_name=config.run_name,
    )
    skip_init = config.driver_flags.get("TESTBED_SKIP_INIT") == "1"
    if not skip_init:
        run.__enter__()
        _apply_fault_injection(config, run)

    run_hash = run._run.hash if run._run is not None else None
    try:
        _drive_raw_body(config, run, run_hash)
    finally:
        if config.close:
            # Support the "double close" driver_flag
            run.__exit__(None, None, None)
            if config.driver_flags.get("TESTBED_DOUBLE_CLOSE") == "1":
                run.__exit__(None, None, None)
    return run_hash


def _drive_raw_body(config: DriverConfig, run, run_hash: str | None) -> None:
    """The shared write loop for raw mode (with/without context manager)."""
    close_at = _int_flag(config, "TESTBED_CLOSE_AT")
    reopen_at = _int_flag(config, "TESTBED_MID_REOPEN_AT")
    restart_aim_at = _int_flag(config, "TESTBED_RESTART_AIM_AT")

    sleep_per_step = (
        1.0 / config.metrics_per_sec if config.metrics_per_sec > 0 else 0.0
    )

    for step in range(config.steps):
        if config.fail_at is not None and step == config.fail_at:
            raise SimulatedFailure(f"simulated failure at step {step}")

        # Mid-close BEFORE this step's writes — close_at=N means step N
        # onward is post-close (writes dropped). Scenarios read this as
        # "close before step N executes" rather than "close after".
        if close_at is not None and step == close_at:
            run.__exit__(None, None, None)

        # Mid-reopen BEFORE writes for the same reason
        if reopen_at is not None and step == reopen_at:
            run.__exit__(None, None, None)
            run.__enter__()

        # Base metrics — use `.log(name, value, step=)` (no prefix) so
        # scenarios can assert on the exact metric name they set. The
        # `.log_train()` helper prepends ``train/`` which is right for
        # production use but wrong for our exact-name assertions.
        for i in range(config.metrics_per_step):
            name = _metric_name(step, i, config.new_metrics_at)
            run.log(name, float(step), step=step)

        # New-metric injection (schema-finalize trigger)
        if step in config.new_metrics_at:
            run.log(f"metric_new_step{step}", float(step), step=step)

        # Validation emit — scenarios assert on ``val/loss`` for this,
        # which is exactly what log_eval produces.
        if step in config.validation_at:
            run.log_eval(loss=float(step), step=step)

        # Restart aim server signal (out-of-scope for driver — handled by harness in ideal case)
        # For now, the scenario tolerates the restart happening asynchronously.

        if sleep_per_step > 0:
            time.sleep(sleep_per_step)


# ---------------------------------------------------------------------------
# Composer driver — real tiny CPU training
# ---------------------------------------------------------------------------


def _run_composer(config: DriverConfig) -> str | None:
    """Real Composer Trainer with a single nn.Linear(4, 1) on fake data.

    Emits explicit ``metric_0..metric_N`` per batch via a Callback that
    calls Composer's ``Logger.log_metrics`` — that's the entry point
    LoggerDestinations (including AstrolabeComposerLogger) receive.
    Also emits ``val/*`` metrics at validation-step boundaries when the
    config asks for them.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from composer import Trainer
    from composer.core import Callback
    from composer.models import ComposerModel
    from astrolabe_callbacks.composer import AstrolabeComposerLogger

    class TinyComposer(ComposerModel):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, batch):
            x, _ = batch
            return self.lin(x)

        def loss(self, outputs, batch):
            _, y = batch
            return ((outputs - y) ** 2).mean()

    class NamedMetricEmitter(Callback):
        """Emit ``metric_0..metric_N`` per batch via Composer's Logger."""

        def __init__(self, metrics_per_step: int, validation_at: list[int], new_metrics_at: list[int]):
            self._metrics_per_step = metrics_per_step
            self._validation_at = set(validation_at)
            self._new_metrics_at = set(new_metrics_at)
            self._step = 0

        def batch_end(self, state, logger):
            step = self._step
            metrics = {f"metric_{i}": float(step) for i in range(self._metrics_per_step)}
            if step in self._new_metrics_at:
                metrics[f"metric_new_step{step}"] = float(step)
            if step in self._validation_at:
                # Composer routes ``val/*``-prefixed metrics via the same log_metrics
                metrics["val/loss"] = float(step)
            logger.log_metrics(metrics)
            self._step += 1

    x = torch.randn(config.steps, 4)
    y = torch.randn(config.steps, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)

    logger = AstrolabeComposerLogger(
        aim_url=config.aim_url,
        experiment_name=config.experiment_name,
        tags=config.tags,
    )
    emitter = NamedMetricEmitter(
        metrics_per_step=config.metrics_per_step,
        validation_at=config.validation_at,
        new_metrics_at=config.new_metrics_at,
    )

    trainer = Trainer(
        model=TinyComposer(),
        train_dataloader=loader,
        max_duration=f"{config.steps}ba",
        loggers=[logger],
        callbacks=[emitter],
        device="cpu",
        progress_bar=False,
    )
    try:
        trainer.fit()
    except SimulatedFailure:
        raise
    # Composer nulls logger._run on close — hash comes from server query.
    return None


# ---------------------------------------------------------------------------
# Lightning driver
# ---------------------------------------------------------------------------


def _run_lightning(config: DriverConfig) -> str | None:
    """Real Lightning Trainer with a tiny LightningModule.

    Emits ``metric_0..metric_N`` per training batch via ``self.log(...)``
    (which routes to Lightning callbacks including AstrolabeLightningLogger).
    Wires a val_dataloader when config.validation_at is non-empty so
    ``validation_step`` fires and emits ``val/loss``.
    """
    import torch
    import torch.nn as nn
    import lightning
    from torch.utils.data import DataLoader, TensorDataset
    from astrolabe_callbacks.lightning import AstrolabeLightningLogger

    metrics_per_step = config.metrics_per_step
    new_metrics_at = set(config.new_metrics_at)

    class TinyLightning(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)
            self._step_counter = 0

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = ((self.lin(x) - y) ** 2).mean()
            step = self._step_counter
            # Emit N named metrics for the buffer-drain / batch-end tests.
            for i in range(metrics_per_step):
                self.log(f"metric_{i}", float(step), on_step=True, on_epoch=False, prog_bar=False)
            if step in new_metrics_at:
                self.log(f"metric_new_step{step}", float(step), on_step=True, on_epoch=False, prog_bar=False)
            self._step_counter += 1
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            loss = ((self.lin(x) - y) ** 2).mean()
            # Lightning's on_validation_end fires after the whole val loop;
            # emitting val/loss here (with on_epoch=True) makes it land
            # once per val loop pass.
            self.log("val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=False)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    x = torch.randn(config.steps, 4)
    y = torch.randn(config.steps, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)

    val_loader = None
    if config.validation_at:
        # Small val set — Lightning triggers val loops based on
        # check_val_every_n_epoch or val_check_interval, not custom
        # step lists. For simplicity: if any validation_at is set, do
        # one val pass at end of training. That covers scenarios that
        # just want "did val emit at least once."
        vx = torch.randn(4, 4)
        vy = torch.randn(4, 1)
        val_loader = DataLoader(TensorDataset(vx, vy), batch_size=1)

    logger = AstrolabeLightningLogger(
        aim_url=config.aim_url,
        experiment_name=config.experiment_name,
        tags=config.tags,
        run_name=config.run_name,
    )

    trainer = lightning.Trainer(
        callbacks=[logger],
        max_epochs=1,
        limit_train_batches=config.steps,
        enable_progress_bar=False,
        accelerator="cpu",
        logger=False,
        num_sanity_val_steps=0,  # skip Lightning's default pre-training val sanity check
    )
    if val_loader is not None:
        trainer.fit(TinyLightning(), train_dataloaders=loader, val_dataloaders=val_loader)
    else:
        trainer.fit(TinyLightning(), train_dataloaders=loader)
    return None


# ---------------------------------------------------------------------------
# HuggingFace driver
# ---------------------------------------------------------------------------


def _run_hf(config: DriverConfig) -> str | None:
    """Real HF Trainer with a tiny model and toy dataset."""
    import torch
    import torch.nn as nn
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset
    from astrolabe_callbacks.huggingface import AstrolabeHFTrainerCallback

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, input_ids=None, labels=None, **kwargs):
            out = self.lin(input_ids.float())
            loss = ((out - labels.float()) ** 2).mean() if labels is not None else None
            return {"loss": loss, "logits": out}

    class ToyDataset(Dataset):
        def __init__(self, n):
            self.n = n
            self.x = torch.randn(n, 4)
            self.y = torch.randn(n, 1)

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return {"input_ids": self.x[i], "labels": self.y[i]}

    cb = AstrolabeHFTrainerCallback(
        aim_url=config.aim_url,
        experiment_name=config.experiment_name,
        tags=config.tags,
        run_name=config.run_name,
    )
    args = TrainingArguments(
        output_dir="/tmp/hf-testbed",
        max_steps=config.steps,
        per_device_train_batch_size=1,
        logging_steps=1,
        disable_tqdm=True,
        report_to=[],
        use_cpu=True,
    )
    trainer = Trainer(model=ToyModel(), args=args, train_dataset=ToyDataset(config.steps), callbacks=[cb])
    trainer.train()
    # HF nulls cb._run on close — hash comes from stats jsonl.
    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _int_flag(config: DriverConfig, key: str) -> int | None:
    v = config.driver_flags.get(key)
    return int(v) if v is not None else None


def main() -> None:
    """Entry point for subprocess invocation."""
    config = DriverConfig.from_env()

    from pathlib import Path
    import os

    # Ensure the stats jsonl directory exists inside the container so
    # the callback library can append to it. Without this, callback
    # writes to a missing dir and silently drops stats events.
    stats_path = Path(config.stats_jsonl_container_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # First-metric marker parent dir. Callback library touches the file
    # itself on first ``track_safely``; we just need the dir to exist.
    # Also unlink any leftover from a prior run to make touched-vs-not
    # unambiguous.
    marker_env = os.environ.get("ASTROLABE_FIRST_METRIC_MARKER")
    if marker_env:
        Path(marker_env).parent.mkdir(parents=True, exist_ok=True)
        Path(marker_env).unlink(missing_ok=True)

    try:
        run_hash = run_driver(config)
    except SimulatedFailure:
        # Sentinel exit code so scenarios distinguish expected vs unexpected failure
        raise SystemExit(42)

    # The ctx-manager path prints the hash inside the with-block so
    # exception scenarios still get it. For the normal path, print
    # here.
    if run_hash and not config.driver_flags.get("TESTBED_USE_CONTEXT_MANAGER") == "1":
        print(f"ASTROLABE_RUN_HASH={run_hash}")

    # First-metric marker check: report whether the callback touched it.
    if marker_env:
        touched = Path(marker_env).exists()
        print(f"ASTROLABE_MARKER_TOUCHED={'true' if touched else 'false'}")


if __name__ == "__main__":
    main()
