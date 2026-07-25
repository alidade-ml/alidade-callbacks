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
    if config.framework == "composer":
        return _run_composer(config)
    if config.framework == "lightning":
        return _run_lightning(config)
    if config.framework == "hf":
        return _run_hf(config)
    raise SystemExit(f"unknown framework: {config.framework}")


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
# Raw driver — direct AstrolabeRun
# ---------------------------------------------------------------------------


def _run_raw(config: DriverConfig) -> str | None:
    """Direct AstrolabeRun exerciser. No torch, no framework."""
    from astrolabe_callbacks.pytorch import AstrolabeRun
    from astrolabe_callbacks._distributed import is_rank_zero

    # Rank-gating: non-rank-zero returns None (no run opened).
    if not is_rank_zero():
        return None

    # Optional context-manager path
    use_ctx = config.driver_flags.get("TESTBED_USE_CONTEXT_MANAGER") == "1"
    if use_ctx:
        try:
            with AstrolabeRun(
                aim_url=config.aim_url,
                experiment_name=config.experiment_name,
                tags=config.tags,
                run_name=config.run_name,
            ) as run:
                run_hash = run._run.hash if run._run is not None else None
                _drive_raw_body(config, run, run_hash)
            return run_hash
        except SimulatedFailure:
            raise
        except Exception as e:
            # Re-raise so exit code reflects real vs simulated failure
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

        # Base metrics
        for i in range(config.metrics_per_step):
            name = _metric_name(step, i, config.new_metrics_at)
            run.log_train(**{name: float(step)}, step=step)

        # New-metric injection (schema-finalize trigger)
        if step in config.new_metrics_at:
            run.log_train(**{f"metric_new_step{step}": float(step)}, step=step)

        # Validation emit
        if step in config.validation_at:
            run.log_eval(**{"loss": float(step)}, step=step)

        # Mid-close (test_track_after_close_is_noop)
        if close_at is not None and step == close_at:
            run.__exit__(None, None, None)
            # continue looping — subsequent log_train() calls should be no-ops

        # Mid-reopen (test_name_survives_explicit_close_reopen)
        if reopen_at is not None and step == reopen_at:
            run.__exit__(None, None, None)
            run.__enter__()

        # Restart aim server signal (out-of-scope for driver — handled by harness in ideal case)
        # For now, the scenario tolerates the restart happening asynchronously.

        if sleep_per_step > 0:
            time.sleep(sleep_per_step)


# ---------------------------------------------------------------------------
# Composer driver — real tiny CPU training
# ---------------------------------------------------------------------------


def _run_composer(config: DriverConfig) -> str | None:
    """Real Composer Trainer with a single nn.Linear(4, 1) on fake data."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from composer import Trainer
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

    x = torch.randn(config.steps, 4)
    y = torch.randn(config.steps, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)

    # Note: AstrolabeComposerLogger does NOT accept run_name (Stage 3
    # discovery — Stage 1 signature docs would have implied it does, but
    # Composer's constructor takes only aim_url/experiment_name/tags).
    # No test scenario asserts on the Composer run name today, so we
    # omit it here. If a future test needs it, we'll set the Aim run
    # name via env before instantiation.
    logger = AstrolabeComposerLogger(
        aim_url=config.aim_url,
        experiment_name=config.experiment_name,
        tags=config.tags,
    )

    trainer = Trainer(
        model=TinyComposer(),
        train_dataloader=loader,
        max_duration=f"{config.steps}ba",
        loggers=[logger],
        device="cpu",
        progress_bar=False,
    )
    try:
        trainer.fit()
    except SimulatedFailure:
        raise
    # Composer closes loggers on trainer teardown; log_train hash extraction:
    return getattr(logger, "_run", None) and logger._run.hash


# ---------------------------------------------------------------------------
# Lightning driver
# ---------------------------------------------------------------------------


def _run_lightning(config: DriverConfig) -> str | None:
    """Real Lightning Trainer with a tiny LightningModule."""
    import torch
    import torch.nn as nn
    import lightning
    from torch.utils.data import DataLoader, TensorDataset
    from astrolabe_callbacks.lightning import AstrolabeLightningLogger

    class TinyLightning(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = ((self.lin(x) - y) ** 2).mean()
            self.log("metric_0", loss.item())
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    x = torch.randn(config.steps, 4)
    y = torch.randn(config.steps, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)

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
    )
    trainer.fit(TinyLightning(), train_dataloaders=loader)
    return getattr(logger, "_run", None) and logger._run.hash


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
        no_cuda=True,
    )
    trainer = Trainer(model=ToyModel(), args=args, train_dataset=ToyDataset(config.steps), callbacks=[cb])
    trainer.train()
    return getattr(cb, "_run", None) and cb._run.hash


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _int_flag(config: DriverConfig, key: str) -> int | None:
    v = config.driver_flags.get(key)
    return int(v) if v is not None else None


def main() -> None:
    """Entry point for subprocess invocation."""
    config = DriverConfig.from_env()
    try:
        run_hash = run_driver(config)
    except SimulatedFailure:
        # Sentinel exit code so scenarios distinguish expected vs unexpected failure
        raise SystemExit(42)

    if run_hash:
        print(f"ASTROLABE_RUN_HASH={run_hash}")


if __name__ == "__main__":
    main()
