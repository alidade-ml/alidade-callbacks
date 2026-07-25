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

from dataclasses import dataclass
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
    """Configuration for a single driver invocation.

    All fields have env var equivalents (``TESTBED_<FIELD>`` in uppercase).
    Env vars win over programmatic values so subprocess invocations
    remain configurable from the pytest harness. ``config_to_env(config)``
    serializes to the env-dict form used by ``compose.exec_in``.

    Attributes
    ----------
    framework : {"raw", "composer", "lightning", "hf"}
        Which driver mode. See module docstring for what each mode does.
        ``raw`` is a direct AstrolabeRun exerciser; ``composer`` /
        ``lightning`` / ``hf`` run real (tiny, CPU-only, seeded)
        training so the framework's own event loop drives the callback.
    steps : int
        Number of training steps (for framework modes: number of batches;
        for raw mode: number of ``track()`` calls per metric).
    metrics_per_step : int
        Distinct metric names logged at each step (excluding
        new-metric events driven by ``new_metrics_at``).
    metrics_per_sec : float
        Target write rate. Driver sleeps between steps to hit this
        rate. 0.0 = as fast as possible (sustained-writes scenarios).
    fail_at : int | None
        Step at which to raise ``SimulatedFailure``. None = clean
        completion.
    new_metrics_at : list[int]
        Steps at which to introduce a distinct new metric name (each
        triggers schema-finalize). Empty list = no new metrics after
        the initial ones seeded at step 0.
    validation_at : list[int]
        Steps at which to emit a validation metric. For framework
        modes, driver invokes the framework's eval path at these steps.
        Empty list = no validation.
    close : bool
        Whether the driver calls ``close()`` (or its framework
        equivalent) at the end. False lets scenarios verify
        context-manager / lifecycle-exit paths.
    aim_url : str
        Aim server URL. Set by the harness from ``TestbedHandle``.
    run_name : str
        Aim run name. Used by name-preservation assertions.
    experiment_name : str
        Aim experiment name.
    tags : dict[str, str]
        **Real Aim tags** — set on the run, land in the Aim record,
        verified by tag-fidelity assertions.
    driver_flags : dict[str, str]
        **Testbed-specific driver toggles** (``TESTBED_KILL_DRAINER_AT``,
        ``TESTBED_BUFFER_CAPACITY``, ``TESTBED_INJECT_TRANSIENT_ERROR_AT``,
        etc.). Never touch Aim — they only control the driver's
        internal branching so scenarios can exercise error paths and
        edge cases.
    stats_jsonl_container_path : str
        Path inside the client container where the callback writes the
        stats jsonl. Bind-mounted to a host tmp path so scenarios read
        it after the invocation completes.
    """

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
        raise NotImplementedError


@dataclass(frozen=True)
class DriverResult:
    """Result of a single driver invocation.

    Attributes
    ----------
    exit_code : int
        Process exit code. 0 = clean completion; 42 = SimulatedFailure
        (fail_at fired as expected); other = real failure.
    run_hash : str | None
        The Aim run hash printed by the driver on completion (24 chars).
        None if the driver failed before opening the Aim run, or the
        rank-gating path skipped opening a run.
    stats_events : list[dict]
        Parsed stats jsonl lines emitted during the invocation. Each
        dict is one event (``{"event": "first_metric", "run_hash": ...}``).
    stdout : str
        Full stdout. Preserved so assertion failures include diagnostic.
    stderr : str
        Full stderr. Same rationale.
    """

    exit_code: int
    run_hash: str | None
    stats_events: list[dict]
    stdout: str
    stderr: str


def config_to_env(config: DriverConfig) -> dict[str, str]:
    """Serialize ``config`` to a dict suitable for ``compose.exec_in(env=...)``.

    Lists and dicts are JSON-encoded. Optional None becomes empty string
    with an explicit ``TESTBED_HAS_<FIELD>`` marker so the receiver can
    distinguish "unset" from "empty."
    """
    raise NotImplementedError


def run_driver(config: DriverConfig) -> None:
    """Execute one driver invocation against ``config.aim_url``.

    Dispatches by ``config.framework``:

    - ``raw``: opens ``AstrolabeRun`` directly, drives track()/close()
      in a loop. No torch, no framework.
    - ``composer`` / ``lightning`` / ``hf``: constructs a tiny CPU
      model (``nn.Linear(4, 1)``), fixed deterministic data, invokes
      the framework's real ``Trainer.fit()`` (or equivalent) with
      the corresponding AstrolabeCallback wired in.

    ``fail_at`` raises ``SimulatedFailure`` (exit code 42) — for framework
    modes this happens inside the training loop and the framework's
    error path exercises the callback's abort semantics for real.

    ``new_metrics_at`` introduces new metric names. For raw mode, this
    is a direct extra ``track()`` call with a fresh name. For framework
    modes, the driver conditionally emits a new metric via the
    framework's own logging API (Composer: ``logger.log_metrics``;
    Lightning: ``self.log()``; HF: return dict from compute_metrics).

    Fault-injection driver_flags apply regardless of mode.

    Prints ``ASTROLABE_RUN_HASH=<24-char>`` on the final stdout line
    so the ``run_driver`` fixture can extract it.
    """
    raise NotImplementedError


def main() -> None:
    """Entry point for subprocess invocation. Reads env, calls run_driver.

    Seeds torch/numpy at import time (``TESTBED_SEED``, default 0) so
    framework paths produce deterministic loss values.
    """
    raise NotImplementedError


class SimulatedFailure(RuntimeError):
    """Raised at ``fail_at`` step to simulate a failure.

    Distinct exit code (42) so scenarios can distinguish expected
    simulated failure from real driver bugs. For framework modes, the
    driver either raises this from inside the training step (Composer:
    inside ``loss()``; Lightning: inside ``training_step``; HF:
    inside the compute-loss path) or wraps the ``Trainer.fit()`` call
    to catch and re-emit with the sentinel exit code.
    """


if __name__ == "__main__":
    main()
