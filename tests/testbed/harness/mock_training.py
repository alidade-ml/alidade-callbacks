"""Env-var-driven mock training script for testbed scenarios.

Reads config from environment, drives ``AstrolabeLogger`` (or a
framework-specific adapter) through a training-shaped lifecycle, exits.
No torch, no CUDA, no framework — this is the callback library's minimum
viable driver so scenarios can exercise the callback without booting a
real framework Trainer.

**Runs inside the ``client`` container.** Scenarios invoke it via
``compose.exec_in(testbed, service="client", cmd=["python", "-m",
"tests.testbed.harness.mock_training"], env={...})``. The docker-compose
bind-mount makes this file visible inside the container at the same
path.

Contract mirrors the astrolabe testbed's ``fixtures/mock_training/`` so
scenarios can transplant.
"""
from __future__ import annotations

from dataclasses import dataclass


__all__ = ["MockTrainingConfig", "run_mock_training", "main"]


@dataclass(frozen=True)
class MockTrainingConfig:
    """Configuration for a mock training run.

    All fields have env var equivalents (``MOCK_<FIELD>`` in uppercase).
    Env vars win over programmatic values so subprocess invocations
    remain configurable from the harness.

    Attributes
    ----------
    steps : int
        Number of training steps to emit.
    metrics_per_step : int
        Distinct metric names logged at each step. Higher values stress
        schema-finalize and dashboard rendering.
    metrics_per_sec : float
        Target write rate. Sleeps between steps to hit this rate. 0.0
        means "as fast as possible" (test_sustained.py scale scenarios).
    fail_at : int | None
        Step index at which to raise ``SimulatedTrainingError``. None →
        clean completion.
    add_metric_at : int | None
        Step index at which to introduce a new metric name (triggers
        schema-finalize). None → no new metrics after step 0.
    aim_url : str
        Aim server URL to connect to. Set by the harness from the
        ``TestbedHandle``.
    run_name : str
        Aim run name. Used by name-preservation assertions.
    experiment_name : str
        Aim experiment name (i.e. ``run.experiment``).
    tags : dict[str, str]
        Extra tags to set on the run before training begins. Used by
        tag-fidelity assertions.
    """

    steps: int
    metrics_per_step: int
    metrics_per_sec: float
    fail_at: int | None
    add_metric_at: int | None
    aim_url: str
    run_name: str
    experiment_name: str
    tags: dict[str, str]

    @classmethod
    def from_env(cls) -> "MockTrainingConfig":
        """Build from ``MOCK_*`` env vars. Missing required vars → SystemExit(2)."""
        raise NotImplementedError


def run_mock_training(config: MockTrainingConfig) -> None:
    """Execute one mock training run against ``config.aim_url``.

    Opens an ``AstrolabeLogger``, drives ``config.steps`` iterations
    emitting ``config.metrics_per_step`` metric names each, respects
    ``fail_at`` / ``add_metric_at`` triggers, closes cleanly on success.

    Raises ``SimulatedTrainingError`` if ``fail_at`` fires. Scenarios
    catch this to verify partial-progress behavior (buffer drain, close
    on error, stats jsonl completeness).
    """
    raise NotImplementedError


def main() -> None:
    """Entry point for subprocess invocation. Reads env, calls run_mock_training."""
    raise NotImplementedError


class SimulatedTrainingError(RuntimeError):
    """Raised at ``fail_at`` step to simulate a training-level failure."""


if __name__ == "__main__":
    main()
