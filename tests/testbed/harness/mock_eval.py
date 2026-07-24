"""Env-var-driven mock eval driver for testbed scenarios.

Exercises the module-level eval helpers (``start_eval_run``,
``log_eval_table``, and — once eval-linkage Milestone 0 lands —
``start_eval_run_from_checkpoint``) against a real aim server. No
framework Trainer, no checkpoints (until M1 wires those in) — just the
minimum shape needed to verify the eval-helper contract.

**Runs inside the ``client`` container.** Scenarios invoke it via
``compose.exec_in(testbed, service="client", cmd=["python", "-m",
"tests.testbed.harness.mock_eval"], env={...})``.

Companion to ``mock_training.py``. Scenarios that verify eval-run
linkage typically run mock_training first, capture the parent run hash,
then run mock_eval pointing at that parent.
"""
from __future__ import annotations

from dataclasses import dataclass


__all__ = ["MockEvalConfig", "run_mock_eval", "main"]


@dataclass(frozen=True)
class MockEvalConfig:
    """Configuration for a mock eval run.

    Attributes
    ----------
    aim_url : str
        Aim server URL. Set by the harness.
    task_set : str
        Eval task set name (e.g. ``"glue"``).
    model_run_hash : str
        Parent training run hash to link this eval to.
    rows : dict[str, tuple[str, float]]
        Task → (metric, score) pairs. Drives ``log_eval_table`` calls
        one-shot. If empty, no table is logged.
    streaming_metrics : list[tuple[str, list[tuple[int, float]]]]
        List of (metric_name, [(step, value), ...]) pairs. Drives
        ``start_eval_run`` for mid-training-style trajectories. Empty
        list → no streaming metrics.
    use_from_checkpoint : bool
        If True, use ``start_eval_run_from_checkpoint`` (requires
        ``checkpoint_path``); if False, use ``start_eval_run``. Ignored
        until eval-linkage Milestone 0 lands.
    checkpoint_path : str | None
        Path to a checkpoint file with embedded meta. Only meaningful
        when ``use_from_checkpoint=True``. None until M1 lands.
    on_missing_parent : str
        Passed to ``start_eval_run_from_checkpoint``: ``"warn"`` or
        ``"raise"``.
    """

    aim_url: str
    task_set: str
    model_run_hash: str
    rows: dict[str, tuple[str, float]]
    streaming_metrics: list[tuple[str, list[tuple[int, float]]]]
    use_from_checkpoint: bool
    checkpoint_path: str | None
    on_missing_parent: str

    @classmethod
    def from_env(cls) -> "MockEvalConfig":
        """Build from ``MOCK_EVAL_*`` env vars. Missing required → SystemExit(2)."""
        raise NotImplementedError


def run_mock_eval(config: MockEvalConfig) -> str:
    """Execute one mock eval run and return the eval run's aim hash.

    Chooses ``log_eval_table`` (if only ``rows`` is set),
    ``start_eval_run`` (if streaming), or
    ``start_eval_run_from_checkpoint`` (if ``use_from_checkpoint``).
    Closes the run cleanly on success.
    """
    raise NotImplementedError


def main() -> None:
    """Entry point for subprocess invocation."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
