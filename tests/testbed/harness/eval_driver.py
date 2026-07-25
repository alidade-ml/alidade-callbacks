"""Eval-helper exerciser — runs inside the ``client`` container.

Drives the real module-level eval helpers (``log_eval_table``,
``start_eval_run``, and — once eval-linkage Milestone 0 lands —
``start_eval_run_from_checkpoint``) against a real Aim server. These
functions are already the researcher-facing surface; there is no
framework to mock and no training to run — this driver just invokes
them with parametrized inputs.

**Runs inside the ``client`` container.** Scenarios invoke it via
``compose.exec_in(testbed, service="client", cmd=["python", "-m",
"tests.testbed.harness.eval_driver"], env={...})``.

Companion to ``driver.py``. Scenarios that verify eval-run linkage
typically run a training driver first, capture the parent run hash,
then run this eval driver pointing at that parent.
"""
from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    "EvalDriverConfig",
    "EvalDriverResult",
    "config_to_env",
    "run_eval_driver",
    "main",
]


@dataclass(frozen=True)
class EvalDriverConfig:
    """Configuration for a single eval-driver invocation.

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
        ``start_eval_run`` for mid-training-style trajectories.
    use_from_checkpoint : bool
        If True, use ``start_eval_run_from_checkpoint`` (requires
        ``checkpoint_path``); if False, use ``start_eval_run`` /
        ``log_eval_table``. Ignored until eval-linkage Milestone 0 lands.
    checkpoint_path : str | None
        Path (inside the client container) to a checkpoint file with
        embedded meta. Only meaningful when ``use_from_checkpoint=True``.
    on_missing_parent : str
        Passed to ``start_eval_run_from_checkpoint``: ``"warn"`` or
        ``"raise"``.
    driver_flags : dict[str, str]
        Testbed-specific driver toggles (``TESTBED_EVAL_INJECT_NON_NUMERIC``,
        ``TESTBED_CREATE_PT_CHECKPOINT_WITH_HASH``, etc.). Same pattern
        as ``DriverConfig.driver_flags`` — never lands as Aim tags.
    """

    aim_url: str
    task_set: str
    model_run_hash: str
    rows: dict[str, tuple[str, float]]
    streaming_metrics: list[tuple[str, list[tuple[int, float]]]]
    use_from_checkpoint: bool
    checkpoint_path: str | None
    on_missing_parent: str
    driver_flags: dict[str, str]

    @classmethod
    def from_env(cls) -> "EvalDriverConfig":
        """Build from ``TESTBED_EVAL_*`` env vars. Missing required → SystemExit(2)."""
        raise NotImplementedError


@dataclass(frozen=True)
class EvalDriverResult:
    """Result of a single eval-driver invocation.

    Attributes
    ----------
    exit_code : int
        Process exit code. 0 = success, 43 = MissingParentError (raised
        by on_missing_parent="raise" when no parent could be resolved).
    eval_run_hash : str | None
        Hash of the eval Aim run. None if the invocation failed before
        opening a run.
    linked : bool
        Whether ``astrolabe.model_run_hash`` was set on the eval run.
        True for both explicit-hash and embedded-meta paths; False for
        unlinked-with-warning.
    stdout : str
    stderr : str
    """

    exit_code: int
    eval_run_hash: str | None
    linked: bool
    stdout: str
    stderr: str


def config_to_env(config: EvalDriverConfig) -> dict[str, str]:
    """Serialize ``config`` to a dict suitable for ``compose.exec_in(env=...)``.

    Mirrors ``driver.config_to_env`` — same JSON-encoding rules for
    lists/dicts, same ``TESTBED_HAS_<FIELD>`` markers for optional fields.
    """
    raise NotImplementedError


def run_eval_driver(config: EvalDriverConfig) -> None:
    """Execute one eval-driver invocation.

    Chooses ``log_eval_table`` (if only ``rows`` is set),
    ``start_eval_run`` (if streaming), or
    ``start_eval_run_from_checkpoint`` (if ``use_from_checkpoint``).
    Closes the run cleanly on success.

    Prints the eval run's hash on stdout as
    ``ASTROLABE_EVAL_RUN_HASH=<24-char-hash>`` and the linkage state as
    ``ASTROLABE_EVAL_LINKED=<true|false>`` so the ``run_eval_driver``
    fixture can extract both.
    """
    raise NotImplementedError


def main() -> None:
    """Entry point for subprocess invocation."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
