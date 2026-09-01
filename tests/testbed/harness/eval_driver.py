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

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


__all__ = [
    "EvalDriverConfig",
    "EvalDriverResult",
    "config_to_env",
    "run_eval_driver",
    "main",
]


@dataclass(frozen=True)
class EvalDriverConfig:
    aim_url: str
    task_set: str
    model_run_hash: str
    rows: dict[str, tuple[str, float]]
    streaming_metrics: list[tuple[str, list[tuple[int, float]]]]
    use_from_checkpoint: bool
    checkpoint_path: str | None
    on_missing_parent: str
    driver_flags: dict[str, str]
    # Env the astrolabe engine would have exported into an eval step
    # (ALIDADE_EXPERIMENT_NAME, AIM_RUN_TAGS). Passed through to the
    # container verbatim rather than serialized: the helpers read the
    # real env var names, so anything else would test a stand-in.
    submit_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "EvalDriverConfig":
        def _req(key: str) -> str:
            v = os.environ.get(key)
            if v is None:
                print(f"missing required env var {key}", file=sys.stderr)
                raise SystemExit(2)
            return v

        # rows come across as {"cola": ["matthews", 0.82]} — convert list → tuple
        raw_rows = json.loads(os.environ.get("TESTBED_EVAL_ROWS", "{}"))
        rows = {k: (v[0], float(v[1])) for k, v in raw_rows.items()}

        raw_streaming = json.loads(os.environ.get("TESTBED_EVAL_STREAMING", "[]"))
        streaming = [
            (name, [(int(step), float(val)) for step, val in points])
            for name, points in raw_streaming
        ]

        return cls(
            aim_url=_req("TESTBED_EVAL_AIM_URL"),
            task_set=_req("TESTBED_EVAL_TASK_SET"),
            model_run_hash=os.environ.get("TESTBED_EVAL_MODEL_RUN_HASH", ""),
            rows=rows,
            streaming_metrics=streaming,
            use_from_checkpoint=os.environ.get("TESTBED_EVAL_USE_FROM_CHECKPOINT", "0") == "1",
            checkpoint_path=(
                os.environ["TESTBED_EVAL_CHECKPOINT_PATH"]
                if os.environ.get("TESTBED_EVAL_HAS_CHECKPOINT_PATH") == "1"
                else None
            ),
            on_missing_parent=os.environ.get("TESTBED_EVAL_ON_MISSING_PARENT", "warn"),
            driver_flags=json.loads(os.environ.get("TESTBED_EVAL_DRIVER_FLAGS", "{}")),
            submit_env={},  # already in os.environ by the time this runs
        )


@dataclass(frozen=True)
class EvalDriverResult:
    exit_code: int
    eval_run_hash: str | None
    linked: bool
    stdout: str
    stderr: str


def config_to_env(config: EvalDriverConfig) -> dict[str, str]:
    """Serialize ``config`` to a dict suitable for ``compose.exec_in(env=...)``."""
    env = {
        "TESTBED_EVAL_AIM_URL": config.aim_url,
        "TESTBED_EVAL_TASK_SET": config.task_set,
        "TESTBED_EVAL_MODEL_RUN_HASH": config.model_run_hash,
        "TESTBED_EVAL_ROWS": json.dumps({k: [v[0], v[1]] for k, v in config.rows.items()}),
        "TESTBED_EVAL_STREAMING": json.dumps(config.streaming_metrics),
        "TESTBED_EVAL_USE_FROM_CHECKPOINT": "1" if config.use_from_checkpoint else "0",
        "TESTBED_EVAL_ON_MISSING_PARENT": config.on_missing_parent,
        "TESTBED_EVAL_DRIVER_FLAGS": json.dumps(config.driver_flags),
        "ALIDADE_AIM_URL": config.aim_url,
    }
    if config.checkpoint_path is not None:
        env["TESTBED_EVAL_CHECKPOINT_PATH"] = config.checkpoint_path
        env["TESTBED_EVAL_HAS_CHECKPOINT_PATH"] = "1"
    env.update(config.submit_env)
    return env


def _prepare_checkpoint(config: EvalDriverConfig) -> None:
    """Honor driver_flags that ask us to create a checkpoint before eval runs.

    Written in the container so the file the helper reads was produced
    by the same library version that reads it — the point of the
    exercise is the real embed-then-read round trip, not a fixture.
    """
    flags = config.driver_flags
    if config.checkpoint_path is None:
        return
    path = Path(config.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from alidade_callbacks.checkpoint import CheckpointMeta, export_checkpoint

    stamped_at = "2026-08-06T00:00:00Z"
    pt_hash = flags.get("TESTBED_CREATE_PT_CHECKPOINT_WITH_HASH")
    if pt_hash:
        export_checkpoint(
            {}, path, fmt="pt",
            meta=CheckpointMeta(aim_run_hash=pt_hash, created_at=stamped_at),
        )
    st_hash = flags.get("TESTBED_CREATE_SAFETENSORS_CHECKPOINT_WITH_HASH")
    if st_hash:
        export_checkpoint(
            {}, path, fmt="safetensors",
            meta=CheckpointMeta(aim_run_hash=st_hash, created_at=stamped_at),
        )
    if flags.get("TESTBED_CREATE_PT_CHECKPOINT_WITHOUT_META"):
        # Raw torch.save, not export_checkpoint: this has to be a valid
        # checkpoint carrying no astrolabe block at all, which is what a
        # file predating the feature looks like.
        import torch

        torch.save({}, path)


def run_eval_driver(config: EvalDriverConfig) -> tuple[str | None, bool]:
    """Execute one eval-driver invocation. Returns (eval_run_hash, linked)."""
    from alidade_callbacks.eval_results import log_eval_table, start_eval_run

    _prepare_checkpoint(config)

    if config.use_from_checkpoint:
        # eval-linkage M0/M1 wires this path; skipped via marker at scenario level
        # until it exists in src/.
        try:
            from alidade_callbacks.eval_results import start_eval_run_from_checkpoint
        except ImportError:
            print("start_eval_run_from_checkpoint not available (eval-linkage M0 not yet landed)", file=sys.stderr)
            raise SystemExit(1)

        run = start_eval_run_from_checkpoint(
            checkpoint=config.checkpoint_path,
            task_set=config.task_set,
            aim_url=config.aim_url,
            model_run_hash=(config.model_run_hash or None),
            on_missing_parent=config.on_missing_parent,
        )
        # Stream any provided points
        for name, points in config.streaming_metrics:
            for step, value in points:
                run.track(value, name=name, step=step)
        linked = getattr(run, "astrolabe_linked", False)
        run_hash = run.hash
        run.close()
        return run_hash, linked

    if config.streaming_metrics:
        run = start_eval_run(
            model_run_hash=config.model_run_hash,
            task_set=config.task_set,
            aim_url=config.aim_url,
        )
        for name, points in config.streaming_metrics:
            for step, value in points:
                run.track(value, name=name, step=step)
        if config.driver_flags.get("TESTBED_EVAL_VERIFY_NO_AUTOCLOSE") == "1":
            # The scenario just cares that the helper didn't force-close.
            # We close explicitly below.
            pass
        run_hash = run.hash
        run.close()
        return run_hash, True

    # log_eval_table path
    if not config.rows:
        raise ValueError("log_eval_table requires non-empty rows")

    rows = config.rows
    if config.driver_flags.get("TESTBED_EVAL_INJECT_NON_NUMERIC") == "1":
        # Force a non-numeric to test rejection
        rows = {k: (v[0], "not-a-number") for k, v in rows.items()}  # type: ignore[misc]

    result_hash = log_eval_table(
        model_run_hash=config.model_run_hash,
        task_set=config.task_set,
        rows=rows,
        aim_url=config.aim_url,
    )
    # log_eval_table may or may not return the hash depending on version;
    # if not, we can't recover it from here — scenarios handle both.
    return (result_hash if isinstance(result_hash, str) else None), True


def main() -> None:
    """Entry point for subprocess invocation."""
    config = EvalDriverConfig.from_env()
    try:
        eval_hash, linked = run_eval_driver(config)
    except Exception as e:
        # MissingParentError → exit code 43
        if type(e).__name__ == "MissingParentError":
            print(f"ALIDADE_EVAL_LINKED=false")
            raise SystemExit(43)
        raise

    if eval_hash:
        print(f"ALIDADE_EVAL_RUN_HASH={eval_hash}")
    print(f"ALIDADE_EVAL_LINKED={'true' if linked else 'false'}")


if __name__ == "__main__":
    main()
