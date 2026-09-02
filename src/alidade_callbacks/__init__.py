"""alidade-callbacks — framework-agnostic Aim instrumentation for ML training.

Public API::

    from alidade_callbacks import AlidadeComposerLogger    # MosaicML Composer
    from alidade_callbacks import AlidadeLightningLogger   # PyTorch Lightning
    from alidade_callbacks import AlidadeHFTrainerCallback # HuggingFace Trainer
    from alidade_callbacks import Run                        # raw PyTorch / JAX / custom loops
    from alidade_callbacks import log_eval_table             # post-training benchmark results

The per-framework training callbacks (and the raw-loop ``Run`` context
manager) stream ``train/`` and ``val/`` metrics as your model trains.
``log_eval_table`` / ``start_eval_run`` log post-training benchmark
suites (GLUE, MMLU, …) under the ``eval/<task_set>/<metric>`` namespace
on a separate Aim run — that's what populates alidade's dashboard
Eval tab.

Each per-framework class is imported lazily — `import alidade_callbacks`
only needs `aim` and `loguru`. Framework dependencies are pulled in on
first reference, surfacing a clear `ImportError` if the matching extras
aren't installed::

    pip install alidade-callbacks[composer]
    pip install alidade-callbacks[lightning]
    pip install alidade-callbacks[hf]
    pip install alidade-callbacks[all]

The eval helpers need only the base install (`aim`) — no framework extra.
"""

from __future__ import annotations

# Disable Aim's default "safe mode" — aim.Run.track is decorated with
# @noexcept which silently swallows exceptions. Our _MetricBuffer's
# retry loop in _core._drain_loop_inner catches exceptions to trigger
# transient-error retries + record ``_retried`` / ``_dropped_failed``
# counters. Without this call, aim silently absorbs those exceptions
# and our retry logic is unreachable. Testbed caught the gap
# (tests/testbed/RED_FLAGS.md — 2026-07-27).
#
# Import-time side effect is intentional: users importing
# alidade_callbacks are opting into our reliability posture, which
# depends on Aim exceptions propagating.
try:
    from aim.ext.exception_resistant import disable_safe_mode as _disable_aim_safe_mode

    _disable_aim_safe_mode()
except ImportError:
    # Older Aim versions or aim not installed — nothing to disable.
    pass

from alidade_callbacks.samples import Sample, SampleInputError, log_samples
from alidade_callbacks.eval_results import (
    EvalInputError,
    MissingParentError,
    log_eval_table,
    start_eval_run,
    start_eval_run_from_checkpoint,
)

__version__ = "2.0.0"

__all__ = [
    "AlidadeComposerLogger",
    "AlidadeComposerCheckpointer",
    "AlidadeLightningCheckpointer",
    "AlidadeHFCheckpointer",
    "CheckpointMeta",
    "build_checkpoint_meta",
    "read_checkpoint_meta",
    "save_derived_checkpoint",
    "stamp_checkpoint",
    "export_checkpoint",
    "save_checkpoint",
    "AlidadeLightningLogger",
    "AlidadeHFTrainerCallback",
    "AlidadeRun",
    "Run",
    "log_eval_table",
    "log_samples",
    "Sample",
    "SampleInputError",
    "start_eval_run",
    "start_eval_run_from_checkpoint",
    "EvalInputError",
    "MissingParentError",
    "__version__",
]


# PEP 562 module-level __getattr__ defers framework imports until a class
# is actually referenced. Without this, `import alidade_callbacks` would
# pull in Composer/Lightning/Transformers eagerly and the base install
# (aim only) would fail.
def __getattr__(name: str):
    if name == "AlidadeComposerLogger":
        from alidade_callbacks.composer import AlidadeComposerLogger
        return AlidadeComposerLogger
    if name == "AlidadeComposerCheckpointer":
        from alidade_callbacks.composer import AlidadeComposerCheckpointer
        return AlidadeComposerCheckpointer
    if name == "AlidadeLightningCheckpointer":
        from alidade_callbacks.lightning import AlidadeLightningCheckpointer
        return AlidadeLightningCheckpointer
    if name == "AlidadeHFCheckpointer":
        from alidade_callbacks.huggingface import AlidadeHFCheckpointer
        return AlidadeHFCheckpointer
    if name in ("CheckpointMeta", "build_checkpoint_meta",
                "read_checkpoint_meta", "export_checkpoint",
                "save_derived_checkpoint", "stamp_checkpoint"):
        # checkpoint.py imports no framework — safe to load eagerly here.
        from alidade_callbacks import checkpoint
        return getattr(checkpoint, name)
    if name == "save_checkpoint":
        from alidade_callbacks.pytorch import save_checkpoint
        return save_checkpoint
    if name == "AlidadeLightningLogger":
        from alidade_callbacks.lightning import AlidadeLightningLogger
        return AlidadeLightningLogger
    if name == "AlidadeHFTrainerCallback":
        from alidade_callbacks.huggingface import AlidadeHFTrainerCallback
        return AlidadeHFTrainerCallback
    if name in ("AlidadeRun", "Run"):
        from alidade_callbacks.pytorch import AlidadeRun
        return AlidadeRun
    raise AttributeError(f"module 'alidade_callbacks' has no attribute {name!r}")
