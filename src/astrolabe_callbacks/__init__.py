"""astrolabe-callbacks — framework-agnostic Aim instrumentation for ML training.

Public API::

    from astrolabe_callbacks import AstrolabeComposerLogger    # MosaicML Composer
    from astrolabe_callbacks import AstrolabeLightningLogger   # PyTorch Lightning
    from astrolabe_callbacks import AstrolabeHFTrainerCallback # HuggingFace Trainer
    from astrolabe_callbacks import Run                        # raw PyTorch / JAX / custom loops
    from astrolabe_callbacks import log_eval_table             # post-training benchmark results

The per-framework training callbacks (and the raw-loop ``Run`` context
manager) stream ``train/`` and ``val/`` metrics as your model trains.
``log_eval_table`` / ``start_eval_run`` log post-training benchmark
suites (GLUE, MMLU, …) under the ``eval/<task_set>/<metric>`` namespace
on a separate Aim run — that's what populates astrolabe's dashboard
Eval tab.

Each per-framework class is imported lazily — `import astrolabe_callbacks`
only needs `aim` and `loguru`. Framework dependencies are pulled in on
first reference, surfacing a clear `ImportError` if the matching extras
aren't installed::

    pip install astrolabe-callbacks[composer]
    pip install astrolabe-callbacks[lightning]
    pip install astrolabe-callbacks[hf]
    pip install astrolabe-callbacks[all]

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
# astrolabe_callbacks are opting into our reliability posture, which
# depends on Aim exceptions propagating.
try:
    from aim.ext.exception_resistant import disable_safe_mode as _disable_aim_safe_mode

    _disable_aim_safe_mode()
except ImportError:
    # Older Aim versions or aim not installed — nothing to disable.
    pass

from astrolabe_callbacks.eval_results import (
    EvalInputError,
    MissingParentError,
    log_eval_table,
    start_eval_run,
    start_eval_run_from_checkpoint,
)

__version__ = "2.0.0"

__all__ = [
    "AstrolabeComposerLogger",
    "AstrolabeComposerCheckpointer",
    "AstrolabeLightningCheckpointer",
    "AstrolabeHFCheckpointer",
    "CheckpointMeta",
    "build_checkpoint_meta",
    "read_checkpoint_meta",
    "save_derived_checkpoint",
    "stamp_checkpoint",
    "export_checkpoint",
    "save_checkpoint",
    "AstrolabeLightningLogger",
    "AstrolabeHFTrainerCallback",
    "AstrolabeRun",
    "Run",
    "log_eval_table",
    "start_eval_run",
    "start_eval_run_from_checkpoint",
    "EvalInputError",
    "MissingParentError",
    "__version__",
]


# PEP 562 module-level __getattr__ defers framework imports until a class
# is actually referenced. Without this, `import astrolabe_callbacks` would
# pull in Composer/Lightning/Transformers eagerly and the base install
# (aim only) would fail.
def __getattr__(name: str):
    if name == "AstrolabeComposerLogger":
        from astrolabe_callbacks.composer import AstrolabeComposerLogger
        return AstrolabeComposerLogger
    if name == "AstrolabeComposerCheckpointer":
        from astrolabe_callbacks.composer import AstrolabeComposerCheckpointer
        return AstrolabeComposerCheckpointer
    if name == "AstrolabeLightningCheckpointer":
        from astrolabe_callbacks.lightning import AstrolabeLightningCheckpointer
        return AstrolabeLightningCheckpointer
    if name == "AstrolabeHFCheckpointer":
        from astrolabe_callbacks.huggingface import AstrolabeHFCheckpointer
        return AstrolabeHFCheckpointer
    if name in ("CheckpointMeta", "build_checkpoint_meta",
                "read_checkpoint_meta", "export_checkpoint",
                "save_derived_checkpoint", "stamp_checkpoint"):
        # checkpoint.py imports no framework — safe to load eagerly here.
        from astrolabe_callbacks import checkpoint
        return getattr(checkpoint, name)
    if name == "save_checkpoint":
        from astrolabe_callbacks.pytorch import save_checkpoint
        return save_checkpoint
    if name == "AstrolabeLightningLogger":
        from astrolabe_callbacks.lightning import AstrolabeLightningLogger
        return AstrolabeLightningLogger
    if name == "AstrolabeHFTrainerCallback":
        from astrolabe_callbacks.huggingface import AstrolabeHFTrainerCallback
        return AstrolabeHFTrainerCallback
    if name in ("AstrolabeRun", "Run"):
        from astrolabe_callbacks.pytorch import AstrolabeRun
        return AstrolabeRun
    raise AttributeError(f"module 'astrolabe_callbacks' has no attribute {name!r}")
