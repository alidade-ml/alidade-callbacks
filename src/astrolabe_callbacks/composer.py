"""MosaicML Composer logger wired to astrolabe-callbacks ``_core``.

Designed to pair with astrolabe's experiment orchestration: astrolabe
sets ``ASTROLABE_EXPERIMENT_NAME`` and ``AIM_RUN_TAGS`` in the
per-experiment env, and this logger reads them on init to tag the Aim
run with version, submit-id, and any other astrolabe conventions.

This is a Composer ``LoggerDestination`` (not a plain ``Callback``) so
**every** metric the user logs via ``logger.log_metrics(...)`` flows
through to Aim — not just ``train/loss``. The user picks what to track
in their training YAML; we pass it through. The only metric we synthesize
is ``wall_time``.

Usage
-----

::

    from astrolabe_callbacks import AstrolabeComposerLogger
    from composer import Trainer

    trainer = Trainer(
        ...,
        loggers=[AstrolabeComposerLogger()],   # NOTE: loggers=, not callbacks=
    )

The ``loggers=`` slot is required because Composer's ``Logger`` only
broadcasts user metrics to ``LoggerDestination`` subclasses registered
there. Attaching via ``callbacks=`` will still get lifecycle hooks
(``init`` / ``batch_end`` / ``fit_end``) but will silently drop every
``log_metrics`` call — losing all the user-named metrics. Composer
typically rejects this attachment with a clear error since v0.20+.

Standalone usage (no astrolabe)::

    AstrolabeComposerLogger(
        aim_url="aim://localhost:43800",
        experiment_name="my-exp",
        tags={"thesis": "scale-laws"},
    )

Metric names
------------

User-logged metrics flow through unchanged. The only renames we apply
are to Composer's automatic emissions, for cleaner display:

- ``loss/train/total`` → ``train/loss`` (Composer's automatic train loss)
- ``metrics/train/<name>`` → ``train/<name>`` (Composer's per-batch
  training metrics, e.g. accuracy)
- ``metrics/eval/<name>`` → ``<EVAL_PREFIX>/<name>`` (Composer's eval
  metrics; ``EVAL_PREFIX`` is ``"val"`` as of v1.0.0)
- Anything else: pass through as-is. Custom user metrics
  (``my_thing/foo``, ``throughput``) land under their original names.

Failure handling
----------------

This logger never crashes training. If Aim is unreachable or
misconfigured, a single ``WARNING`` is logged at init and every
subsequent operation no-ops. Set ``ASTROLABE_CALLBACK_STRICT=1`` to
flip warnings into raised exceptions for fail-fast CI behavior. See
``_core.py`` for the full failure-handling contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from astrolabe_callbacks import _core
from astrolabe_callbacks._distributed import is_rank_zero
from astrolabe_callbacks.checkpoint import (
    CheckpointMeta,
    _meta_from_block,
    build_checkpoint_meta,
    export_checkpoint,
    write_first_checkpoint_marker_once,
)

try:
    # ``LoggerDestination`` itself inherits from ``Callback``, so we get
    # both the lifecycle hooks (``init``, ``batch_end``, ``fit_end``)
    # and the metric-broadcast hook (``log_metrics``) on a single base
    # class. The user attaches via ``loggers=`` rather than
    # ``callbacks=``; see the module docstring.
    from composer.loggers import LoggerDestination

    # The checkpointer is a plain Callback: it attaches via
    # ``callbacks=`` and has no metric-broadcast role.
    from composer.core import Callback
except ImportError:  # pragma: no cover — Composer is an optional extra
    LoggerDestination = object  # type: ignore[misc,assignment]
    Callback = object  # type: ignore[misc,assignment]

# Re-export from _core for callers reaching for the parser directly.
parse_aim_run_tags = _core.parse_aim_run_tags

__all__ = [
    "AstrolabeComposerLogger",
    "AstrolabeComposerCheckpointer",
    "parse_aim_run_tags",
]


class AstrolabeComposerLogger(LoggerDestination):
    """Composer LoggerDestination that streams all logged metrics to Aim.

    Reads astrolabe env vars (``ASTROLABE_EXPERIMENT_NAME``,
    ``AIM_RUN_TAGS``, ``ASTROLABE_AIM_URL``) on init; constructor
    arguments are the standalone fallback. Attach via ``Trainer(...,
    loggers=[AstrolabeComposerLogger()])``.

    Parameters
    ----------
    aim_url : str | None
        Aim tracking URL. Overridden by ``ASTROLABE_AIM_URL`` env;
        defaults to the astrolabe SSH-tunneled URL when neither is set.
    experiment_name : str | None
        Aim experiment name. Overridden by ``ASTROLABE_EXPERIMENT_NAME``
        env.
    tags : dict[str, str] | None
        Tags applied to the Aim run on init. Overridden by
        ``AIM_RUN_TAGS`` env when set.
    """

    def __init__(
        self,
        aim_url: str | None = None,
        experiment_name: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        self._cfg = _core.resolve_run_config(
            experiment_name=experiment_name,
            aim_url=aim_url,
            tags=tags,
        )
        self._run: Any = None
        self._wall_time = _core.WallTimeTracker()
        self._rank_zero = is_rank_zero()
        # Schema-phase: tracks which metric names have been registered
        # via a close-and-reopen cycle so the NUC-side sync sidecar
        # can see them in stable SST files (not just memtable). The
        # cycle fires at framework boundary hooks via
        # ``_core.maybe_finalize_schema`` whenever new names have
        # appeared since the previous finalize.
        self._schema_state = _core.SchemaPhaseState()

    # ------------------------------------------------------------------
    # LoggerDestination metric hooks — pass through everything
    # ------------------------------------------------------------------

    def log_metrics(
        self, metrics: dict[str, Any], step: int | None = None
    ) -> None:
        """Pass every user-logged metric through to Aim.

        Composer's ``Logger`` calls this for every
        ``logger.log_metrics(...)`` invocation in the training YAML
        and any user callbacks. The result: any metric the user logs
        — ``MaskedLanguagePerplexity``, custom throughput counters,
        gradient norms — lands in Aim under the same name (with
        cosmetic renames for Composer's automatic emissions; see
        ``_normalize_composer_metric_name``).

        Non-numeric values are silently skipped — Aim only accepts
        scalars. Tensors with ``.item()`` are unwrapped.

        v1.1.2 instrumentation: counters + sampled disk-stats lines
        let post-mortem diagnose whether Composer stopped calling us,
        we skipped at the ``_run is None`` guard, or per-value
        coercion stripped specific metrics. None of these were
        visible before — silent path through the code.
        """
        # Lazily init counters so existing instances keep working in
        # tests / older constructors.
        if not hasattr(self, "_log_metrics_calls"):
            self._log_metrics_calls = 0
            self._log_metrics_skipped_run_none = 0
            self._scalar_skip_counts: dict[str, int] = {}

        self._log_metrics_calls += 1
        # Sample log: first call + every 100th. Distinguishes
        # "Composer stopped calling us" from "Composer called us
        # but we skipped". H2 detector.
        if self._log_metrics_calls == 1 or self._log_metrics_calls % 100 == 0:
            _core._append_stats_line(
                kind="log_metrics_called",
                total_calls=self._log_metrics_calls,
                run_is_none=(self._run is None),
                metrics_size=len(metrics) if metrics else 0,
                step=step,
            )

        if not self._rank_zero:
            return
        if not metrics:
            return
        if self._run is None:
            # H1 smoking gun: log_metrics called WITH metrics but
            # _run got nulled. Pre-v1.1.2 fix this was the silent
            # cliff. Post-fix this should not happen — but log if it
            # does so we know the fix didn't take.
            self._log_metrics_skipped_run_none += 1
            if (self._log_metrics_skipped_run_none == 1
                    or self._log_metrics_skipped_run_none % 100 == 0):
                _core._append_stats_line(
                    kind="log_metrics_skipped_run_none",
                    total_skips=self._log_metrics_skipped_run_none,
                    sample_metric_keys=list(metrics.keys())[:3],
                    step=step,
                )
            return

        for raw_name, value in metrics.items():
            scalar = _to_scalar(value)
            if scalar is None:
                # H3 detector: which specific metric values does
                # _to_scalar reject? Log first occurrence per metric
                # name + every 100th to cap disk write volume.
                name = _normalize_composer_metric_name(raw_name)
                cnt = self._scalar_skip_counts.get(name, 0) + 1
                self._scalar_skip_counts[name] = cnt
                if cnt == 1 or cnt % 100 == 0:
                    _core._append_stats_line(
                        kind="skip_to_scalar",
                        metric=name,
                        value_type=type(value).__name__,
                        per_metric_skips=cnt,
                        step=step,
                    )
                continue
            name = _normalize_composer_metric_name(raw_name)
            _core.observe_name(self._schema_state, name)
            _core.track_safely(
                self._run, name=name, value=scalar, step=step
            )

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        """Apply hyperparameters as Aim run params."""
        if not self._rank_zero or self._run is None or not hyperparameters:
            return
        try:
            self._run["hparams"] = hyperparameters
        except Exception as exc:
            if _core.is_strict():
                raise
            logger.debug("Failed to set hparams: {}", exc)

    # ------------------------------------------------------------------
    # Composer Callback lifecycle (LoggerDestination inherits Callback)
    # ------------------------------------------------------------------

    def init(self, state: Any, logger_obj: Any) -> None:
        """Open the Aim run and apply astrolabe tags."""
        _core._append_stats_line(kind="lifecycle", hook="init", run_hash=None)
        if not self._rank_zero:
            return
        if self._run is not None:
            return  # double-open guard

        run_name = getattr(state, "run_name", None)
        self._run = _core.open_aim_run(self._cfg, run_name=run_name)

    def fit_start(self, state: Any, logger_obj: Any) -> None:
        """Lifecycle marker + defensive reopen.

        New in v1.1.2. If ``_run`` is somehow ``None`` at fit_start
        (e.g., a previous ``fit_end`` closed it because of pre-v1.1.2
        behavior, or an external caller closed it), reopen rather than
        silently no-op the entire fit. Pairs with the v1.1.2 change to
        ``fit_end`` that no longer nulls ``_run``.
        """
        # Full-fidelity run_hash in the jsonl: this file is a data channel,
        # not a debug log — downstream consumers exact-match against
        # aim.Run.hash (24 chars). Log-format strings may truncate for
        # readability; the jsonl must not.
        run_hash = getattr(self._run, "hash", None) if self._run is not None else None
        _core._append_stats_line(
            kind="lifecycle", hook="fit_start",
            run_hash=run_hash,
        )
        if not self._rank_zero:
            return
        if self._run is None:
            run_name = getattr(state, "run_name", None)
            self._run = _core.open_aim_run(self._cfg, run_name=run_name)
            _core._append_stats_line(
                kind="run_reopened",
                reason="fit_start_with_run_none",
                new_run_hash=getattr(self._run, "hash", None),
            )

    def eval_start(self, state: Any, logger_obj: Any) -> None:
        """Pause wall-time accounting during eval."""
        if not self._rank_zero or self._run is None:
            return
        self._wall_time.pause_for_eval()

    def batch_end(self, state: Any, logger_obj: Any) -> None:
        """Anchor wall-time at first training batch + log it, then
        check whether a schema-phase finalize is due.

        Composer's auto-logged train metrics flow through ``log_metrics``
        (called by Composer's Logger before this hook on the same
        batch), so this hook only handles the synthesized ``wall_time``
        metric. The first call also anchors the wall-time clock.

        After the wall_time write, ``_core.maybe_finalize_schema`` runs.
        On batch 1 (first time around), this is the schema-phase
        finalize that registers all of batch 0's observed metric names.
        On subsequent batches with no new names, it's a cheap no-op.
        """
        if not self._rank_zero or self._run is None:
            return
        self._wall_time.mark_first_batch()

        try:
            step = int(state.timestamp.batch)
        except Exception:
            step = None
        _core.observe_name(self._schema_state, "wall_time")
        _core.track_safely(
            self._run,
            name="wall_time",
            value=self._wall_time.elapsed(),
            step=step,
        )

        self._run = _core.maybe_finalize_schema(
            self._run, self._schema_state, cfg=self._cfg
        )

    def eval_end(self, state: Any, logger_obj: Any) -> None:
        """Resume wall-time accounting after eval, then check whether
        eval introduced any new metric names that need a schema-phase
        finalize.

        Eval metrics (``val/loss``, ``val/accuracy``, ...) typically
        first appear at this hook, after batch 0's finalize has
        already happened. The maybe_finalize_schema call here picks
        them up and registers them so they go live on the dashboard
        within the next sync cycle.
        """
        # Resume even if logging was disabled — accounting must be
        # right for the next batch_end.
        self._wall_time.resume()

        if not self._rank_zero or self._run is None:
            return
        self._run = _core.maybe_finalize_schema(
            self._run, self._schema_state, cfg=self._cfg
        )

    def fit_end(self, state: Any, logger_obj: Any) -> None:
        """Lifecycle marker — does NOT close the Aim run.

        v1.1.2 change: pre-fix, this method called ``close_run`` and
        set ``self._run = None``. That was correct for the
        one-trainer-one-fit case but silently broke multi-fit
        trainings (e.g., the Muon→AdamW handoff): the first fit_end
        nulled _run, every subsequent log_metrics hit the
        ``_run is None`` guard and returned silently, the second
        phase's metrics never reached the buffer (or Aim). TB kept
        writing in parallel because it has no equivalent guard.

        Now: fit_end is a lifecycle marker only. The actual close
        runs at ``post_close`` (Composer's trainer-destruction hook)
        which fires exactly once per Trainer lifetime, regardless of
        how many fit() calls happened in between. See callbacks
        CHANGELOG entry for v1.1.2 + the 2026-06-03 investigation.
        """
        run_hash = getattr(self._run, "hash", None) if self._run is not None else None
        _core._append_stats_line(
            kind="lifecycle", hook="fit_end",
            run_hash=run_hash,
        )
        self._fit_end_seen = True

    def post_close(self) -> None:
        """LoggerDestination hook fired during Trainer cleanup.

        v1.1.2 — this is now the SOLE close path. Status is
        ``completed`` if at least one ``fit_end`` was seen (training
        ran to completion at least once), else ``failed`` (trainer
        torn down without ever finishing a fit). Idempotent on
        ``_run is None``.
        """
        run_hash = getattr(self._run, "hash", None) if self._run is not None else None
        _core._append_stats_line(
            kind="lifecycle", hook="post_close",
            run_hash=run_hash,
            fit_end_seen=getattr(self, "_fit_end_seen", False),
        )
        if not self._rank_zero or self._run is None:
            return
        status = "completed" if getattr(self, "_fit_end_seen", False) else "failed"
        _core.close_run(self._run, status=status)
        self._run = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_scalar(value: Any) -> float | None:
    """Coerce a value to ``float``, or return ``None`` if not numeric.

    Handles tensors (anything with ``.item()``), Python scalars, and
    rejects strings, dicts, None, NaN-shaped Aim-rejected types.
    """
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_composer_metric_name(name: str) -> str:
    """Re-namespace Composer's automatic emissions to astrolabe convention.

    Cosmetic only — user-named metrics pass through unchanged. The
    full mapping:

    - ``loss/train/total`` → ``train/loss``
    - ``metrics/train/<name>`` → ``train/<name>``
    - ``metrics/eval/<name>`` → ``<EVAL_PREFIX>/<name>``
    - everything else → unchanged
    """
    if name == "loss/train/total":
        return "train/loss"
    if name.startswith("metrics/train/"):
        return f"train/{name[len('metrics/train/') :]}"
    if name.startswith("metrics/eval/"):
        suffix = name[len("metrics/eval/") :]
        return f"{_core.EVAL_METRIC_PREFIX}/{suffix}"
    return name


def _composer_checkpoint_dir(state) -> str:
    """Directory Composer's ``CheckpointSaver`` last wrote to.

    Composer exposes no checkpoint path on ``State``; the saver's own
    ``saved_checkpoints`` list is the only handle on where the primary
    file landed.
    """
    for callback in getattr(state, "callbacks", []):
        saved = getattr(callback, "saved_checkpoints", None)
        if saved:
            return str(Path(str(saved[-1])).parent)
    raise ValueError("no Composer CheckpointSaver output found — pass export_dir=")


class AstrolabeComposerCheckpointer(Callback):
    """Stamps astrolabe provenance into Composer checkpoints.

    Attach via ``callbacks=`` (not ``loggers=``) alongside your
    existing ``AstrolabeComposerLogger``::

        trainer = Trainer(
            ...,
            save_folder="checkpoints",
            loggers=[AstrolabeComposerLogger()],
            callbacks=[AstrolabeComposerCheckpointer()],
        )

    **This does not save checkpoints.** Composer's own
    ``CheckpointSaver`` — constructed automatically from ``save_folder``
    — does that. This callback only contributes provenance, plus
    optional derived-format exports.

    The embedding is free: Composer serializes every callback's
    ``state_dict()`` into the checkpoint under its class qualname and
    replays it through ``load_state_dict`` on resume. No file is
    reopened or rewritten on this path.

    Parameters
    ----------
    export_formats : list of {"pt", "safetensors"}, optional
        Additional formats to write alongside each Composer checkpoint.
        Empty by default — framework-native only. ``safetensors``
        requires the ``[safetensors]`` extra.
    export_dir : str, optional
        Destination for derived exports. Defaults to the Composer
        checkpoint's own directory.
    """

    def __init__(
        self,
        *,
        export_formats: list[str] | None = None,
        export_dir: str | None = None,
    ) -> None:
        self.export_formats = list(export_formats or [])
        self.export_dir = export_dir
        self._parent: CheckpointMeta | None = None

    def state_dict(self) -> dict:
        """Provenance block written into the checkpoint by Composer.

        Composer calls this on every save. Returns the ``to_dict()``
        form of a freshly-built :class:`~astrolabe_callbacks.checkpoint.CheckpointMeta`
        so the embedded hash reflects the run live at save time, not at
        callback construction.
        """
        return build_checkpoint_meta(parent=self._parent).to_dict()

    def load_state_dict(self, state: dict) -> None:
        """Capture the parent's provenance on resume.

        Retained so a checkpoint written by a resumed run can record
        what it was resumed *from*. Never raises on a malformed or
        absent block — resume must not fail on provenance.
        """
        # Composer replays exactly what state_dict() returned, so ``state``
        # is the block itself rather than a checkpoint wrapping it.
        self._parent = _meta_from_block(state)

    def batch_checkpoint(self, state, logger) -> None:
        write_first_checkpoint_marker_once()
        self._export_derived(state)

    def epoch_checkpoint(self, state, logger) -> None:
        write_first_checkpoint_marker_once()
        self._export_derived(state)

    def _export_derived(self, state) -> None:
        if not self.export_formats or not is_rank_zero():
            return
        try:
            destination = Path(self.export_dir or _composer_checkpoint_dir(state))
            weights = state.model.state_dict()
            meta = build_checkpoint_meta(parent=self._parent)
            for fmt in self.export_formats:
                export_checkpoint(
                    weights,
                    destination / f"{state.run_name}-derived.{fmt}",
                    fmt=fmt,
                    meta=meta,
                )
        except Exception as exc:
            logger.warning("Derived checkpoint export failed: {}", exc)
