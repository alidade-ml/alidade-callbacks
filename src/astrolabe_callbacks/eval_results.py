"""Producer-side helpers for logging benchmark eval results.

Researchers call :func:`log_eval_table` (primary, one-shot) or
:func:`start_eval_run` (lower-level escape hatch for streams / custom
metric names) from a post-training eval script. Both emit an Aim run
tagged with the three-tag contract that astrolabe's dashboard
discovers from the model-run page:

* ``astrolabe.kind = "eval"`` — discriminator alongside ``"metadata"``
  (engine-written cost runs) and the implicit training runs.
* ``astrolabe.task_set = "glue"`` — human label that groups sections
  on the dashboard's Eval tab.
* ``astrolabe.model_run_hash = "<training_run_hash>"`` — the join key.
  One eval run scores exactly one training run.

Metric path convention: ``eval/<task>/<metric>`` — the dashboard parses
this to populate the table's row (task) and metric column.

This lives in ``astrolabe-callbacks`` rather than the main ``astrolabe``
package so training/eval repos depend on **one** lightweight library
for all Aim instrumentation — they never pull in the orchestration
framework. It uses the same ``aim_url`` / ``ASTROLABE_AIM_URL``
connection convention as the framework callbacks and the raw-PyTorch
``Run`` context manager.

The ``eval/`` namespace here is distinct from ``val/`` (during-training
validation metrics emitted by the framework callbacks). ``val/`` lives
on the training run and the dashboard's Training tab; ``eval/`` lives on
a separate eval run and the Eval tab. See the package README for the
full namespace split.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from astrolabe_callbacks import contract

from ._core import DEFAULT_AIM_URL

__all__ = [
    "EvalInputError",
    "MissingParentError",
    "log_eval_table",
    "start_eval_run",
    "start_eval_run_from_checkpoint",
]


class EvalInputError(ValueError):
    """Raised when an eval helper receives malformed input.

    Surfaces in the researcher's eval script with a clear message —
    we'd rather fail loudly at the call site than write a half-formed
    Aim run that confuses the dashboard later.
    """


class MissingParentError(Exception):
    """Raised when an eval cannot be attributed to any training run.

    Deliberately not an :class:`EvalInputError`: the call was
    well-formed, the *artifact* carries no provenance. Distinct so a
    pilot running ``on_missing_parent="raise"`` in CI can catch exactly
    this without also catching malformed arguments.
    """


def _resolve_aim_url(aim_url: str | None) -> str:
    """Resolve the Aim connection URL with the lib's standard precedence.

    ``ASTROLABE_AIM_URL`` env wins over the constructor argument, which
    wins over :data:`DEFAULT_AIM_URL`. Matches ``resolve_run_config`` so
    an eval script run on the same instance as training connects the
    same way without extra configuration.

    Note this deliberately does NOT reuse ``resolve_run_config``: that
    helper resolves a run *name* and applies constructor-supplied tags,
    neither of which an eval run takes. The identity env vars it reads
    are picked up by :func:`_ambient_identity` instead.
    """
    return os.environ.get("ASTROLABE_AIM_URL") or aim_url or DEFAULT_AIM_URL


def _ambient_identity() -> dict[str, str]:
    """The submit identity the engine exported into this process.

    An eval script launched as an astrolabe step inherits the same
    ``AIM_RUN_TAGS`` the training callback reads — submit, version,
    submitter, and the GPU rate the cost views bill against. Reading it
    here writes nothing new; the identity was already in the process and
    was being discarded, leaving evals unattributable to the submit that
    paid for them.

    ``ASTROLABE_EXPERIMENT_NAME`` is authoritative over the tag payload
    for the experiment, matching ``resolve_run_config``'s precedence.
    """
    tags = {
        key: value
        for key, value in contract.parse_aim_run_tags(
            os.environ.get(contract.ENV_AIM_RUN_TAGS)
        ).items()
        if value
    }
    experiment = os.environ.get(contract.ENV_EXPERIMENT_NAME)
    if experiment:
        tags[contract.TAG_EXPERIMENT] = experiment
    return tags


def _open_eval_run(*, task_set: str, aim_url: str | None) -> Any:
    """Open an Aim run carrying everything an eval has except its model.

    Shared by the linked and unlinked paths so filing and identity stay
    identical between them — an unlinked run someone stamps later has to
    end up indistinguishable from one that resolved on the first try.

    Filing follows the submitted config, not the benchmark: an
    experiment is one hypothesis, and ``eval/<task_set>`` puts the
    benchmark on that axis instead. It is also the only filing that
    survives both transports — in local-aim mode the sync sidecar
    rewrites synced runs to the submit's experiment name, so
    ``eval/<task_set>`` already does not hold there. ``task_set`` stays a
    tag, which is what groups the dashboard's tables. Outside a submit
    there is no experiment to inherit, so the old name remains the
    fallback.
    """
    from aim import Run

    identity = _ambient_identity()
    experiment = identity.get(contract.TAG_EXPERIMENT) or f"eval/{task_set}"

    run = Run(experiment=experiment, repo=_resolve_aim_url(aim_url))
    # Identity first, contract tags second: the discovery tags are the
    # only reason the dashboard can see this run at all, so an unexpected
    # key in the ambient payload must not be able to shadow one.
    for key, value in identity.items():
        run[key] = value
    run[contract.TAG_KIND] = contract.KIND_EVAL
    run[contract.TAG_TASK_SET] = task_set
    return run


# Entries minted in this process, keyed by (name, aim url). Scoring one
# downloaded model on GLUE and MMLU from the same script should give it
# one row, not two. Deliberately process-local: reusing an entry from an
# earlier step or submit would mean asking Aim which run to reuse, and
# that lookup is what this whole path exists to avoid — it cannot work
# under local-aim transport, where the compute host sees only its own
# submit's runs.
_EXTERNAL_ENTRIES: dict[tuple[str, str], str] = {}


def _register_external_model(*, name: str, aim_url: str | None = None) -> str:
    """Give a model astrolabe never trained a record in Aim, and return
    its run hash.

    Internal on purpose. ``external_name=`` on the eval helper is the
    only supported way in, because that path only runs where the submit
    identity exists. Called directly outside a submit there is no
    experiment to file under and the entry lands in Aim's ``default``
    bucket — detached from any experiment, which is the one place a
    model entry must never be.

    A downloaded checkpoint has no training run, so an eval scoring it
    has nothing to attribute to and the dashboard has no row to put in a
    leaderboard. This creates that record. It carries no metrics — it
    exists to be pointed at.

    Filed under the submitting experiment, like every other run this
    submit produces, and carrying the same ambient identity. Registering
    the same model from a later submit makes a second entry; that is the
    accepted cost of never reading from Aim (see below), and each
    experiment page stays self-contained.

    **Deliberately does not look for an existing entry.** Searching Aim
    by name is the guess-by-resemblance pattern the checkpoint helpers
    exist to replace, and it cannot work at all under local-aim
    transport, where the compute host writes to a local repo holding
    only this submit's runs — the search would find nothing and mint a
    duplicate every time, silently. Call this once per script and reuse
    the hash for every task set.

    Parameters
    ----------
    name : str
        What to call the model — ``"roberta-base"``. Becomes the Aim run
        name and the label on every leaderboard row.
    aim_url : str, optional
        Aim tracking URL. ``ASTROLABE_AIM_URL`` wins over this argument.

    Returns
    -------
    str
        The new run's hash. Pass it as ``model_run_hash``.

    Raises
    ------
    EvalInputError
        ``name`` is empty or not a string.
    """
    if not isinstance(name, str) or not name.strip():
        raise EvalInputError("name must be a non-empty string")

    url = _resolve_aim_url(aim_url)
    cached = _EXTERNAL_ENTRIES.get((name, url))
    if cached:
        return cached

    from aim import Run

    identity = _ambient_identity()
    experiment = identity.get(contract.TAG_EXPERIMENT)

    run = Run(experiment=experiment, repo=_resolve_aim_url(aim_url))
    try:
        for key, value in identity.items():
            run[key] = value
        run[contract.TAG_KIND] = contract.KIND_EXTERNAL_CHECKPOINT
        try:
            run.name = name
        except Exception as exc:  # older Aim treats name as read-only
            logger.debug("Failed to set Aim run name {}: {}", name, exc)
        _EXTERNAL_ENTRIES[(name, url)] = run.hash
        return run.hash
    finally:
        run.close()


def _validate_identity(model_run_hash: str, task_set: str) -> None:
    if not isinstance(model_run_hash, str) or not model_run_hash:
        raise EvalInputError("model_run_hash must be a non-empty string")
    if not isinstance(task_set, str) or not task_set:
        raise EvalInputError("task_set must be a non-empty string")


def _validate_rows(rows: dict[str, tuple[str, float]]) -> None:
    if not isinstance(rows, dict):
        raise EvalInputError(
            f"rows must be a dict, got {type(rows).__name__}"
        )
    if not rows:
        raise EvalInputError("rows must contain at least one task")
    for task, value in rows.items():
        if not isinstance(task, str) or not task:
            raise EvalInputError(
                f"task name must be a non-empty string, got {task!r}"
            )
        if "/" in task:
            # The dashboard parses ``eval/<task>/<metric>`` — embedding
            # a slash in the task name silently scrambles which segment
            # is which.
            raise EvalInputError(
                f"task name {task!r} must not contain '/'; "
                f"use a flat label per task"
            )
        if not isinstance(value, tuple) or len(value) != 2:
            raise EvalInputError(
                f"row {task!r} must be a (metric, score) tuple, got {value!r}"
            )
        metric, score = value
        if not isinstance(metric, str) or not metric:
            raise EvalInputError(
                f"metric label for task {task!r} must be a non-empty string, "
                f"got {metric!r}"
            )
        if "/" in metric:
            raise EvalInputError(
                f"metric label {metric!r} for task {task!r} must not contain '/'"
            )
        # bool is a subclass of int in Python — exclude it explicitly so
        # ``log_eval_table(rows={"cola": ("accuracy", True)})`` fails
        # loudly instead of logging 1.0.
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise EvalInputError(
                f"score for task {task!r} must be a number, "
                f"got {type(score).__name__}"
            )


def start_eval_run(
    *,
    model_run_hash: str,
    task_set: str,
    aim_url: str | None = None,
) -> Any:
    """Open an Aim run tagged for eval discovery.

    Lower-level helper for mid-training rolling evals, custom metric
    names, or anywhere the researcher needs full control over the Aim
    run's lifecycle. For the common case (one-shot post-training table)
    use :func:`log_eval_table` instead.

    The caller owns the returned ``aim.Run`` — they call ``.track(...)``
    to log values and ``.close()`` when finished. Forgetting to close
    leaves the run's ``end_time`` as zero; the dashboard will still
    display the run, but cost / duration views may treat it as
    in-flight indefinitely.

    Parameters
    ----------
    model_run_hash : str
        Hash of the training Aim run this eval scores. Must be
        non-empty. Becomes ``astrolabe.model_run_hash`` on the tag set;
        the dashboard uses this to discover eval runs from the model's
        experiment page.
    task_set : str
        Human label for the benchmark suite (``"glue"``, ``"mmlu"``,
        ``"agent-rollouts-2026q2"``, etc.). Becomes
        ``astrolabe.task_set``. Groups sections in the dashboard.
    aim_url : str | None
        Aim tracking URL. ``ASTROLABE_AIM_URL`` env wins over this
        argument; defaults to ``aim://localhost:43800`` (the SSH
        reverse tunnel astrolabe opens on GPU instances). Accepts a
        filesystem path too — ``aim.Run`` resolves either.

    Returns
    -------
    aim.Run
        An open Aim run with the three identity tags already set.

    Raises
    ------
    EvalInputError
        If ``model_run_hash`` or ``task_set`` is empty or not a string.
    ImportError
        If the ``aim`` package isn't installed (re-raised, not
        swallowed — eval scripts that can't reach Aim should fail
        loudly).

    Examples
    --------
    >>> run = start_eval_run(
    ...     model_run_hash="abc123",
    ...     task_set="glue",
    ... )
    >>> for checkpoint in (10_000, 20_000, 30_000):
    ...     run.track(
    ...         score_at(checkpoint),
    ...         name="eval/cola/matthews",
    ...         step=checkpoint,
    ...     )
    >>> run.close()
    """
    _validate_identity(model_run_hash, task_set)

    run = _open_eval_run(task_set=task_set, aim_url=aim_url)
    run[contract.TAG_MODEL_RUN_HASH] = model_run_hash
    return run


def start_eval_run_from_checkpoint(
    *,
    checkpoint: str | Path | dict[str, Any],
    task_set: str,
    aim_url: str | None = None,
    model_run_hash: str | None = None,
    external_name: str | None = None,
    on_missing_parent: str = "raise",
) -> Any:
    """Open an eval run already linked to the training that produced a
    checkpoint.

    The eval author names the file they are about to evaluate; the
    training run comes out of the file's own provenance. No hash, no
    tag key, and no knowledge of astrolabe internals at the call site.

    Resolution order: an explicit ``model_run_hash`` wins, then the
    checkpoint's embedded ``aim_run_hash``, then ``external_name`` (which
    registers the model), then ``on_missing_parent`` decides. Resolution
    is **offline** — the checkpoint is read, Aim is never queried. A file
    that carries a submit but no run is treated as unresolved rather than
    triggering a lookup, because picking among several runs under one
    submit would be a guess, and guessing by resemblance is the mechanism
    this helper exists to replace.

    A derived checkpoint (surgery, quantization, extraction) carries the
    run that trained the model it came from, so evaluating one attributes
    to the original training rather than to nothing.

    Parameters
    ----------
    checkpoint : str | Path | dict
        Path to a ``.pt`` / ``.safetensors`` file, or an already-loaded
        state dict. Paths are read cheaply — header-only for
        safetensors, ``map_location="meta"`` for torch pickle — so
        passing a path costs no tensor allocation.
    task_set : str
        Benchmark suite label (``"glue"``, ``"mmlu"``, ...).
    aim_url : str, optional
        Aim tracking URL. ``ASTROLABE_AIM_URL`` wins over this argument.
    model_run_hash : str, optional
        Explicit parent, overriding whatever the file says. For
        checkpoints written before provenance existed, or when you know
        the parent and the artifact does not.
    external_name : str, optional
        Name for a model astrolabe never trained — ``"roberta-base"``.
        Registers it and attributes
        the eval to that entry. Needed only when the checkpoint carries
        no provenance, so it never appears in code evaluating your own
        models. A checkpoint that does carry provenance wins over this
        argument.
    on_missing_parent : {"raise", "warn"}, default "raise"
        What to do when nothing resolves. ``"raise"`` raises
        :class:`MissingParentError`. Default because it fires at the
        *start* of an eval, before any scoring: refusing to begin costs
        nothing, whereas ``"warn"`` returns a run with no
        ``model_run_hash``, which lands in Aim and stays invisible to the
        dashboard forever — an hour of GPU time producing numbers nobody
        can find. ``"warn"`` remains for callers that stamp afterwards.

    Returns
    -------
    aim.Run
        Open, and carrying ``astrolabe_linked`` so a caller can branch
        on whether attribution actually happened rather than inferring
        it from log output.

    Raises
    ------
    EvalInputError
        ``task_set`` or an explicitly passed ``model_run_hash`` is
        empty or not a string, or ``on_missing_parent`` is not one of
        the two accepted values.
    MissingParentError
        No parent resolved and ``on_missing_parent="raise"``.
    FileNotFoundError
        ``checkpoint`` is a path that does not exist.
    ValueError
        ``checkpoint`` is a path that is not a recognized format.

    Examples
    --------
    >>> run = start_eval_run_from_checkpoint(
    ...     checkpoint="ckpt.pt",
    ...     task_set="cola",
    ... )
    >>> for step, mcc in scores:
    ...     run.track(mcc, name="eval/cola/matthews", step=step)
    >>> run.close()
    """
    if on_missing_parent not in ("warn", "raise"):
        raise EvalInputError(
            f"on_missing_parent must be 'warn' or 'raise', got {on_missing_parent!r}"
        )
    if not isinstance(task_set, str) or not task_set:
        raise EvalInputError("task_set must be a non-empty string")
    if model_run_hash is not None and (
        not isinstance(model_run_hash, str) or not model_run_hash
    ):
        raise EvalInputError(
            "model_run_hash, when given, must be a non-empty string; pass None "
            "to resolve from the checkpoint"
        )

    resolved = model_run_hash or _parent_run_hash(checkpoint)
    if resolved and external_name:
        # A sweep over a mix of your own and downloaded models will pass
        # external_name unconditionally; refusing the overlap would break
        # the obvious way to write that loop. The file knows better than
        # the argument, so the file wins — but say so, or the name looks
        # like it took effect.
        logger.info(
            "ignoring external_name={!r} — the checkpoint carries its own "
            "provenance, which wins",
            external_name,
        )
    if not resolved and external_name:
        resolved = _register_external_model(name=external_name, aim_url=aim_url)

    if resolved:
        run = start_eval_run(
            model_run_hash=resolved, task_set=task_set, aim_url=aim_url
        )
        run.astrolabe_linked = True
        return run

    if on_missing_parent == "raise":
        raise MissingParentError(
            "nothing to attribute this eval to: the checkpoint carries no "
            "astrolabe provenance, and neither model_run_hash= nor "
            "external_name= was given. If astrolabe trained this model, stamp "
            "the checkpoint or pass model_run_hash=. If you downloaded it, "
            'pass external_name= to name it — e.g. external_name="roberta-base".'
        )

    logger.warning(
        "Eval run for task_set={} is UNLINKED — the checkpoint carries no "
        "training run hash. It will not appear in the dashboard's Eval tab "
        "until the run is stamped with its parent.",
        task_set,
    )
    # Not routed through start_eval_run, which requires a non-empty hash
    # by design — a half-tagged run written through the normal path would
    # be worse than an openly unlinked one.
    run = _open_eval_run(task_set=task_set, aim_url=aim_url)
    run.astrolabe_linked = False
    return run


def _parent_run_hash(checkpoint: str | Path | dict[str, Any]) -> str | None:
    """The training run a checkpoint attributes to, or ``None``.

    Deliberately a two-line read rather than a resolution helper: the
    checkpoint's own ``aim_run_hash`` is the answer, because transforms
    copy it forward at write time instead of leaving the reader to walk
    a chain.
    """
    from astrolabe_callbacks.checkpoint import read_checkpoint_meta

    meta = read_checkpoint_meta(checkpoint)
    return meta.aim_run_hash if meta else None


def log_eval_table(
    *,
    model_run_hash: str,
    task_set: str,
    rows: dict[str, tuple[str, float]],
    aim_url: str | None = None,
) -> str:
    """Log a one-shot benchmark table for a single training run.

    Primary surface — researchers hand a dict, the library handles the
    Aim mechanics. Each ``(task, (metric, score))`` entry becomes a
    metric tracked at ``step=0`` under the name ``eval/<task>/<metric>``.
    The dashboard's table block parses this convention to populate the
    leaderboard column for that task.

    The Aim run is opened, tagged, populated, and closed atomically.
    If validation fails, no Aim run is created.

    Parameters
    ----------
    model_run_hash : str
        Hash of the training Aim run this eval scores.
    task_set : str
        Human label for the benchmark suite (``"glue"``, ``"mmlu"``, …).
    rows : dict[str, tuple[str, float]]
        ``{task_name: (metric_label, score)}``. ``task_name`` and
        ``metric_label`` are flat strings (no slashes). ``score`` is a
        number (int or float, not bool). At least one row is required.

        For an averaged-across-tasks summary column, log it as one of
        the rows (conventionally ``"avg"``) — the dashboard renders
        ``"avg"`` as the last column. The library does not compute
        aggregates itself; that's the researcher's call (mean? harmonic
        mean? a paper-specific subset?).
    aim_url : str | None
        Aim tracking URL. ``ASTROLABE_AIM_URL`` env wins; defaults to
        ``aim://localhost:43800``.

    Returns
    -------
    str
        The Aim run hash of the newly-created eval run.

    Raises
    ------
    EvalInputError
        If any input is malformed. No Aim run is created in that case.
    ImportError
        If the ``aim`` package isn't installed.

    Examples
    --------
    >>> log_eval_table(
    ...     model_run_hash="abc123",
    ...     task_set="glue",
    ...     rows={
    ...         "cola": ("matthews", 0.822),
    ...         "sst2": ("accuracy", 0.943),
    ...         "mnli": ("accuracy_matched", 0.864),
    ...         "avg":  ("mean", 0.876),
    ...     },
    ... )
    'b73e9c8d4f6a...'
    """
    _validate_identity(model_run_hash, task_set)
    _validate_rows(rows)

    run = start_eval_run(
        model_run_hash=model_run_hash,
        task_set=task_set,
        aim_url=aim_url,
    )
    try:
        for task, (metric, score) in rows.items():
            run.track(float(score), name=f"eval/{task}/{metric}", step=0)
        run_hash = run.hash
    finally:
        run.close()
    return run_hash
