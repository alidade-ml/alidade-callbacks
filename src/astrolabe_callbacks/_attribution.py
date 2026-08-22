"""Who is a result about?

Moved here from ``eval_results.py`` so ``log_samples`` shares one implementation
rather than growing a second. Behaviour is unchanged.

Resolution order: explicit ``model_run_hash`` > the checkpoint's embedded
provenance > ``external_name`` (which registers an entry) > raise.

Three constraints, each of which a plausible change would break:

- **Never read Aim to resolve.** Under local-aim transport the compute host
  sees a repo holding only its own submit's runs, so a name lookup finds
  nothing and registers a duplicate.
- **A file with a submit but no run is unresolved.** Do not fall back to
  picking among that submit's runs.
- **Raise before doing any work.** An unattributed run is invisible to the
  dashboard forever, so failing late costs a benchmark; failing early costs
  nothing.
"""

from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any

from loguru import logger

__all__ = [
    "AttributionInputError",
    "MissingParentError",
    "resolve_parent",
]


class AttributionInputError(ValueError):
    """Raised when an attribution argument is malformed.

    Surfaces at the researcher's call site rather than writing a half-formed
    Aim run that confuses the dashboard later.
    """


class MissingParentError(Exception):
    """Raised when a result cannot be attributed to any run.

    Deliberately not an :class:`AttributionInputError`: the call was
    well-formed, the *artifact* carries no provenance. Distinct so a pilot
    running with raising enabled in CI can catch exactly this without also
    catching malformed arguments.
    """


def _parent_run_hash(checkpoint: str | Path | dict[str, Any] | None) -> str | None:
    """The training run a checkpoint attributes to, or ``None``.

    Deliberately a two-line read rather than a resolution helper: the
    checkpoint's own ``aim_run_hash`` is the answer, because transforms copy it
    forward at write time instead of leaving the reader to walk a chain.
    """
    if checkpoint is None:
        return None
    from astrolabe_callbacks.checkpoint import read_checkpoint_meta

    meta = read_checkpoint_meta(checkpoint)
    return meta.aim_run_hash if meta else None


def resolve_parent(
    *,
    checkpoint: str | Path | dict[str, Any] | None = None,
    model_run_hash: str | None = None,
    external_name: str | None = None,
    on_missing_parent: str = "raise",
    aim_url: str | None = None,
    what: str = "result",
) -> str | None:
    """Resolve the run this result is about. ``None`` only when not raising.

    ``what`` names the caller ("eval", "samples") in error text; it changes
    nothing else.
    """
    if on_missing_parent not in ("warn", "raise"):
        raise AttributionInputError(
            f"on_missing_parent must be 'warn' or 'raise', got {on_missing_parent!r}"
        )
    if model_run_hash is not None and (
        not isinstance(model_run_hash, str) or not model_run_hash
    ):
        raise AttributionInputError(
            "model_run_hash, when given, must be a non-empty string; pass None "
            "to resolve from the checkpoint"
        )
    if external_name is not None and (
        not isinstance(external_name, str) or not external_name.strip()
    ):
        # Checked here rather than at use: an empty string is falsy, so it
        # would otherwise fall through to "nothing was given" and report a
        # missing parent at a call site that plainly supplied a name.
        raise AttributionInputError(
            "external_name, when given, must be a non-empty string"
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
        resolved = mint_model_entry(external_name, aim_url)

    if resolved:
        return resolved

    if on_missing_parent == "raise":
        raise MissingParentError(
            f"nothing to attribute this {what} to: the checkpoint carries no "
            "astrolabe provenance, and neither model_run_hash= nor "
            f"external_name= was given. If astrolabe trained this model, stamp "
            "the checkpoint or pass model_run_hash=. If you downloaded it, "
            'pass external_name= to name it — e.g. external_name="roberta-base".'
        )
    return None


def mint_model_entry(name: str, aim_url: str | None) -> str:
    """Record a model astrolabe never trained, and return its run hash.

    Only reachable through ``external_name=``. A downloaded checkpoint has no
    training run, so a result scoring it has nothing to attribute to and the
    dashboard has no row to put in a leaderboard; this is the row. It carries
    no metrics — it exists to be pointed at.

    Filed under the submitting experiment, carrying the same identity as every
    other run of that submit. There is no way to reach this outside a submit,
    which matters: with no experiment to inherit, the entry would land in Aim's
    ``default`` bucket, attached to nothing.

    **One entry per submit, not per call.** The first call for a name mints an
    entry and records it in the per-submit registry the engine points
    ``ASTROLABE_EXTERNAL_MODELS`` at; later calls in the same submit read it
    back. Scoring one downloaded model on GLUE, MMLU and BEIR as three steps
    therefore produces three results about one model, rather than three models
    with one result each — which broke ``--include`` silently, since including
    by name resolves to whichever entry was newest and carried a third of the
    evidence.

    A local file rather than a lookup: asking Aim which entry to reuse returns
    nothing under local-aim transport, where the compute host sees only its own
    submit's runs, so it would mint a duplicate anyway and say nothing. Steps of
    a submit share an instance and run sequentially, so a file on that instance
    is exactly the right scope and needs no locking.

    **Across submits the identity is deliberately NOT shared.** A later submit
    mints a new entry for the same name. Sharing would need a registry that
    outlives the instance, and there is nowhere to put one that compute can
    reach.
    """

    from aim import Run

    from astrolabe_callbacks import contract
    from astrolabe_callbacks._identity import ambient_identity, resolve_aim_url

    recorded = _read_external_model(name)
    if recorded:
        logger.debug("reusing external model entry {} for {!r}", recorded, name)
        return recorded

    identity = ambient_identity()
    run = Run(
        experiment=identity.get(contract.TAG_EXPERIMENT),
        repo=resolve_aim_url(aim_url),
    )
    try:
        for key, value in identity.items():
            run[key] = value
        run[contract.TAG_KIND] = contract.KIND_EXTERNAL_CHECKPOINT
        try:
            run.name = name
        except Exception as exc:  # older Aim treats name as read-only
            logger.debug("Failed to set Aim run name {}: {}", name, exc)
        _record_external_model(name, run.hash)
        return run.hash
    finally:
        run.close()


def _external_models_path() -> Path | None:
    """The per-submit registry path, or None outside a submit.

    Absent outside astrolabe orchestration, where there is no submit to
    scope an identity to and every call minting its own entry is the
    honest answer.
    """
    from astrolabe_callbacks import contract

    raw = os.environ.get(contract.ENV_EXTERNAL_MODELS)
    return Path(os.path.expanduser(raw)) if raw else None


def _read_external_model(name: str) -> str | None:
    """Look up a previously minted entry for this name in this submit.

    Best-effort in every failure mode: a missing, unreadable or corrupt
    registry means "mint a new one", which is exactly the behaviour that
    existed before the registry did. Losing the file costs a duplicate
    entry; raising here would cost the result.
    """
    path = _external_models_path()
    if path is None or not path.exists():
        return None
    try:
        entries = json.loads(path.read_text())
        value = entries.get(name)
        return value if isinstance(value, str) and value else None
    except Exception as exc:  # noqa: BLE001 — a bad registry must not fail a result
        logger.debug("unreadable external-model registry at {}: {}", path, exc)
        return None


def _record_external_model(name: str, run_hash: str) -> None:
    """Record a minted entry so later steps of this submit reuse it.

    Written whole rather than appended, and via a temp file plus rename,
    so a reader never sees a half-written registry. Steps run
    sequentially on one instance, so there is no writer to race with —
    the rename is for crash safety, not concurrency.
    """
    path = _external_models_path()
    if path is None:
        return
    try:
        entries = {}
        if path.exists():
            try:
                entries = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 — a corrupt registry is replaced
                entries = {}
        entries[name] = run_hash
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(entries, indent=2))
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 — best-effort, never fail a result
        logger.debug("could not record external model {!r}: {}", name, exc)
