"""Who is a result about?

One implementation, shared by every surface that logs something *about* a
model — evals and samples today. The question is identical in both cases, and
two implementations would drift: a model attributed one way in the Eval tab
and another in Examples is a bug unexplainable from either side.

Resolution order, highest first:

1. an explicit ``model_run_hash``
2. the checkpoint's own embedded provenance
3. ``external_name``, which registers an entry for a model astrolabe never
   trained
4. ``on_missing_parent`` decides — raise (default) or return ``None``

Three properties are load-bearing and easy to lose in a refactor:

**Resolution never reads Aim.** The checkpoint is read offline — header only
for safetensors, ``map_location="meta"`` for torch pickle. This is not an
optimisation. Looking a parent up by name returns *nothing* under local-aim
transport, where the compute host sees a repo holding only its own submit's
runs, so a lookup silently finds nothing and registers a duplicate every time.

**A file carrying a submit but no run is unresolved, not a lookup trigger.**
Picking among several runs under one submit would be a guess, and guessing by
resemblance is the mechanism this exists to replace.

**Raising happens before any work.** An unattributed run lands in Aim and is
invisible to the dashboard forever — an hour of GPU time producing results
nobody can find. Refusing to start costs nothing.
"""

from __future__ import annotations

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

    Never reads from Aim. Passing the same name again — another step, another
    submit — makes another entry. Reusing one would mean asking Aim which run
    to reuse, and that lookup cannot work under local-aim transport, where the
    compute host sees only its own submit's runs: it would find nothing and
    mint a duplicate anyway, silently.
    """
    from aim import Run

    from astrolabe_callbacks import contract
    from astrolabe_callbacks._identity import ambient_identity, resolve_aim_url

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
        return run.hash
    finally:
        run.close()
