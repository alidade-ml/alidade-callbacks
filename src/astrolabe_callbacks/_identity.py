"""Where a result connects, and whose submit it belongs to.

Both answers come from the environment the engine exported into this process,
so any surface that opens its own Aim run — evals, samples — needs them and
none of them should re-derive them.
"""

from __future__ import annotations

import os

from astrolabe_callbacks import contract

__all__ = ["ambient_identity", "resolve_aim_url"]


def resolve_aim_url(aim_url: str | None) -> str:
    """Resolve the Aim connection URL with the lib's standard precedence.

    Order: ``ALIDADE_AIM_REPO_PATH`` > ``ALIDADE_AIM_URL`` > the argument
    > ``contract.DEFAULT_AIM_URL``.

    The repo path comes first because it is the engine saying which transport
    it chose. In local-aim mode it exports that path and opens **no** tunnel,
    while the local ``aim server`` is ``atexit``-registered inside the training
    process — so any later step, being a separate process, finds nothing
    listening on the default ``aim://`` address. Reaching that default ahead of
    a path the engine explicitly supplied is what broke every eval and sample
    write under that mode (AIMURL-1).

    Writing straight to the repo is not a workaround: ``aim.Run`` takes a
    filesystem path as readily as a URL, and the sync sidecar discovers
    whatever runs appear in that directory on each cycle rather than locking
    onto one hash.

    Deliberately does NOT reuse ``resolve_run_config``: that helper resolves a
    run *name* and applies constructor-supplied tags, neither of which these
    runs take. The identity env vars it reads are picked up by
    :func:`ambient_identity` instead.
    """
    # Local-aim mode first: the engine has already decided the transport and
    # said so by exporting a repo path, and it opens no tunnel in that mode.
    # Reaching a hardcoded aim:// default ahead of this is what made every
    # write from a non-training process fail there — see AIMURL-1.
    # aim.Run accepts a filesystem path as readily as a URL; the sync sidecar
    # discovers whatever runs appear in that repo, per cycle.
    repo_path = os.environ.get(contract.ENV_AIM_REPO_PATH)
    if repo_path:
        return repo_path
    return (
        os.environ.get("ALIDADE_AIM_URL")
        or aim_url
        or contract.DEFAULT_AIM_URL
    )


def ambient_identity() -> dict[str, str]:
    """The submit identity the engine exported into this process.

    A script launched as an astrolabe step inherits the same ``AIM_RUN_TAGS``
    the training callback reads — submit, version, submitter, and the GPU rate
    the cost views bill against. Reading it here writes nothing new; the
    identity was already in the process and was being discarded, leaving
    results unattributable to the submit that paid for them.

    ``ALIDADE_EXPERIMENT_NAME`` is authoritative over the tag payload for the
    experiment, matching ``resolve_run_config``'s precedence.
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
