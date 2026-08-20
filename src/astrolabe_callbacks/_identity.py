"""Where a result connects, and whose submit it belongs to.

Both answers come from the environment the engine exported into this process,
so any surface that opens its own Aim run — evals, samples — needs them and
none of them should re-derive them.
"""

from __future__ import annotations

import os

from astrolabe_callbacks import contract
from astrolabe_callbacks._core import DEFAULT_AIM_URL

__all__ = ["ambient_identity", "resolve_aim_url"]


def resolve_aim_url(aim_url: str | None) -> str:
    """Resolve the Aim connection URL with the lib's standard precedence.

    ``ASTROLABE_AIM_URL`` env wins over the constructor argument, which wins
    over :data:`DEFAULT_AIM_URL`.

    **Known gap — AIMURL-1.** Nothing sets ``ASTROLABE_AIM_URL``: the engine
    never exports it, so this always falls through to the default
    ``aim://localhost:43800``. That is the reverse SSH tunnel under the default
    transport and is correct there. Under local-aim mode the engine opens no
    tunnel, and the local ``aim server`` belongs to the training process — so a
    later eval or sample step, being a separate process, finds nothing
    listening. Carried forward unchanged here rather than fixed in a refactor.

    Deliberately does NOT reuse ``resolve_run_config``: that helper resolves a
    run *name* and applies constructor-supplied tags, neither of which these
    runs take. The identity env vars it reads are picked up by
    :func:`ambient_identity` instead.
    """
    return os.environ.get("ASTROLABE_AIM_URL") or aim_url or DEFAULT_AIM_URL


def ambient_identity() -> dict[str, str]:
    """The submit identity the engine exported into this process.

    A script launched as an astrolabe step inherits the same ``AIM_RUN_TAGS``
    the training callback reads — submit, version, submitter, and the GPU rate
    the cost views bill against. Reading it here writes nothing new; the
    identity was already in the process and was being discarded, leaving
    results unattributable to the submit that paid for them.

    ``ASTROLABE_EXPERIMENT_NAME`` is authoritative over the tag payload for the
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
