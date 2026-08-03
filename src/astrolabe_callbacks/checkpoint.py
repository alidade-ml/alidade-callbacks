"""Framework-agnostic checkpoint provenance: embed astrolabe identity
into checkpoints so post-training eval can attribute results to the
training that produced the model.

Design notes that are not obvious from the code:

**Identity is propagated, not looked up.** ``submit_id`` /
``experiment`` / ``version`` all come from the engine-set env
(``contract.ENV_*``). They are available with no live Aim run, no
network, and no framework object. ``aim_run_hash`` is the sole field
Aim mints at run-open time, so it is *enrichment only* — present when
a logger happens to be live, absent otherwise, never required for a
checkpoint to be stamped.

**Why the hash is still carried.** One submit can produce several Aim
training runs — a multi-step experiment runs each step as its own
process, and every step inherits the same ``AIM_RUN_TAGS``. Propagated
identity is therefore 1:N, and the hash is what disambiguates. Single
-step experiments (the common case) do not need it.

**Three embedding mechanisms, by necessity.** Every one of them writes
provenance *into* the artifact — never beside it — because a sidecar is
lost the moment someone copies just the weights.

1. **Native slot** (Composer, Lightning). Both serialize callback state
   into the checkpoint themselves, so we fill their slot and never
   reopen the written file.
2. **Top-level key** (raw PyTorch, derived exports). No framework
   involved, so the block goes in directly under ``META_KEY``.
3. **Registered buffer** (HuggingFace). HF has no checkpoint-dict hook
   at all: ``on_save`` fires post-write with no dict, and
   ``save_pretrained`` hardcodes the safetensors metadata. The
   alternatives were subclassing ``Trainer`` (permanent residence in
   the user's MRO) or rewriting the finished file (O(model size) I/O
   per save). A persistent uint8 buffer rides into ``state_dict()``
   instead, costs O(metadata), and needs no override.

   Mechanism 3 has a real cost: it adds a key a freshly-built model
   lacks, so ``load_state_dict(..., strict=True)`` on such a model
   raises. ``from_pretrained`` is non-strict and only warns. See
   :func:`embed_meta_as_buffer` and :func:`strip_meta_buffer`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from astrolabe_callbacks import contract

__all__ = [
    "CheckpointMeta",
    "META_KEY",
    "build_checkpoint_meta",
    "read_checkpoint_meta",
    "stamp_state_dict",
    "export_checkpoint",
    "embed_meta_as_buffer",
    "read_meta_from_buffer",
    "strip_meta_buffer",
    "BUFFER_NAME",
]


# Top-level key under which the meta block is embedded in formats that
# have no native callback-state slot (.pt via raw torch.save, and the
# safetensors header).
META_KEY = "_astrolabe_meta"

ExportFormat = Literal["pt", "safetensors"]


@dataclass(frozen=True)
class CheckpointMeta:
    """Astrolabe provenance embedded in a checkpoint.

    Attributes
    ----------
    submit_id : str | None
        Engine-minted submit identifier. ``None`` outside astrolabe
        orchestration (ad-hoc local training) — a stamped-but-unlinked
        checkpoint is a supported state, not an error.
    experiment : str | None
        Experiment name from ``ASTROLABE_EXPERIMENT_NAME``.
    version : str | None
        Submit version label (``v1``, ``v2``, ...).
    aim_run_hash : str | None
        Hash of the Aim run that produced this checkpoint. Enrichment
        only — populated when a live astrolabe logger is registered in
        this process, ``None`` otherwise. Readers that need to
        disambiguate multiple runs under one submit require this.
    created_at : str
        UTC ISO-8601 timestamp of the stamp.
    derived_from : str | None
        ``aim_run_hash`` of the parent when this checkpoint was
        produced by transforming another (surgery, distillation,
        format conversion). ``None`` for checkpoints written directly
        by training.
    derivation_chain_length : int
        Number of transform hops from an originally-trained
        checkpoint. ``0`` for training output.
    """

    submit_id: str | None = None
    experiment: str | None = None
    version: str | None = None
    aim_run_hash: str | None = None
    created_at: str = ""
    derived_from: str | None = None
    derivation_chain_length: int = 0

    @property
    def linked(self) -> bool:
        """``True`` when enough identity is present to attribute this
        checkpoint to a training run."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the on-disk form. Omits ``None`` fields so the
        embedded block stays small and forward-compatible."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CheckpointMeta":
        """Parse an embedded block, tolerating unknown keys written by
        a newer callback version."""
        raise NotImplementedError


def build_checkpoint_meta(
    *,
    derived_from: str | None = None,
    derivation_chain_length: int = 0,
) -> CheckpointMeta:
    """Construct provenance for a checkpoint about to be written.

    Reads propagated identity from the environment and opportunistically
    attaches the live Aim run hash if a logger is active in this
    process.

    Parameters
    ----------
    derived_from : str, optional
        Parent run hash when stamping a transformed checkpoint.
    derivation_chain_length : int, default 0
        Hops from an originally-trained checkpoint.

    Returns
    -------
    CheckpointMeta
        Never raises. Outside astrolabe orchestration every propagated
        field is ``None`` and the result is an unlinked stamp.
    """
    raise NotImplementedError


def read_checkpoint_meta(
    checkpoint: str | Path | dict[str, Any],
) -> CheckpointMeta | None:
    """Extract embedded provenance from a checkpoint.

    Parameters
    ----------
    checkpoint : str | Path | dict
        Path to a ``.pt`` / ``.safetensors`` file, or an already-loaded
        state dict. Paths are read cheaply — safetensors via a header
        -only open, torch pickle via ``map_location="meta"`` so no
        tensor storage is allocated.

    Returns
    -------
    CheckpointMeta | None
        ``None`` when the file carries no astrolabe block (pre-upgrade
        checkpoints, externally-sourced models). Callers decide whether
        that is a warning or an error; this function never raises on an
        unstamped-but-valid checkpoint.

    Raises
    ------
    FileNotFoundError
        Path does not exist.
    ValueError
        Path exists but is not a recognized checkpoint format.
    """
    raise NotImplementedError


def stamp_state_dict(
    state_dict: dict[str, Any],
    meta: CheckpointMeta | None = None,
) -> dict[str, Any]:
    """Inject a meta block into a state dict under :data:`META_KEY`.

    For paths with no native callback-state slot — raw ``torch.save``
    and derived exports. Composer and Lightning should not use this;
    they fill the framework's own slot instead.

    Mutates and returns ``state_dict`` for call-site convenience.
    Re-stamping overwrites any existing block.
    """
    raise NotImplementedError


def export_checkpoint(
    state_dict: dict[str, Any],
    path: str | Path,
    *,
    fmt: ExportFormat,
    meta: CheckpointMeta | None = None,
) -> Path:
    """Write ``state_dict`` to ``path`` in ``fmt``, carrying provenance.

    Provenance embedding is format-specific: ``pt`` gets a top-level
    :data:`META_KEY` entry; ``safetensors`` gets a JSON-serialized
    block in the file header's ``metadata`` field (that field is
    ``dict[str, str]``, so the block is JSON text, not a nested dict).

    Parameters
    ----------
    state_dict : dict
        Tensors to write. Any existing :data:`META_KEY` entry is
        stripped before writing and replaced by ``meta``.
    path : str | Path
        Destination. Parent directories are created.
    fmt : {"pt", "safetensors"}
        Output format. ``safetensors`` requires the ``[safetensors]``
        extra.
    meta : CheckpointMeta, optional
        Provenance to embed. Defaults to :func:`build_checkpoint_meta`.

    Returns
    -------
    Path
        The written path.

    Raises
    ------
    ImportError
        ``fmt="safetensors"`` without the extra installed.
    ValueError
        ``fmt="safetensors"`` with a state dict containing non-tensor
        values — safetensors holds tensors only, unlike torch pickle.
    """
    raise NotImplementedError


def write_first_checkpoint_marker_once() -> None:
    """Touch ``$ASTROLABE_FIRST_CHECKPOINT_MARKER`` on first invocation.

    Mirrors ``_core._write_first_metric_marker_once``. The engine sets
    this when a step's healing policy uses ``until: first_checkpoint``;
    existence closes the healing window at step-failure time.

    Silent no-op when the env var is unset. Best-effort: a failed touch
    degrades the healing bound but must never fail training.
    """
    # TODO(stage3): reads contract.ENV_FIRST_CHECKPOINT_MARKER, which
    # does not exist until the engine PR lands it and this repo
    # re-vendors via tools/vendor-contract.py. Blocked on that.
    raise NotImplementedError


def _read_meta_from_safetensors(path: Path) -> dict[str, Any] | None:
    raise NotImplementedError


def _read_meta_from_torch(path: Path) -> dict[str, Any] | None:
    raise NotImplementedError


def _sniff_format(path: Path) -> ExportFormat | None:
    """Identify checkpoint format by magic bytes, not extension —
    callers name files whatever they like."""
    raise NotImplementedError


BUFFER_NAME = "_astrolabe_meta"


def embed_meta_as_buffer(model: Any, meta: CheckpointMeta | None = None) -> None:
    """Register provenance on ``model`` as a persistent uint8 buffer.

    The escape hatch for frameworks with no checkpoint-dict hook —
    HuggingFace specifically. A registered buffer lands in
    ``state_dict()``, so it rides into every subsequent save (including
    safetensors, which holds tensors only) with no override of the
    framework's save path and no rewrite of the finished file.

    Cost, and it is deliberate: this adds a key that a freshly-built
    model does not have, so ``load_state_dict(..., strict=True)`` on
    such a model raises ``Unexpected key(s)``. HF's ``from_pretrained``
    loads non-strict and only warns. The break is asymmetric — saving
    with the callback and loading without it. See
    :func:`strip_meta_buffer`.

    Idempotent: re-registering replaces the existing buffer so the
    embedded hash tracks the live run.
    """
    raise NotImplementedError


def read_meta_from_buffer(state_dict: dict[str, Any]) -> CheckpointMeta | None:
    """Recover provenance written by :func:`embed_meta_as_buffer`.

    Returns ``None`` when the buffer is absent or undecodable — an
    unstamped checkpoint is not an error.
    """
    raise NotImplementedError


def strip_meta_buffer(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove the provenance buffer so a strict load succeeds.

    For callers who want the weights without the extra key. Returns a
    new dict; the input is not mutated.
    """
    raise NotImplementedError
