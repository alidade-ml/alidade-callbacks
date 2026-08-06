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

import json
import os
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from astrolabe_callbacks import _core, contract

__all__ = [
    "CheckpointMeta",
    "META_KEY",
    "build_checkpoint_meta",
    "read_checkpoint_meta",
    "save_derived_checkpoint",
    "stamp_checkpoint",
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

# Composer stores callback state under the callback's class qualname.
# Renaming AstrolabeComposerCheckpointer orphans every checkpoint
# written by an older version, so the reader pins the old name here.
_COMPOSER_STATE_KEY = "AstrolabeComposerCheckpointer"

# Keyed by path rather than a single boolean: touch() bumps mtime, so
# "once" has to mean once per marker file. One submit only ever sets one
# path, so this is the module-level latch in practice.
_FIRST_CHECKPOINT_MARKERS_WRITTEN: set[str] = set()

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
        # experiment alone is not identity — names repeat across submits.
        return bool(self.submit_id or self.aim_run_hash)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the on-disk form. Omits ``None`` fields so the
        embedded block stays small and forward-compatible."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CheckpointMeta":
        """Parse an embedded block, tolerating unknown keys written by
        a newer callback version."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


# TODO(stage3): replace the two derivation kwargs with `parent: CheckpointMeta
# | None = None` and absorb _derivation_kwargs. aim_run_hash becomes
# `current_run_hash() or parent.aim_run_hash` so a logger-free transform
# inherits the training run instead of writing None. Unreleased surface —
# no compatibility burden.
def build_checkpoint_meta(
    *,
    parent: CheckpointMeta | None = None,
    derived_from: str | None = None,
    derivation_chain_length: int = 0,
) -> CheckpointMeta:
    """Construct provenance for a checkpoint about to be written.

    Reads propagated identity from the environment and opportunistically
    attaches the live Aim run hash if a logger is active in this
    process.

    Parameters
    ----------
    parent : CheckpointMeta, optional
        Provenance of the checkpoint this one was produced from.
        Supplying it makes ``aim_run_hash`` resolve to the live run if
        there is one and the parent's otherwise, so a logger-free
        transform inherits the training run rather than recording none.
        Supersedes ``derived_from`` / ``derivation_chain_length``,
        which Stage 3 removes.
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
    if parent is not None:
        raise NotImplementedError
    submit_id = experiment = version = aim_run_hash = None
    try:
        submit_id = os.environ.get(contract.ENV_SUBMIT_ID) or None
        experiment = os.environ.get(contract.ENV_EXPERIMENT_NAME) or None
        tags = contract.parse_aim_run_tags(os.environ.get(contract.ENV_AIM_RUN_TAGS))
        version = tags.get(contract.TAG_VERSION) or None
        aim_run_hash = _core.current_run_hash()
    except Exception as exc:
        logger.debug("Checkpoint identity lookup failed, stamping unlinked: {}", exc)

    return CheckpointMeta(
        submit_id=submit_id,
        experiment=experiment,
        version=version,
        aim_run_hash=aim_run_hash,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        derived_from=derived_from,
        derivation_chain_length=derivation_chain_length,
    )


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
    if isinstance(checkpoint, dict):
        return _meta_from_block(_extract_meta_block(checkpoint))

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"no such checkpoint: {path}")

    fmt = _sniff_format(path)
    if fmt is None:
        raise ValueError(
            f"{path} is not a recognized checkpoint "
            f"(expected safetensors or torch-pickle magic bytes)"
        )

    reader = _read_meta_from_safetensors if fmt == "safetensors" else _read_meta_from_torch
    return _meta_from_block(reader(path))


def save_derived_checkpoint(
    state_dict: dict[str, Any],
    dest: str | Path,
    parent: str | Path | CheckpointMeta,
    *,
    fmt: ExportFormat | None = None,
) -> Path:
    """Write a transformed checkpoint that stays attributable to the
    training run behind its source.

    For user-owned transforms — surgery, layer extraction, quantization,
    LoRA distillation, anything that turns one checkpoint into another.
    Such code usually runs as a preprocessing step with no Aim logger
    live, which is exactly when provenance would otherwise be lost: the
    output would record no run of its own and evaluate as unlinked.

    Provenance carried forward:

    - ``aim_run_hash`` — **the live run if there is one, otherwise the
      parent's.** A transform script has no live run, so the output
      inherits the run that trained the model it came from. That is what
      keeps the read side a plain lookup: whoever evaluates ``dest``
      reads one field and gets a real training run, however many
      logger-free hops happened in between.
    - ``derived_from`` — the parent's own ``aim_run_hash``. Equal to the
      inherited value in the no-logger case, and genuinely distinct when
      a live run derived from someone else's checkpoint.
    - ``derivation_chain_length`` — parent's, plus one.

    Needs no logger, no Aim, and no network. Copying a string does not
    require a live run.

    Parameters
    ----------
    state_dict : dict
        Transformed tensors to write.
    dest : str | Path
        Destination path. Parent directories are created.
    parent : str | Path | CheckpointMeta
        The checkpoint this was derived from — a path to read
        provenance from, or an already-read block. A parent carrying no
        astrolabe provenance is not an error; the result is simply
        unlinked, same as any unstamped checkpoint.
    fmt : {"pt", "safetensors"}, optional
        Output format. Inferred from ``dest``'s extension when omitted.

    Returns
    -------
    Path
        The written path.

    Raises
    ------
    FileNotFoundError
        ``parent`` is a path that does not exist.
    ValueError
        ``fmt`` omitted and ``dest``'s extension is not recognized, or
        ``fmt="safetensors"`` with non-tensor values in ``state_dict``.
    """
    raise NotImplementedError


def stamp_checkpoint(
    path: str | Path,
    *,
    aim_run_hash: str | None = None,
    submit_id: str | None = None,
    experiment: str | None = None,
    version: str | None = None,
) -> Path:
    """Retrofit astrolabe provenance onto an existing checkpoint, in place.

    For files written before the checkpointer existed, or by code that
    never had it wired. Fields left as ``None`` fall back to the
    propagated environment, so inside an astrolabe step this can be
    called bare and pick up the ambient identity.

    Cost differs sharply by format: ``safetensors`` rewrites the header
    only, so it is fast regardless of model size. Torch pickle has to
    load and re-save, which is O(model size) — acceptable because
    retrofitting is a once-per-file operation, not a hot path.

    Re-stamping overwrites any existing block rather than merging.
    Retrofitting a wrong hash then correcting it is a supported
    sequence.

    Parameters
    ----------
    path : str | Path
        Checkpoint to stamp. Modified in place.
    aim_run_hash : str, optional
        Training run to attribute to. This is the field an eval reads,
        so a checkpoint stamped without it stays unlinked.
    submit_id, experiment, version : str, optional
        Propagated identity. Default to the ambient environment.

    Returns
    -------
    Path
        The stamped path, for chaining.

    Raises
    ------
    FileNotFoundError
        No such file.
    ValueError
        Not a recognized checkpoint format.
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
    state_dict[META_KEY] = (meta or build_checkpoint_meta()).to_dict()
    return state_dict


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
    if fmt not in ("pt", "safetensors"):
        raise ValueError(f"unknown export format {fmt!r}; expected 'pt' or 'safetensors'")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    block = (meta or build_checkpoint_meta()).to_dict()
    tensors = {k: v for k, v in state_dict.items() if k != META_KEY}

    if fmt == "pt":
        import torch

        torch.save({**tensors, META_KEY: block}, out)
        return out

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors export needs the extra: pip install 'astrolabe-callbacks[safetensors]'"
        ) from exc

    import torch

    non_tensors = sorted(k for k, v in tensors.items() if not isinstance(v, torch.Tensor))
    if non_tensors:
        raise ValueError(
            f"safetensors holds tensors only; non-tensor keys: {non_tensors}. "
            f"Use fmt='pt' to keep them."
        )
    save_file(tensors, str(out), metadata={META_KEY: json.dumps(block)})
    return out


def write_first_checkpoint_marker_once() -> None:
    """Touch ``$ASTROLABE_FIRST_CHECKPOINT_MARKER`` on first invocation.

    Mirrors ``_core._write_first_metric_marker_once``. The engine sets
    this when a step's healing policy uses ``until: first_checkpoint``;
    existence closes the healing window at step-failure time.

    Silent no-op when the env var is unset. Best-effort: a failed touch
    degrades the healing bound but must never fail training.
    """
    path = os.environ.get(contract.ENV_FIRST_CHECKPOINT_MARKER)
    if not path or path in _FIRST_CHECKPOINT_MARKERS_WRITTEN:
        return
    _FIRST_CHECKPOINT_MARKERS_WRITTEN.add(path)
    try:
        Path(os.path.expanduser(path)).touch()
    except Exception as exc:
        logger.debug("first-checkpoint marker touch failed ({}): {}", path, exc)


# TODO(stage3): fold into build_checkpoint_meta(parent=...) and delete. The
# `not parent.aim_run_hash -> {}` guard is the bug: it drops linkage whenever a
# hop had no logger. Same-run resume must still read as continuation.
def _derivation_kwargs(parent: CheckpointMeta | None) -> dict[str, Any]:
    """``build_checkpoint_meta`` kwargs linking a checkpoint to the one
    its run resumed from.

    Empty when there is no parent, when the parent was never linked, or
    when the parent is this same Aim run — resuming into the run that
    wrote the parent is continuation, not derivation.
    """
    if parent is None or not parent.aim_run_hash:
        return {}
    if parent.aim_run_hash == _core.current_run_hash():
        return {}
    return {
        "derived_from": parent.aim_run_hash,
        "derivation_chain_length": parent.derivation_chain_length + 1,
    }


def _meta_from_block(block: Any) -> CheckpointMeta | None:
    if block is None:
        return None
    if not isinstance(block, dict):
        # The buffer mechanism writes a uint8 tensor under the same key.
        return read_meta_from_buffer({BUFFER_NAME: block})
    try:
        return CheckpointMeta.from_dict(block)
    except Exception as exc:
        logger.debug("Unusable astrolabe meta block, treating as absent: {}", exc)
        return None


def _extract_meta_block(obj: dict[str, Any]) -> Any:
    """Find the meta block wherever the writing mechanism put it."""
    block = obj.get(META_KEY)
    if block is not None:
        return block
    # Composer keys each callback's state by class qualname rather than
    # writing to the top level, so its block lives one nest down.
    state = obj.get("state")
    callbacks = state.get("callbacks") if isinstance(state, dict) else None
    if isinstance(callbacks, dict):
        return callbacks.get(_COMPOSER_STATE_KEY)
    return None


def _read_meta_from_safetensors(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as fh:
            header_len = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(header_len).decode("utf-8"))
            block = (header.get("__metadata__") or {}).get(META_KEY)
            if block:
                return json.loads(block)
            return _read_buffer_tensor(fh, header, data_start=8 + header_len)
    except Exception as exc:
        logger.debug("Could not read safetensors header of {}: {}", path, exc)
        return None


def _read_buffer_tensor(fh: Any, header: dict[str, Any], *, data_start: int) -> dict[str, Any] | None:
    """Decode the provenance buffer out of a safetensors *tensor*.

    HuggingFace writes the file itself and hardcodes its own
    ``__metadata__``, so mechanism 3's block arrives as a uint8 tensor
    among the weights rather than in the header. Without this branch
    ``read_checkpoint_meta`` returns ``None`` for every HF checkpoint —
    the artifact carries provenance the public reader cannot see.

    https://github.com/huggingface/safetensors#format
    """
    entry = header.get(BUFFER_NAME)
    if not isinstance(entry, dict) or entry.get("dtype") != "U8":
        return None
    start, end = entry["data_offsets"]
    fh.seek(data_start + start)
    raw = json.loads(fh.read(end - start).decode("utf-8"))
    return raw if isinstance(raw, dict) else None


def _read_meta_from_torch(path: Path) -> dict[str, Any] | None:
    import torch

    try:
        obj = torch.load(path, map_location="meta", weights_only=True)
    except Exception:
        # Framework checkpoints carry non-tensor state (optimizer configs,
        # RNG objects, numpy scalars) that weights_only refuses to unpickle.
        try:
            obj = torch.load(path, map_location="meta", weights_only=False)
        except Exception as exc:
            logger.debug("Could not load {}: {}", path, exc)
            return None
    return _extract_meta_block(obj) if isinstance(obj, dict) else None


def _sniff_format(path: Path) -> ExportFormat | None:
    """Identify checkpoint format by magic bytes, not extension —
    callers name files whatever they like."""
    with path.open("rb") as fh:
        head = fh.read(9)
    # torch.save writes a zip archive (modern) or a protocol-2+ pickle
    # stream (legacy `_use_new_zipfile_serialization=False`).
    if head[:4] == b"PK\x03\x04" or head[:1] == b"\x80":
        return "pt"
    # safetensors: u64 little-endian header length, then that many bytes
    # of JSON. https://github.com/huggingface/safetensors#format
    if len(head) == 9 and head[8:9] == b"{":
        header_len = int.from_bytes(head[:8], "little")
        if 0 < header_len <= path.stat().st_size - 8:
            return "safetensors"
    return None


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
    import torch

    payload = json.dumps((meta or build_checkpoint_meta()).to_dict()).encode("utf-8")
    model.register_buffer(
        BUFFER_NAME,
        torch.frombuffer(bytearray(payload), dtype=torch.uint8),
        persistent=True,
    )


def read_meta_from_buffer(state_dict: dict[str, Any]) -> CheckpointMeta | None:
    """Recover provenance written by :func:`embed_meta_as_buffer`.

    Returns ``None`` when the buffer is absent or undecodable — an
    unstamped checkpoint is not an error.
    """
    buffer = state_dict.get(BUFFER_NAME)
    if buffer is None:
        return None
    try:
        raw = json.loads(bytes(buffer.tolist()).decode("utf-8"))
    except Exception as exc:
        logger.debug("Undecodable astrolabe provenance buffer: {}", exc)
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return CheckpointMeta.from_dict(raw)
    except Exception as exc:
        logger.debug("Unusable astrolabe provenance buffer: {}", exc)
        return None


def strip_meta_buffer(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove the provenance buffer so a strict load succeeds.

    For callers who want the weights without the extra key. Returns a
    new dict; the input is not mutated.
    """
    return {k: v for k, v in state_dict.items() if k != BUFFER_NAME}
