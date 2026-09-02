"""Checkpoint-provenance exerciser — runs inside the ``client`` container.

Companion to ``driver.py``. Where that one exercises metric streaming,
this one drives a **real** framework training loop with a checkpointer
attached, lets the framework write its own checkpoints, and reports what
actually landed on disk.

Four framework paths, one per embedding mechanism:

- ``composer`` — Composer's ``CheckpointSaver`` writes the file;
  ``AstrolabeComposerCheckpointer.state_dict()`` fills Composer's
  callback-state slot.
- ``lightning`` — Lightning's ``ModelCheckpoint`` writes the file;
  ``on_save_checkpoint`` mutates the dict in place.
- ``pytorch`` — no framework; ``save_checkpoint`` writes an explicit
  top-level ``_astrolabe_meta`` key.
- ``hf`` — a real ``BertForSequenceClassification`` (a genuine
  ``PreTrainedModel``, not an ``nn.Module`` stand-in, so
  ``save_pretrained`` / ``from_pretrained`` / sharding are the real
  code paths) with the provenance buffer riding into ``state_dict()``.

Why real frameworks rather than calling the hooks by hand: the claim
under test is "the framework serializes and replays our block", and a
hand-called hook proves nothing about the framework.

Reporting contract: one ``ALIDADE_CKPT_PROBE=<json>`` line on stdout.
The driver only *observes* — every judgement lives in the scenario. The
checkpoint files themselves are left in ``config.workdir`` so the
harness can ``docker cp`` them out for host-side reads with the real
libraries.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


__all__ = [
    "CheckpointDriverConfig",
    "CheckpointDriverResult",
    "CheckpointFramework",
    "PROBE_PREFIX",
    "config_to_env",
    "run_checkpoint_driver",
    "main",
]


CheckpointFramework = Literal["composer", "lightning", "pytorch", "hf"]

PROBE_PREFIX = "ALIDADE_CKPT_PROBE="

# Both the HF trainer and the HF-only scenarios build this model, and a
# from_pretrained on the host has to reconstruct the same architecture.
HF_MODEL_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 32,
    "num_labels": 2,
}


@dataclass(frozen=True)
class CheckpointDriverConfig:
    """See module docstring. ``driver_flags`` carry the fault injections.

    Recognized ``driver_flags``:

    ``TESTBED_CORRUPT_PARENT_META``
        Overwrite the parent checkpoint's provenance block with a
        non-decodable value before resuming from it.
    ``TESTBED_PROBE_MARKER_LATCH``
        After training, unlink the marker and call
        ``write_first_checkpoint_marker_once`` again, reporting whether
        it came back. Directly probes the once-per-path latch.
    ``TESTBED_HF_SHARD_SAVE``
        After training, additionally ``save_pretrained`` with a tiny
        ``max_shard_size`` so HF's sharded-save path runs for real.
    ``TESTBED_DERIVE_CHAIN``
        After training closes its run, derive two checkpoints in
        sequence with no logger live — the shape a preprocessing script
        (surgery, quantization) actually has. Reports the run live at
        derive time plus each hop's provenance.
    ``TESTBED_HF_LOAD_PROBE``
        Replay the two documented load paths against the written
        checkpoint — ``from_pretrained`` (non-strict) and a manual
        ``load_state_dict(strict=True)`` with and without
        ``strip_meta_buffer`` — and report what each did. Runs in the
        container because the load has to use the same transformers
        version that wrote the file.
    """

    framework: CheckpointFramework
    aim_url: str
    experiment_name: str
    run_name: str
    submit_id: str
    version: str
    steps: int
    save_every: int
    with_logger: bool
    embed_in_weights: bool
    export_formats: list[str]
    new_metrics_at: list[int]
    workdir: str
    marker_path: str | None
    resume_from: str | None
    stats_jsonl_container_path: str
    driver_flags: dict[str, str]

    @classmethod
    def from_env(cls) -> "CheckpointDriverConfig":
        def _req(key: str) -> str:
            value = os.environ.get(key)
            if value is None:
                print(f"missing required env var {key}", file=sys.stderr)
                raise SystemExit(2)
            return value

        return cls(
            framework=_req("TESTBED_CKPT_FRAMEWORK"),  # type: ignore[arg-type]
            aim_url=_req("TESTBED_CKPT_AIM_URL"),
            experiment_name=_req("TESTBED_CKPT_EXPERIMENT_NAME"),
            run_name=_req("TESTBED_CKPT_RUN_NAME"),
            submit_id=os.environ.get("ALIDADE_SUBMIT_ID", ""),
            version=os.environ.get("TESTBED_CKPT_VERSION", ""),
            steps=int(_req("TESTBED_CKPT_STEPS")),
            save_every=int(os.environ.get("TESTBED_CKPT_SAVE_EVERY", "1")),
            with_logger=os.environ.get("TESTBED_CKPT_WITH_LOGGER", "1") == "1",
            embed_in_weights=os.environ.get("TESTBED_CKPT_EMBED_IN_WEIGHTS", "1") == "1",
            export_formats=json.loads(os.environ.get("TESTBED_CKPT_EXPORT_FORMATS", "[]")),
            new_metrics_at=json.loads(os.environ.get("TESTBED_CKPT_NEW_METRICS_AT", "[]")),
            workdir=_req("TESTBED_CKPT_WORKDIR"),
            marker_path=os.environ.get("ALIDADE_FIRST_CHECKPOINT_MARKER") or None,
            resume_from=os.environ.get("TESTBED_CKPT_RESUME_FROM") or None,
            stats_jsonl_container_path=_req("TESTBED_CKPT_STATS_JSONL_PATH"),
            driver_flags=json.loads(os.environ.get("TESTBED_CKPT_DRIVER_FLAGS", "{}")),
        )


@dataclass(frozen=True)
class CheckpointDriverResult:
    exit_code: int
    probe: dict[str, Any]
    stdout: str
    stderr: str
    stats_events: list[dict]
    host_workdir: Path

    def checkpoints(self) -> list[dict[str, Any]]:
        """Per-checkpoint observations, in the order they were written."""
        return self.probe.get("checkpoints", [])

    def primary(self) -> dict[str, Any]:
        """The first primary checkpoint. Raises if none were written —
        a scenario asserting on provenance has nothing to say otherwise."""
        primaries = [c for c in self.checkpoints() if c["role"] == "primary"]
        if not primaries:
            raise AssertionError(
                f"driver wrote no primary checkpoint (probe: {self.probe}, stderr: {self.stderr})"
            )
        return primaries[0]

    def meta_of(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """The provenance block recovered from a checkpoint. Raises when
        absent — every path under test is supposed to stamp."""
        meta = checkpoint.get("meta")
        if meta is None:
            raise AssertionError(
                f"checkpoint {checkpoint['path']!r} carries no astrolabe block "
                f"(keys seen: {checkpoint.get('top_level_keys')})"
            )
        return meta


def config_to_env(config: CheckpointDriverConfig) -> dict[str, str]:
    """Serialize ``config`` for ``compose.exec_in(env=...)``."""
    env = {
        "TESTBED_CKPT_FRAMEWORK": config.framework,
        "TESTBED_CKPT_AIM_URL": config.aim_url,
        "TESTBED_CKPT_EXPERIMENT_NAME": config.experiment_name,
        "TESTBED_CKPT_RUN_NAME": config.run_name,
        "TESTBED_CKPT_VERSION": config.version,
        "TESTBED_CKPT_STEPS": str(config.steps),
        "TESTBED_CKPT_SAVE_EVERY": str(config.save_every),
        "TESTBED_CKPT_WITH_LOGGER": "1" if config.with_logger else "0",
        "TESTBED_CKPT_EMBED_IN_WEIGHTS": "1" if config.embed_in_weights else "0",
        "TESTBED_CKPT_EXPORT_FORMATS": json.dumps(config.export_formats),
        "TESTBED_CKPT_NEW_METRICS_AT": json.dumps(config.new_metrics_at),
        "TESTBED_CKPT_WORKDIR": config.workdir,
        "TESTBED_CKPT_STATS_JSONL_PATH": config.stats_jsonl_container_path,
        "TESTBED_CKPT_DRIVER_FLAGS": json.dumps(config.driver_flags),
        # The callback library's own env contract — this is the identity
        # the checkpointers are supposed to propagate into the artifact.
        "ALIDADE_AIM_URL": config.aim_url,
        "ALIDADE_EXPERIMENT_NAME": config.experiment_name,
        "ALIDADE_SUBMIT_ID": config.submit_id,
        "ALIDADE_CALLBACK_STATS_PATH": config.stats_jsonl_container_path,
    }
    if config.version:
        env["AIM_RUN_TAGS"] = f"astrolabe.version={config.version}"
    if config.marker_path:
        env["ALIDADE_FIRST_CHECKPOINT_MARKER"] = config.marker_path
    if config.resume_from:
        env["TESTBED_CKPT_RESUME_FROM"] = config.resume_from
    return env


# ---------------------------------------------------------------------------
# Inspection — pure observation, no judgement
# ---------------------------------------------------------------------------


def _inspect(path: Path, role: str) -> dict[str, Any]:
    """Structural facts about one written checkpoint.

    ``meta`` comes from the library's own public reader. Everything else
    is read independently (raw ``torch.load`` / raw safetensors header)
    so a scenario can tell "the framework put our block where we think
    it did" apart from "our reader found something somewhere".
    """
    from alidade_callbacks.checkpoint import (
        BUFFER_NAME,
        read_checkpoint_meta,
        read_meta_from_buffer,
    )

    facts: dict[str, Any] = {
        "path": str(path),
        "role": role,
        "exists": path.exists(),
        "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
        "meta": None,
        "buffer_meta": None,
        "top_level_keys": None,
        "composer_callback_keys": None,
        "composer_block": None,
        "tensor_names": None,
        "safetensors_header_metadata": None,
    }
    if not path.exists():
        return facts

    try:
        meta = read_checkpoint_meta(path)
        facts["meta"] = meta.to_dict() if meta is not None else None
    except Exception as exc:
        facts["meta_error"] = repr(exc)

    if path.suffix == ".safetensors":
        facts.update(_inspect_safetensors(path))
        names = facts.get("tensor_names") or []
        if BUFFER_NAME in names:
            from safetensors.torch import load_file

            facts["buffer_meta"] = _meta_or_none(
                read_meta_from_buffer(load_file(str(path)))
            )
        return facts

    obj = _torch_load(path)
    if not isinstance(obj, dict):
        return facts
    facts["top_level_keys"] = sorted(obj.keys())
    facts["buffer_meta"] = _meta_or_none(read_meta_from_buffer(obj))
    state = obj.get("state")
    callbacks = state.get("callbacks") if isinstance(state, dict) else None
    if isinstance(callbacks, dict):
        facts["composer_callback_keys"] = sorted(callbacks.keys())
        block = callbacks.get("AstrolabeComposerCheckpointer")
        facts["composer_block"] = block if isinstance(block, dict) else None
    return facts


def _inspect_safetensors(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len).decode("utf-8"))
    return {
        "tensor_names": sorted(k for k in header if k != "__metadata__"),
        "safetensors_header_metadata": header.get("__metadata__"),
    }


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Framework checkpoints carry optimizer / RNG / numpy state that
        # weights_only refuses to unpickle.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None


def _meta_or_none(meta: Any) -> dict[str, Any] | None:
    return meta.to_dict() if meta is not None else None


GARBAGE_BLOCK = "not-a-provenance-block"


def _corrupt_meta_block(path: Path) -> None:
    """Replace a checkpoint's provenance with something undecodable.

    Writes a bare string where a block is expected — the shape a
    truncated copy or a hand-edited checkpoint produces, and the one the
    reader has to survive.
    """
    if path.is_dir():
        _corrupt_hf_checkpoint(path)
        return

    import torch

    from alidade_callbacks.checkpoint import META_KEY

    obj = _torch_load(path)
    if not isinstance(obj, dict):
        raise ValueError(f"cannot corrupt non-dict checkpoint {path}")
    state = obj.get("state")
    callbacks = state.get("callbacks") if isinstance(state, dict) else None
    if isinstance(callbacks, dict) and "AstrolabeComposerCheckpointer" in callbacks:
        callbacks["AstrolabeComposerCheckpointer"] = GARBAGE_BLOCK
    else:
        obj[META_KEY] = GARBAGE_BLOCK
    torch.save(obj, path)


def _corrupt_hf_checkpoint(directory: Path) -> None:
    """Corrupt both places an HF checkpoint carries provenance.

    ``trainer_state.json`` is what the resume path reads; the weights
    buffer is what an eval script reads. A resume has to survive either
    one being garbage.
    """
    import torch
    from safetensors.torch import load_file, save_file

    from alidade_callbacks.checkpoint import BUFFER_NAME

    state_path = directory / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        stored = (state.get("stateful_callbacks") or {}).get("AstrolabeHFCheckpointer")
        if isinstance(stored, dict):
            stored["attributes"] = {"astrolabe_provenance": GARBAGE_BLOCK}
            state_path.write_text(json.dumps(state))

    weights_path = directory / "model.safetensors"
    if weights_path.exists():
        tensors = load_file(str(weights_path))
        if BUFFER_NAME in tensors:
            tensors[BUFFER_NAME] = torch.frombuffer(
                bytearray(b"\xff\xfe not json \xff"), dtype=torch.uint8
            )
        save_file(tensors, str(weights_path), metadata={"format": "pt"})


# ---------------------------------------------------------------------------
# Framework drivers
# ---------------------------------------------------------------------------


def phase(label: str) -> None:
    """Mark a stage boundary on stderr, flushed.

    The driver runs inside ``docker compose exec`` under a 900s cap in
    the harness. When it stalls, the harness reports only that the cap
    expired — the captured output ends wherever the driver last wrote,
    which is HuggingFace's own training log, so every stall looks
    identical no matter which stage it happened in. These markers make
    the last line before silence name the stage.

    Flushed because a stalled process never drains its buffer, and an
    unflushed marker is exactly the one you needed. stderr because
    stdout carries the probe payload the scenarios parse.

    See astrolabe plans/tickets/TESTBED-1.md.
    """
    print(f"[driver] {label}", file=sys.stderr, flush=True)


def run_checkpoint_driver(config: CheckpointDriverConfig) -> dict[str, Any]:
    """Execute one invocation. Returns the probe dict."""
    phase(f"start framework={config.framework} steps={config.steps} "
          f"save_every={config.save_every} embed={config.embed_in_weights}")
    _seed_all()
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    marker_existed_at_start = _prepare_marker(config)
    if config.resume_from and config.driver_flags.get("TESTBED_CORRUPT_PARENT_META"):
        _corrupt_meta_block(Path(config.resume_from))

    phase("training: enter")
    written = {
        "composer": _run_composer,
        "lightning": _run_lightning,
        "pytorch": _run_pytorch,
        "hf": _run_hf,
    }[config.framework](config, workdir)
    phase(f"training: done, {len(written)} checkpoint(s) written")

    from alidade_callbacks.checkpoint import write_first_checkpoint_marker_once

    probe: dict[str, Any] = {
        "framework": config.framework,
        "workdir": config.workdir,
        "run_hash": _resolve_run_hash(config),
        "checkpoints": _inspect_all(written),
        "marker": _marker_report(config, marker_existed_at_start),
    }
    if config.marker_path and config.driver_flags.get("TESTBED_PROBE_MARKER_LATCH"):
        Path(config.marker_path).unlink(missing_ok=True)
        write_first_checkpoint_marker_once()
        probe["marker"]["recreated_after_unlink"] = Path(config.marker_path).exists()
    if config.driver_flags.get("TESTBED_DERIVE_CHAIN"):
        probe["derivation"] = _derive_chain_probe(written, workdir)
    if config.driver_flags.get("TESTBED_HF_LOAD_PROBE"):
        probe["hf_load"] = _hf_load_probe(written)
    if config.driver_flags.get("TESTBED_HF_SHARD_SAVE"):
        probe["hf_shard"] = _hf_shard_probe(workdir / "hf" / "sharded")
    phase("probe: done")
    return probe


def _inspect_all(written: list[tuple[Path, str]]) -> list[dict[str, Any]]:
    """Inspect each checkpoint, naming the file before reading it.

    Reading a checkpoint back is one of the two stages a stall could be
    hiding in, and it happens once per file — so the marker carries the
    filename, not just the stage.
    """
    out = []
    for i, (path, role) in enumerate(written, start=1):
        phase(f"inspect {i}/{len(written)} {role} {path.name}")
        out.append(_inspect(path, role))
    return out


def _derive_chain_probe(
    written: list[tuple[Path, str]], workdir: Path
) -> dict[str, Any]:
    """Two logger-free transforms in sequence.

    The precondition is half the value: real transform code runs after
    training has closed its run, so nothing is registered. Unit tests
    reach that state by monkeypatching the registry — here it has to be
    genuinely true, which is why ``live_run_at_derive`` is reported
    rather than assumed.
    """
    import torch

    from alidade_callbacks import _core
    from alidade_callbacks.checkpoint import save_derived_checkpoint

    report: dict[str, Any] = {
        "live_run_at_derive": _core.current_run_hash(),
        "hops": [],
    }
    primaries = [path for path, role in written if role == "primary"]
    if not primaries:
        report["error"] = "driver wrote no primary checkpoint to derive from"
        return report

    parent = primaries[0]
    for hop in (1, 2):
        dest = workdir / "derived" / f"hop{hop}.pt"
        save_derived_checkpoint({"w": torch.zeros(2)}, dest, parent)
        report["hops"].append(_inspect(dest, f"derived-hop{hop}"))
        parent = dest
    return report


def _hf_load_probe(written: list[tuple[Path, str]]) -> dict[str, Any]:
    """Replay both documented load paths against the written checkpoint.

    ``from_pretrained`` is the supported one and must stay a warning;
    the manual strict load is the documented footgun and must stay a
    hard failure that :func:`strip_meta_buffer` clears.
    """
    from safetensors.torch import load_file
    from transformers import AutoModelForSequenceClassification

    from alidade_callbacks.checkpoint import strip_meta_buffer

    directories = sorted(
        {path.parent for path, role in written if path.suffix == ".safetensors"},
        key=lambda p: str(p),
    )
    if not directories:
        return {"error": "no safetensors checkpoint written"}
    checkpoint_dir = directories[0]
    report: dict[str, Any] = {"checkpoint_dir": str(checkpoint_dir)}

    try:
        model, info = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_dir), output_loading_info=True
        )
        report.update(
            {
                "from_pretrained_ok": model is not None,
                "from_pretrained_unexpected_keys": list(info.get("unexpected_keys", [])),
                "from_pretrained_missing_keys": list(info.get("missing_keys", [])),
                "from_pretrained_error": None,
            }
        )
    except Exception as exc:
        report.update({"from_pretrained_ok": False, "from_pretrained_error": repr(exc)})

    state_dict = load_file(str(checkpoint_dir / "model.safetensors"))
    report["strict_load_error"] = _strict_load_error(state_dict)
    report["strict_load_after_strip_error"] = _strict_load_error(
        strip_meta_buffer(state_dict)
    )
    return report


def _strict_load_error(state_dict: dict[str, Any]) -> str | None:
    try:
        _hf_model().load_state_dict(state_dict, strict=True)
    except Exception as exc:
        return str(exc)
    return None


def _hf_shard_probe(sharded_dir: Path) -> dict[str, Any]:
    """What HF's sharded save did with the provenance buffer."""
    from alidade_callbacks.checkpoint import BUFFER_NAME, read_meta_from_buffer

    index_path = sharded_dir / "model.safetensors.index.json"
    report: dict[str, Any] = {
        "dir": str(sharded_dir),
        "shard_files": sorted(p.name for p in sharded_dir.glob("*.safetensors")),
        "index_exists": index_path.exists(),
    }
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        report["weight_map_keys"] = sorted(weight_map)
        report["buffer_shard"] = weight_map.get(BUFFER_NAME)

    merged: dict[str, Any] = {}
    for shard in sorted(sharded_dir.glob("*.safetensors")):
        from safetensors.torch import load_file

        merged.update(load_file(str(shard)))
    report["merged_meta"] = _meta_or_none(read_meta_from_buffer(merged))
    return report


def _prepare_marker(config: CheckpointDriverConfig) -> bool:
    """Clear any leftover marker so touched-vs-not is unambiguous.

    The parent directory is deliberately NOT created: the unwritable-path
    scenario points the marker into a directory that does not exist, and
    creating it here would defeat the test.
    """
    if not config.marker_path:
        return False
    path = Path(config.marker_path)
    existed = path.exists()
    if existed:
        path.unlink()
    return existed


def _marker_report(config: CheckpointDriverConfig, existed_at_start: bool) -> dict[str, Any]:
    path = Path(config.marker_path) if config.marker_path else None
    exists = bool(path and path.exists())
    return {
        "path": config.marker_path,
        "existed_at_start": existed_at_start,
        "exists_at_end": exists,
        "mtime_ns": path.stat().st_mtime_ns if exists else None,
    }


def _resolve_run_hash(config: CheckpointDriverConfig) -> str | None:
    """The hash Aim actually minted, read back from the server.

    Deliberately not taken from the callback object: the scenario's whole
    question is whether the embedded hash matches the run that exists.
    """
    if not config.with_logger:
        return None
    from tests.testbed.harness.driver import _find_recent_framework_run

    return _find_recent_framework_run(
        config.aim_url, config.experiment_name, config.run_name
    )


def _seed_all(seed: int = 0) -> None:
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)


def _tensor_dataset(steps: int):
    import torch
    from torch.utils.data import TensorDataset

    return TensorDataset(torch.randn(max(steps, 1) * 2, 4), torch.randn(max(steps, 1) * 2, 1))


def _run_pytorch(
    config: CheckpointDriverConfig, workdir: Path
) -> list[tuple[Path, str]]:
    """Raw-PyTorch path: hand-rolled loop calling ``save_checkpoint``."""
    import torch
    import torch.nn as nn

    from alidade_callbacks.pytorch import AstrolabeRun, save_checkpoint

    model = nn.Linear(4, 1)
    if config.resume_from:
        parent = torch.load(config.resume_from, map_location="cpu", weights_only=False)
        # The researcher's resume is exactly this: load the saved dict and
        # keep training from it. The provenance key rides along.
        model.load_state_dict(
            {k: v for k, v in parent.items() if k in model.state_dict()}
        )

    written: list[tuple[Path, str]] = []

    def _save(step: int) -> None:
        state = dict(model.state_dict())
        if config.resume_from:
            # Preserve whatever the parent file carried so the resume path
            # sees what a real ``torch.load``-then-``save`` cycle sees.
            state.update(
                {
                    k: v
                    for k, v in torch.load(
                        config.resume_from, map_location="cpu", weights_only=False
                    ).items()
                    if k not in state
                }
            )
        path = workdir / f"ckpt-{step}.pt"
        save_checkpoint(state, str(path), export_formats=config.export_formats)
        written.append((path, "primary"))
        written.extend(
            (path.with_suffix(f".{fmt}"), "export") for fmt in config.export_formats
        )

    run = (
        AstrolabeRun(
            aim_url=config.aim_url,
            experiment_name=config.experiment_name,
            run_name=config.run_name,
        )
        if config.with_logger
        else None
    )
    if run is not None:
        run.__enter__()
    try:
        for step in range(config.steps):
            if run is not None:
                run.log("metric_0", float(step), step=step)
                if step in config.new_metrics_at:
                    run.log(f"metric_new_step{step}", float(step), step=step)
            if step % config.save_every == 0:
                _save(step)
    finally:
        if run is not None:
            run.__exit__(None, None, None)
    return written


def _run_composer(
    config: CheckpointDriverConfig, workdir: Path
) -> list[tuple[Path, str]]:
    import torch.nn as nn
    from composer import Trainer
    from composer.core import Callback
    from composer.models import ComposerModel
    from torch.utils.data import DataLoader

    from alidade_callbacks.composer import (
        AstrolabeComposerCheckpointer,
        AstrolabeComposerLogger,
    )

    class TinyComposer(ComposerModel):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, batch):
            x, _ = batch
            return self.lin(x)

        def loss(self, outputs, batch):
            _, y = batch
            return ((outputs - y) ** 2).mean()

    new_metrics_at = set(config.new_metrics_at)

    class MetricEmitter(Callback):
        """Emits ``metric_0`` every batch, plus a previously-unseen name
        at each chosen batch — the only thing that makes
        ``maybe_finalize_schema`` do work."""

        def __init__(self):
            self._step = 0

        def batch_end(self, state, logger_obj):
            metrics = {"metric_0": float(self._step)}
            if self._step in new_metrics_at:
                metrics[f"metric_new_step{self._step}"] = float(self._step)
            logger_obj.log_metrics(metrics)
            self._step += 1

    save_dir = workdir / "composer"
    checkpointer = AstrolabeComposerCheckpointer(
        export_formats=config.export_formats,
        export_dir=str(save_dir),
    )
    loggers = (
        [
            AstrolabeComposerLogger(
                aim_url=config.aim_url, experiment_name=config.experiment_name
            )
        ]
        if config.with_logger
        else []
    )
    # Resuming means picking up where the parent stopped, so the child's
    # budget has to exceed the parent's or fit() returns immediately.
    duration = config.steps * (2 if config.resume_from else 1)
    trainer = Trainer(
        model=TinyComposer(),
        train_dataloader=DataLoader(_tensor_dataset(duration), batch_size=1),
        max_duration=f"{duration}ba",
        run_name=config.run_name,
        loggers=loggers,
        callbacks=[MetricEmitter(), checkpointer],
        save_folder=str(save_dir),
        save_filename="ba{batch}.pt",
        save_interval=f"{config.save_every}ba",
        save_num_checkpoints_to_keep=-1,
        save_overwrite=True,
        load_path=config.resume_from,
        load_weights_only=False,
        device="cpu",
        progress_bar=False,
    )
    trainer.fit()
    return _collect(save_dir, primary_suffixes=(".pt",), export_marker="-derived.")


def _run_lightning(
    config: CheckpointDriverConfig, workdir: Path
) -> list[tuple[Path, str]]:
    import lightning
    import torch
    import torch.nn as nn
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    from alidade_callbacks.lightning import (
        AstrolabeLightningCheckpointer,
        AstrolabeLightningLogger,
    )

    new_metrics_at = set(config.new_metrics_at)
    co_callback_key = "co_attached_callback_state"

    class TinyLightning(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 1)
            self._step = 0

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = ((self.lin(x) - y) ** 2).mean()
            self.log("metric_0", float(self._step), on_step=True, on_epoch=False)
            if self._step in new_metrics_at:
                self.log(
                    f"metric_new_step{self._step}",
                    float(self._step),
                    on_step=True,
                    on_epoch=False,
                )
            self._step += 1
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    class CoAttachedMutator(lightning.pytorch.callbacks.Callback):
        """A second callback writing its own top-level key, the way a
        user's own bookkeeping callback would."""

        def __init__(self, tag: str):
            self._tag = tag

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            checkpoint.setdefault(co_callback_key, []).append(self._tag)

    save_dir = workdir / "lightning"
    callbacks: list[Any] = [CoAttachedMutator("before")]
    if config.with_logger:
        callbacks.append(
            AstrolabeLightningLogger(
                aim_url=config.aim_url,
                experiment_name=config.experiment_name,
                run_name=config.run_name,
            )
        )
    callbacks.extend(
        [
            AstrolabeLightningCheckpointer(
                export_formats=config.export_formats, export_dir=str(save_dir)
            ),
            CoAttachedMutator("after"),
            ModelCheckpoint(
                dirpath=str(save_dir),
                filename="step{step}",
                every_n_train_steps=config.save_every,
                save_top_k=-1,
            ),
        ]
    )

    steps = config.steps
    trainer = lightning.Trainer(
        callbacks=callbacks,
        # A resumed run restores the parent's epoch counter, so a
        # single-epoch budget is already spent and fit() returns without
        # training. The child needs an epoch of its own.
        max_epochs=2 if config.resume_from else 1,
        limit_train_batches=steps,
        enable_progress_bar=False,
        accelerator="cpu",
        logger=False,
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )
    trainer.fit(
        TinyLightning(),
        train_dataloaders=DataLoader(_tensor_dataset(steps), batch_size=1),
        ckpt_path=config.resume_from,
    )
    return _collect(save_dir, primary_suffixes=(".ckpt",), export_marker="-derived.")


def _hf_model():
    from transformers import BertConfig, BertForSequenceClassification

    return BertForSequenceClassification(BertConfig(**HF_MODEL_CONFIG))


def _run_hf(config: CheckpointDriverConfig, workdir: Path) -> list[tuple[Path, str]]:
    import torch
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from alidade_callbacks.huggingface import (
        AstrolabeHFCheckpointer,
        AstrolabeHFTrainerCallback,
    )

    seq_len = 8
    new_metrics_at = set(config.new_metrics_at)

    class ToyDataset(Dataset):
        def __init__(self, n: int):
            self.input_ids = torch.randint(0, HF_MODEL_CONFIG["vocab_size"], (n, seq_len))
            self.labels = torch.randint(0, HF_MODEL_CONFIG["num_labels"], (n,))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {
                "input_ids": self.input_ids[i],
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels": self.labels[i],
            }

    class MetricEmitter(TrainerCallback):
        """Mutates HF's logs dict before Astrolabe's on_log sees it —
        callback order in the list is what makes that work."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None or any(k.startswith("eval_") for k in logs):
                return
            logs["metric_0"] = float(state.global_step)
            if state.global_step in new_metrics_at:
                logs[f"metric_new_step{state.global_step}"] = float(state.global_step)

    save_dir = workdir / "hf"
    callbacks: list[Any] = [MetricEmitter()]
    if config.with_logger:
        callbacks.append(
            AstrolabeHFTrainerCallback(
                aim_url=config.aim_url,
                experiment_name=config.experiment_name,
                run_name=config.run_name,
            )
        )
    callbacks.append(
        AstrolabeHFCheckpointer(
            embed_in_weights=config.embed_in_weights,
            export_formats=config.export_formats,
        )
    )

    steps = config.steps * (2 if config.resume_from else 1)
    trainer = Trainer(
        model=_hf_model(),
        args=TrainingArguments(
            output_dir=str(save_dir),
            max_steps=steps,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_steps=config.save_every,
            save_strategy="steps",
            save_total_limit=None,
            disable_tqdm=True,
            report_to=[],
            use_cpu=True,
        ),
        train_dataset=ToyDataset(steps * 2 + 4),
        callbacks=callbacks,
    )
    phase("hf: trainer.train enter")
    trainer.train(resume_from_checkpoint=config.resume_from)
    phase("hf: trainer.train returned")

    written = _collect(save_dir, primary_suffixes=(".safetensors",), export_marker="derived.")
    if config.driver_flags.get("TESTBED_HF_SHARD_SAVE"):
        phase("hf: sharded save_pretrained enter")
        sharded = save_dir / "sharded"
        # 1KB forces a real multi-shard write on a model this small — the
        # path where a stray buffer could be dropped from the index.
        trainer.model.save_pretrained(str(sharded), max_shard_size="1KB")
        phase("hf: sharded save_pretrained returned")
    return written


def _collect(
    root: Path, *, primary_suffixes: tuple[str, ...], export_marker: str
) -> list[tuple[Path, str]]:
    """Every checkpoint the framework left under ``root``, oldest first.

    Role is decided by filename: our derived exports carry
    ``export_marker`` in the name, everything else the framework wrote
    itself is primary. Composer's ``latest-rank0.pt`` is a duplicate of
    the newest save and would double-count, so it is dropped.
    """
    if not root.exists():
        return []
    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: (p.stat().st_mtime_ns, str(p)),
    )
    collected: list[tuple[Path, str]] = []
    for path in files:
        if path.name.startswith("latest-"):
            continue
        if export_marker in path.name:
            collected.append((path, "export"))
        elif path.suffix in primary_suffixes:
            collected.append((path, "primary"))
    return collected


def main() -> None:
    """Entry point for subprocess invocation."""
    config = CheckpointDriverConfig.from_env()
    Path(config.stats_jsonl_container_path).parent.mkdir(parents=True, exist_ok=True)
    probe = run_checkpoint_driver(config)
    print(PROBE_PREFIX + json.dumps(probe), flush=True)


if __name__ == "__main__":
    main()
