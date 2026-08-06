"""Stage 2 contract tests for checkpoint provenance.

Contract re-derived from purpose (not from the Stage 1 bodies, which
are stubs) per ~/.claude/rules/test-guide.md:

- Stamping NEVER fails training. Missing env, unwritable marker,
  absent Aim run, unstamped parent — all degrade to an unlinked or
  partial result, never an exception.
- Reading an UNSTAMPED-but-valid checkpoint returns None, not an
  error. Reading a nonexistent or non-checkpoint path DOES raise:
  the caller passed something wrong, which is different from a
  checkpoint that predates the feature.
- Propagated identity (submit_id / experiment / version) is
  independent of Aim. aim_run_hash is enrichment; its absence must
  never change whether a stamp happens.
- Format is decided by magic bytes, never by extension.
"""

from __future__ import annotations

import json
import os

import pytest

from astrolabe_callbacks import _core, contract
from astrolabe_callbacks.checkpoint import (
    BUFFER_NAME,
    META_KEY,
    CheckpointMeta,
    build_checkpoint_meta,
    export_checkpoint,
    read_checkpoint_meta,
    save_derived_checkpoint,
    stamp_checkpoint,
    embed_meta_as_buffer,
    read_meta_from_buffer,
    stamp_state_dict,
    strip_meta_buffer,
    write_first_checkpoint_marker_once,
)


def make_meta(
    *,
    submit_id="8a562a3a-c392-4661-ab86-a31649a12d97",
    experiment="06b-rtd-calibration",
    version="v1",
    aim_run_hash="eeca8aeb8e8b45d1a64c71e1",
    created_at="2026-08-02T12:00:00Z",
    **overrides,
):
    return CheckpointMeta(
        submit_id=submit_id,
        experiment=experiment,
        version=version,
        aim_run_hash=aim_run_hash,
        created_at=created_at,
        **overrides,
    )


@pytest.fixture
def astrolabe_env(monkeypatch):
    """Env as the engine sets it during an orchestrated submit."""
    monkeypatch.setenv(contract.ENV_SUBMIT_ID, "8a562a3a-c392-4661-ab86-a31649a12d97")
    monkeypatch.setenv(contract.ENV_EXPERIMENT_NAME, "06b-rtd-calibration")
    monkeypatch.setenv(
        contract.ENV_AIM_RUN_TAGS,
        contract.format_aim_run_tags(
            {
                contract.TAG_SUBMIT_ID: "8a562a3a-c392-4661-ab86-a31649a12d97",
                contract.TAG_EXPERIMENT: "06b-rtd-calibration",
                contract.TAG_VERSION: "v1",
            }
        ),
    )


@pytest.fixture
def bare_env(monkeypatch):
    """No astrolabe orchestration — researcher running locally."""
    for name in (
        contract.ENV_SUBMIT_ID,
        contract.ENV_EXPERIMENT_NAME,
        contract.ENV_AIM_RUN_TAGS,
    ):
        monkeypatch.delenv(name, raising=False)


class TestBuildMetaDegradesWithoutAstrolabe:
    def test_returns_unlinked_meta_when_env_absent(self, bare_env):
        meta = build_checkpoint_meta()
        assert meta.submit_id is None
        assert meta.experiment is None
        assert meta.version is None
        assert meta.linked is False

    def test_does_not_raise_when_env_absent(self, bare_env):
        build_checkpoint_meta()

    def test_hash_absent_does_not_prevent_propagated_identity(
        self, astrolabe_env, monkeypatch
    ):
        monkeypatch.setattr(_core, "current_run_hash", lambda: None)
        meta = build_checkpoint_meta()
        assert meta.aim_run_hash is None
        assert meta.submit_id == "8a562a3a-c392-4661-ab86-a31649a12d97"
        assert meta.version == "v1"
        assert meta.linked is True

    def test_created_at_is_utc_iso8601(self, astrolabe_env):
        assert build_checkpoint_meta().created_at.endswith("Z")


class TestMetaSerialization:
    def test_to_dict_omits_none_fields(self, bare_env):
        raw = CheckpointMeta(submit_id="abc", created_at="2026-08-02T12:00:00Z").to_dict()
        assert "experiment" not in raw
        assert "aim_run_hash" not in raw
        assert raw["submit_id"] == "abc"

    def test_from_dict_tolerates_unknown_keys(self):
        raw = make_meta().to_dict()
        raw["field_from_a_newer_callback"] = "whatever"
        assert CheckpointMeta.from_dict(raw).submit_id is not None

    def test_from_dict_tolerates_missing_keys(self):
        assert CheckpointMeta.from_dict({}).linked is False

    def test_round_trips(self):
        meta = make_meta()
        assert CheckpointMeta.from_dict(meta.to_dict()) == meta

    def test_submit_id_survives_at_full_fidelity(self):
        """IDs are a data channel; truncation belongs in log strings only."""
        full = "8a562a3a-c392-4661-ab86-a31649a12d97"
        assert CheckpointMeta.from_dict(make_meta(submit_id=full).to_dict()).submit_id == full

    def test_run_hash_survives_at_full_fidelity(self):
        full = "eeca8aeb8e8b45d1a64c71e1"
        assert (
            CheckpointMeta.from_dict(make_meta(aim_run_hash=full).to_dict()).aim_run_hash
            == full
        )


class TestLinked:
    @pytest.mark.parametrize(
        "meta",
        [
            CheckpointMeta(),
            CheckpointMeta(created_at="2026-08-02T12:00:00Z"),
            CheckpointMeta(experiment="exp"),
        ],
    )
    def test_false_without_identity(self, meta):
        assert meta.linked is False

    def test_true_on_propagated_identity_alone(self):
        assert CheckpointMeta(submit_id="abc").linked is True

    def test_true_on_hash_alone(self):
        assert CheckpointMeta(aim_run_hash="eeca8aeb").linked is True


class TestReadMissingAndMalformed:
    def test_raises_on_nonexistent_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_checkpoint_meta(tmp_path / "nope.pt")

    def test_raises_on_non_checkpoint_file(self, tmp_path):
        junk = tmp_path / "notes.txt"
        junk.write_text("this is not a checkpoint")
        with pytest.raises(ValueError):
            read_checkpoint_meta(junk)

    def test_returns_none_for_unstamped_state_dict(self):
        assert read_checkpoint_meta({"weight": [1, 2, 3]}) is None

    def test_returns_none_when_meta_block_is_corrupt(self):
        """A garbled block is indistinguishable from absent to the
        caller — both mean 'no usable provenance', neither is fatal."""
        assert read_checkpoint_meta({META_KEY: "not-a-dict"}) is None


class TestStamp:
    def test_injects_under_meta_key(self, bare_env):
        assert META_KEY in stamp_state_dict({"weight": [1]})

    def test_preserves_existing_entries(self, bare_env):
        out = stamp_state_dict({"weight": [1], "optimizer": {"lr": 0.1}})
        assert out["weight"] == [1]
        assert out["optimizer"] == {"lr": 0.1}

    def test_restamp_overwrites(self, bare_env):
        sd = stamp_state_dict({}, make_meta(submit_id="first"))
        sd = stamp_state_dict(sd, make_meta(submit_id="second"))
        assert read_checkpoint_meta(sd).submit_id == "second"

    def test_round_trips_through_read(self, bare_env):
        meta = make_meta()
        assert read_checkpoint_meta(stamp_state_dict({}, meta)) == meta


class TestExportFailureModes:
    def test_rejects_unknown_format(self, tmp_path, bare_env):
        with pytest.raises(ValueError):
            export_checkpoint({}, tmp_path / "m.bin", fmt="pickle")

    def test_safetensors_rejects_non_tensor_values(self, tmp_path, bare_env):
        pytest.importorskip("safetensors")
        with pytest.raises(ValueError):
            export_checkpoint(
                {"epoch": 3, "notes": "resumed"},
                tmp_path / "m.safetensors",
                fmt="safetensors",
            )

    def test_creates_missing_parent_dirs(self, tmp_path, bare_env):
        out = export_checkpoint({}, tmp_path / "a" / "b" / "m.pt", fmt="pt")
        assert out.exists()

    def test_meta_key_not_written_as_a_tensor(self, tmp_path, bare_env):
        """META_KEY is stripped before write and re-embedded per-format,
        so it must not leak into the tensor namespace."""
        pytest.importorskip("safetensors")
        from safetensors import safe_open

        path = export_checkpoint({}, tmp_path / "m.safetensors", fmt="safetensors")
        with safe_open(str(path), framework="pt") as f:
            assert META_KEY not in f.keys()

    def test_safetensors_meta_lands_in_header(self, tmp_path, bare_env):
        pytest.importorskip("safetensors")
        from safetensors import safe_open

        path = export_checkpoint(
            {}, tmp_path / "m.safetensors", fmt="safetensors", meta=make_meta()
        )
        with safe_open(str(path), framework="pt") as f:
            assert json.loads(f.metadata()[META_KEY])["submit_id"] is not None


class TestFormatSniffing:
    def test_reads_pt_content_named_safetensors(self, tmp_path, bare_env):
        """Extension lies; magic bytes don't."""
        liar = tmp_path / "model.safetensors"
        export_checkpoint({}, liar, fmt="pt", meta=make_meta())
        assert read_checkpoint_meta(liar).submit_id is not None

    def test_reads_extensionless_file(self, tmp_path, bare_env):
        path = tmp_path / "checkpoint"
        export_checkpoint({}, path, fmt="pt", meta=make_meta())
        assert read_checkpoint_meta(path).submit_id is not None


# Auto-clears the moment tools/vendor-contract.py pulls engine
# CONTRACT_VERSION 1.4.0 in. Not a mute: the reason names the blocker,
# and nothing has to be remembered when astrolabe#68 merges.
_MARKER_CONST_MISSING = not hasattr(contract, "ENV_FIRST_CHECKPOINT_MARKER")


@pytest.mark.skipif(
    _MARKER_CONST_MISSING,
    reason=(
        "blocked on astrolabe#68 + re-vendor: "
        "contract.ENV_FIRST_CHECKPOINT_MARKER does not exist in the "
        "vendored contract yet (engine is at 1.3.0, needs 1.4.0)"
    ),
)
class TestFirstCheckpointMarker:
    def test_noop_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(contract.ENV_FIRST_CHECKPOINT_MARKER, raising=False)
        write_first_checkpoint_marker_once()

    def test_touches_marker_when_env_set(self, tmp_path, monkeypatch):
        marker = tmp_path / "first-checkpoint.marker"
        monkeypatch.setenv(contract.ENV_FIRST_CHECKPOINT_MARKER, str(marker))
        write_first_checkpoint_marker_once()
        assert marker.exists()

    def test_unwritable_path_does_not_raise(self, monkeypatch):
        """A failed touch degrades the healing bound. It must never take
        down a training run that is otherwise fine."""
        monkeypatch.setenv(
            contract.ENV_FIRST_CHECKPOINT_MARKER, "/nonexistent-dir/x.marker"
        )
        write_first_checkpoint_marker_once()

    def test_writes_at_most_once(self, tmp_path, monkeypatch):
        marker = tmp_path / "m.marker"
        monkeypatch.setenv(contract.ENV_FIRST_CHECKPOINT_MARKER, str(marker))
        write_first_checkpoint_marker_once()
        first = marker.stat().st_mtime_ns
        write_first_checkpoint_marker_once()
        assert marker.stat().st_mtime_ns == first

    def test_marker_is_independent_of_first_metric_marker(self, tmp_path, monkeypatch):
        """Distinct windows: a run emits metrics long before it writes a
        checkpoint. Sharing one marker would collapse both bounds."""
        assert (
            contract.ENV_FIRST_CHECKPOINT_MARKER != contract.ENV_FIRST_METRIC_MARKER
        )


class TestCurrentRunRegistry:
    def test_returns_none_with_no_registered_run(self):
        assert _core.current_run_hash() is None

    def test_returns_hash_after_register(self):
        run = type("R", (), {"hash": "eeca8aeb8e8b45d1a64c71e1"})()
        _core.register_current_run(run)
        try:
            assert _core.current_run_hash() == "eeca8aeb8e8b45d1a64c71e1"
        finally:
            _core.unregister_current_run(run)

    def test_unregister_clears(self):
        run = type("R", (), {"hash": "abc"})()
        _core.register_current_run(run)
        _core.unregister_current_run(run)
        assert _core.current_run_hash() is None

    def test_unregister_of_other_run_does_not_clear(self):
        """Out-of-order close: run A finishes after run B registered.
        A's teardown must not blank B's entry."""
        a = type("R", (), {"hash": "aaa"})()
        b = type("R", (), {"hash": "bbb"})()
        _core.register_current_run(a)
        _core.register_current_run(b)
        try:
            _core.unregister_current_run(a)
            assert _core.current_run_hash() == "bbb"
        finally:
            _core.unregister_current_run(b)

    def test_sequential_registration_last_wins(self):
        """Hyperparameter sweeps open runs back to back in one process."""
        a = type("R", (), {"hash": "aaa"})()
        b = type("R", (), {"hash": "bbb"})()
        _core.register_current_run(a)
        _core.register_current_run(b)
        try:
            assert _core.current_run_hash() == "bbb"
        finally:
            _core.unregister_current_run(b)


class TestBufferEmbedding:
    """HuggingFace's mechanism. Verified against real safetensors in
    the testbed; these pin the contract around it."""

    def test_buffer_lands_in_state_dict(self):
        torch = pytest.importorskip("torch")
        model = torch.nn.Linear(4, 4)
        embed_meta_as_buffer(model, make_meta())
        assert BUFFER_NAME in model.state_dict()

    def test_round_trips_through_state_dict(self):
        torch = pytest.importorskip("torch")
        model = torch.nn.Linear(4, 4)
        meta = make_meta()
        embed_meta_as_buffer(model, meta)
        assert read_meta_from_buffer(model.state_dict()) == meta

    def test_reembedding_replaces_rather_than_duplicates(self):
        """The hash must track the live run, not the first one seen."""
        torch = pytest.importorskip("torch")
        model = torch.nn.Linear(4, 4)
        embed_meta_as_buffer(model, make_meta(aim_run_hash="first"))
        embed_meta_as_buffer(model, make_meta(aim_run_hash="second"))
        assert read_meta_from_buffer(model.state_dict()).aim_run_hash == "second"

    def test_read_returns_none_when_buffer_absent(self):
        torch = pytest.importorskip("torch")
        assert read_meta_from_buffer(torch.nn.Linear(4, 4).state_dict()) is None

    def test_read_returns_none_on_undecodable_buffer(self):
        torch = pytest.importorskip("torch")
        garbage = torch.tensor([255, 254, 253], dtype=torch.uint8)
        assert read_meta_from_buffer({BUFFER_NAME: garbage}) is None

    def test_strip_restores_a_strict_loadable_state_dict(self):
        """The documented escape hatch for the strict-load footgun."""
        torch = pytest.importorskip("torch")
        model = torch.nn.Linear(4, 4)
        embed_meta_as_buffer(model, make_meta())
        stripped = strip_meta_buffer(model.state_dict())
        torch.nn.Linear(4, 4).load_state_dict(stripped, strict=True)

    def test_strip_does_not_mutate_input(self):
        torch = pytest.importorskip("torch")
        model = torch.nn.Linear(4, 4)
        embed_meta_as_buffer(model, make_meta())
        sd = model.state_dict()
        strip_meta_buffer(sd)
        assert BUFFER_NAME in sd

    def test_buffer_name_matches_meta_key(self):
        """One name across all three mechanisms — a reader scanning a
        state dict shouldn't have to know which path wrote it."""
        assert BUFFER_NAME == META_KEY


# --- Stage 2 (EVAL-6): copy-forward linkage + transform helpers -------------
#
# Contract re-derived from purpose, not from the Stage 1 stubs:
#
# - A transform normally runs with NO logger (surgery, quantization and
#   friends are preprocessing steps). That is precisely when provenance
#   must survive, so "no live run" is the primary path here, not an edge.
# - aim_run_hash resolves to the live run if there is one, the parent's
#   otherwise. Inheritance is what keeps the read side a plain lookup.
# - derived_from always records the parent's OWN hash. It equals the
#   inherited value in the no-logger case and diverges when a live run
#   derives from someone else's checkpoint.
# - Resuming into the run that wrote the parent is continuation, not
#   derivation: the chain must not grow.
# - An unstamped parent is not an error. It yields an unlinked result,
#   the same as any checkpoint predating the feature.

ORIGIN_HASH = "aaaa1111bbbb2222cccc3333"
LIVE_HASH = "dddd4444eeee5555ffff6666"


def _no_live_run(monkeypatch):
    monkeypatch.setattr(_core, "current_run_hash", lambda: None)


def _live_run(monkeypatch, run_hash=LIVE_HASH):
    monkeypatch.setattr(_core, "current_run_hash", lambda: run_hash)


def _parent_file(tmp_path, name="parent.pt", **meta_kwargs):
    return export_checkpoint(
        {}, tmp_path / name, fmt="pt", meta=make_meta(**meta_kwargs)
    )


class TestDerivedCheckpointFailureModes:
    def test_missing_parent_path_raises(self, tmp_path, bare_env, monkeypatch):
        _no_live_run(monkeypatch)
        with pytest.raises(FileNotFoundError):
            save_derived_checkpoint({}, tmp_path / "out.pt", tmp_path / "nope.pt")

    def test_unrecognizable_parent_raises_value_error(
        self, tmp_path, bare_env, monkeypatch
    ):
        _no_live_run(monkeypatch)
        junk = tmp_path / "junk.pt"
        junk.write_bytes(b"not a checkpoint at all")
        with pytest.raises(ValueError):
            save_derived_checkpoint({}, tmp_path / "out.pt", junk)

    def test_unknown_dest_extension_without_fmt_raises(
        self, tmp_path, bare_env, monkeypatch
    ):
        _no_live_run(monkeypatch)
        parent = _parent_file(tmp_path)
        with pytest.raises(ValueError):
            save_derived_checkpoint({}, tmp_path / "out.bin", parent)

    def test_safetensors_rejects_non_tensor_values(
        self, tmp_path, bare_env, monkeypatch
    ):
        pytest.importorskip("safetensors")
        _no_live_run(monkeypatch)
        parent = _parent_file(tmp_path)
        with pytest.raises(ValueError):
            save_derived_checkpoint(
                {"epoch": 3}, tmp_path / "out.safetensors", parent
            )

    def test_unstamped_parent_yields_unlinked_result_not_an_error(
        self, tmp_path, bare_env, monkeypatch
    ):
        """A checkpoint predating the feature is a supported input."""
        _no_live_run(monkeypatch)
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        out = save_derived_checkpoint({}, tmp_path / "out.pt", plain)
        assert read_checkpoint_meta(out).aim_run_hash is None


class TestCopyForwardWithoutALogger:
    """The primary path: transforms run as preprocessing, no run live."""

    def test_inherits_parent_run_hash(self, tmp_path, bare_env, monkeypatch):
        _no_live_run(monkeypatch)
        parent = _parent_file(tmp_path, aim_run_hash=ORIGIN_HASH)
        out = save_derived_checkpoint({}, tmp_path / "out.pt", parent)
        assert read_checkpoint_meta(out).aim_run_hash == ORIGIN_HASH

    def test_records_parent_as_derived_from(self, tmp_path, bare_env, monkeypatch):
        _no_live_run(monkeypatch)
        parent = _parent_file(tmp_path, aim_run_hash=ORIGIN_HASH)
        out = save_derived_checkpoint({}, tmp_path / "out.pt", parent)
        assert read_checkpoint_meta(out).derived_from == ORIGIN_HASH

    def test_increments_chain_length(self, tmp_path, bare_env, monkeypatch):
        _no_live_run(monkeypatch)
        parent = _parent_file(tmp_path, aim_run_hash=ORIGIN_HASH)
        out = save_derived_checkpoint({}, tmp_path / "out.pt", parent)
        assert read_checkpoint_meta(out).derivation_chain_length == 1

    def test_two_chained_transforms_still_resolve_to_origin(
        self, tmp_path, bare_env, monkeypatch
    ):
        """The regression this slice exists for.

        Surgery then quantization, neither with a logger. Before the
        fix the second hop saw a parent with no hash of its own, gave
        up, and wrote a checkpoint indistinguishable from one trained
        from scratch.
        """
        _no_live_run(monkeypatch)
        origin = _parent_file(tmp_path, aim_run_hash=ORIGIN_HASH)
        hop1 = save_derived_checkpoint({}, tmp_path / "hop1.pt", origin)
        hop2 = save_derived_checkpoint({}, tmp_path / "hop2.pt", hop1)

        meta = read_checkpoint_meta(hop2)
        assert meta.aim_run_hash == ORIGIN_HASH
        assert meta.derivation_chain_length == 2

    def test_accepts_an_already_read_meta_block_as_parent(
        self, tmp_path, bare_env, monkeypatch
    ):
        _no_live_run(monkeypatch)
        out = save_derived_checkpoint(
            {}, tmp_path / "out.pt", make_meta(aim_run_hash=ORIGIN_HASH)
        )
        assert read_checkpoint_meta(out).aim_run_hash == ORIGIN_HASH


class TestCopyForwardWithALiveRun:
    def test_live_run_wins_over_inheritance(self, tmp_path, bare_env, monkeypatch):
        """A tracked fine-tune is its own training run, not a passthrough."""
        _live_run(monkeypatch)
        parent = _parent_file(tmp_path, aim_run_hash=ORIGIN_HASH)
        out = save_derived_checkpoint({}, tmp_path / "out.pt", parent)
        assert read_checkpoint_meta(out).aim_run_hash == LIVE_HASH

    def test_derived_from_still_records_the_parent(
        self, tmp_path, bare_env, monkeypatch
    ):
        """Where derived_from earns its place: it diverges from
        aim_run_hash exactly when a live run derives from someone
        else's checkpoint."""
        _live_run(monkeypatch)
        parent = _parent_file(tmp_path, aim_run_hash=ORIGIN_HASH)
        meta = read_checkpoint_meta(
            save_derived_checkpoint({}, tmp_path / "out.pt", parent)
        )
        assert meta.derived_from == ORIGIN_HASH
        assert meta.aim_run_hash != meta.derived_from


class TestContinuationIsNotDerivation:
    def test_resuming_own_checkpoint_does_not_grow_the_chain(self, bare_env, monkeypatch):
        """Run B resuming a checkpoint B itself wrote is continuation.

        The guard for this currently sits inline next to the bug being
        removed, which makes it the easiest thing to break in Stage 3.
        """
        _live_run(monkeypatch, ORIGIN_HASH)
        parent = make_meta(aim_run_hash=ORIGIN_HASH, derivation_chain_length=0)
        meta = build_checkpoint_meta(parent=parent)
        assert meta.derivation_chain_length == 0
        assert meta.derived_from is None

    def test_resuming_someone_elses_checkpoint_is_derivation(
        self, bare_env, monkeypatch
    ):
        _live_run(monkeypatch)
        parent = make_meta(aim_run_hash=ORIGIN_HASH)
        meta = build_checkpoint_meta(parent=parent)
        assert meta.derived_from == ORIGIN_HASH
        assert meta.derivation_chain_length == 1


class TestStampCheckpointFailureModes:
    def test_missing_file_raises(self, tmp_path, bare_env):
        with pytest.raises(FileNotFoundError):
            stamp_checkpoint(tmp_path / "nope.pt", aim_run_hash=ORIGIN_HASH)

    def test_non_checkpoint_raises_value_error(self, tmp_path, bare_env):
        junk = tmp_path / "junk.pt"
        junk.write_bytes(b"still not a checkpoint")
        with pytest.raises(ValueError):
            stamp_checkpoint(junk, aim_run_hash=ORIGIN_HASH)


class TestStampCheckpoint:
    def test_retrofits_a_hash_onto_an_unstamped_file(self, tmp_path, bare_env):
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        stamp_checkpoint(plain, aim_run_hash=ORIGIN_HASH)
        assert read_checkpoint_meta(plain).aim_run_hash == ORIGIN_HASH

    def test_restamping_overwrites_rather_than_merges(self, tmp_path, bare_env):
        """Correcting a wrong hash is a supported sequence."""
        path = _parent_file(tmp_path, aim_run_hash=ORIGIN_HASH)
        stamp_checkpoint(path, aim_run_hash=LIVE_HASH)
        assert read_checkpoint_meta(path).aim_run_hash == LIVE_HASH

    def test_omitted_fields_fall_back_to_the_environment(
        self, tmp_path, astrolabe_env, monkeypatch
    ):
        _no_live_run(monkeypatch)
        plain = export_checkpoint({}, tmp_path / "plain.pt", fmt="pt")
        stamp_checkpoint(plain)
        meta = read_checkpoint_meta(plain)
        assert meta.submit_id == "8a562a3a-c392-4661-ab86-a31649a12d97"
        assert meta.version == "v1"

    def test_stamps_safetensors_via_header(self, tmp_path, bare_env):
        pytest.importorskip("safetensors")
        path = export_checkpoint({}, tmp_path / "m.safetensors", fmt="safetensors")
        stamp_checkpoint(path, aim_run_hash=ORIGIN_HASH)
        assert read_checkpoint_meta(path).aim_run_hash == ORIGIN_HASH


class TestDerivedCheckpointRoundTrip:
    def test_writes_a_readable_checkpoint(self, tmp_path, bare_env, monkeypatch):
        _no_live_run(monkeypatch)
        parent = _parent_file(tmp_path, aim_run_hash=ORIGIN_HASH)
        out = save_derived_checkpoint({}, tmp_path / "nested" / "out.pt", parent)
        assert out.exists()
        assert read_checkpoint_meta(out) is not None

    def test_propagated_identity_survives_the_hop(
        self, tmp_path, astrolabe_env, monkeypatch
    ):
        _no_live_run(monkeypatch)
        parent = _parent_file(tmp_path, aim_run_hash=ORIGIN_HASH)
        meta = read_checkpoint_meta(
            save_derived_checkpoint({}, tmp_path / "out.pt", parent)
        )
        assert meta.submit_id is not None
        assert meta.linked is True
