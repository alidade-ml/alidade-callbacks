"""Integration tests for checkpoint provenance against a real Aim server.

Unit tests (``tests/test_checkpoint.py``) cover the meta block, format
sniffing, and the failure modes — all answerable without Aim. This file
covers what isn't:

1. **The framework actually writes our block.** Unit tests can assert
   our hook returns the right dict; only a real save proves the
   framework serializes it and replays it on resume. That's a contract
   we don't own, and it differs per framework.

2. **The embedded hash matches the run Aim actually opened.** The hash
   is minted at run-open, so a mock proves nothing. A mismatch is
   silent — the eval attaches to a run that doesn't exist.

3. **Provenance survives schema-finalize.** ``maybe_finalize_schema``
   closes and reopens the Run mid-training. It already lost ``run.name``
   once (the v2.0.0-rc1 regression that broke CoLA probe). A checkpoint
   written after a finalize carrying a stale hash is the same bug in a
   new hat.

Framework coverage
------------------

Parametrized across every framework that ships a checkpointer, because
the *behavior* is a cross-framework contract even though the *mechanism*
differs:

- **Composer** — collects our dict via ``Callback.state_dict()``
- **Lightning** — hands us the dict to mutate via ``on_save_checkpoint``
- **raw PyTorch** — no framework slot; explicit ``_astrolabe_meta`` key
- **HuggingFace** — no hook at all; a registered uint8 buffer rides
  into ``state_dict()`` and therefore into every save

HF's mechanism is the one that needs real exercising: it was chosen over
subclassing ``Trainer`` and over rewriting the finished file, and it was
validated against plain ``nn.Module`` + safetensors, NOT through a live
``Trainer``. "The mechanism is PyTorch-level so it should carry" is
exactly the shape of claim this repo has been burned by. See
``TestHuggingFaceBufferMechanism``.

Each framework skips if its extra is not installed.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from tests.testbed.harness.assertions import get_metric_series
from tests.testbed.harness.checkpoint_driver import (
    CheckpointDriverConfig,
    CheckpointDriverResult,
)

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


# Mechanism differs per framework; behavior must not. Any test taking
# this fixture asserts a cross-framework contract.
FRAMEWORKS = ["composer", "lightning", "pytorch", "huggingface"]

# The driver names the HF path after the extra it needs, the scenario
# after the vendor. One translation table rather than two vocabularies.
_DRIVER_NAME = {
    "composer": "composer",
    "lightning": "lightning",
    "pytorch": "pytorch",
    "huggingface": "hf",
}

_IMPORT_NAME = {
    "composer": "composer",
    "lightning": "lightning",
    "pytorch": "torch",
    "huggingface": "transformers",
}

# Long enough that any truncation to a "readable" prefix shows up as an
# inequality rather than passing by luck.
SUBMIT_ID = "sub-01JQ8Z4K7V9XW2N6TB3RCDEFGH-full-fidelity"
VERSION = "v7"

RunCheckpointFixture = Callable[[CheckpointDriverConfig], CheckpointDriverResult]


@pytest.fixture(params=FRAMEWORKS)
def framework(request):
    """Drives one training run per framework with its checkpointer
    attached. Skips when the framework's extra is absent."""
    pytest.importorskip(_IMPORT_NAME[request.param])
    return request.param


def _config(
    testbed: "TestbedHandle",
    framework: str,
    stats_path: Path,
    **overrides,
) -> CheckpointDriverConfig:
    """One driver config with a per-invocation identity.

    ``run_name`` / ``workdir`` are uniquified because the client
    container and the Aim repo both persist across a session under
    ``TESTBED_KEEP=1``, and a stale run of the same name would be
    indistinguishable from this one's.
    """
    token = uuid.uuid4().hex[:10]
    defaults = dict(
        framework=_DRIVER_NAME[framework],
        aim_url=testbed.aim_url_from_client,
        experiment_name=f"testbed-ckpt-{token}",
        run_name=f"ckpt-{framework}-{token}",
        submit_id=SUBMIT_ID,
        version=VERSION,
        steps=3,
        save_every=1,
        with_logger=True,
        embed_in_weights=True,
        export_formats=[],
        new_metrics_at=[],
        workdir=f"/tmp/ckpt-testbed/{token}",
        marker_path=f"/tmp/ckpt-testbed/{token}-marker.tag",
        resume_from=None,
        stats_jsonl_container_path=f"/tmp/ckpt-testbed/{token}-stats.jsonl",
        driver_flags={},
    )
    defaults.update(overrides)
    return CheckpointDriverConfig(**defaults)


def _run(
    run_checkpoint_driver: RunCheckpointFixture, config: CheckpointDriverConfig
) -> CheckpointDriverResult:
    result = run_checkpoint_driver(config)
    assert result.exit_code == 0, f"driver failed:\n{result.stderr}"
    return result


def _finalize_count(result: CheckpointDriverResult) -> int:
    return sum(1 for e in result.stats_events if e.get("kind") == "schema_finalized")


class TestCheckpointCarriesLiveRunIdentity:
    def test_embedded_hash_matches_the_opened_aim_run(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """The join key the dashboard's Eval tab uses. If it drifts, evals
        attach to a run that was never created and the tab is silently
        empty."""
        config = _config(testbed, framework, stats_jsonl_path)
        result = _run(run_checkpoint_driver, config)

        run_hash = result.probe["run_hash"]
        assert run_hash is not None, "no Aim run was opened; nothing to match against"
        # Not merely equal to itself: the run has to exist in the repo,
        # with the metrics this training actually wrote.
        assert get_metric_series(aim_repo, run_hash, "metric_0"), (
            f"run {run_hash!r} carries no metric_0 — the embedded hash "
            f"would point at a run the dashboard cannot resolve"
        )
        for checkpoint in result.checkpoints():
            assert result.meta_of(checkpoint)["aim_run_hash"] == run_hash, (
                f"{checkpoint['path']} embeds a hash that is not the opened run"
            )

    def test_embedded_submit_id_is_full_fidelity(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """Truncation belongs in log strings, never a data channel. The
        stats jsonl already got this wrong once."""
        config = _config(testbed, framework, stats_jsonl_path)
        result = _run(run_checkpoint_driver, config)

        meta = result.meta_of(result.primary())
        assert meta["submit_id"] == SUBMIT_ID
        assert meta["experiment"] == config.experiment_name
        assert meta["version"] == VERSION

    def test_propagated_identity_present_without_a_logger(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """Checkpointer attached, logger absent: the stamp still happens
        from env, with aim_run_hash simply None."""
        config = _config(testbed, framework, stats_jsonl_path, with_logger=False)
        result = _run(run_checkpoint_driver, config)

        meta = result.meta_of(result.primary())
        assert meta["submit_id"] == SUBMIT_ID
        assert meta["experiment"] == config.experiment_name
        assert meta["version"] == VERSION
        assert "aim_run_hash" not in meta, (
            "no run was opened, so there is no hash to claim"
        )


class TestFrameworkSerializesOurBlock:
    """The mechanism is per-framework; that it works is not optional for
    any of them."""

    def test_block_round_trips_through_a_real_save(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        config = _config(testbed, framework, stats_jsonl_path, steps=2)
        result = _run(run_checkpoint_driver, config)

        primaries = [c for c in result.checkpoints() if c["role"] == "primary"]
        assert len(primaries) >= 2, f"expected repeated saves, got {primaries}"
        for checkpoint in primaries:
            meta = result.meta_of(checkpoint)
            assert meta["submit_id"] == SUBMIT_ID
            assert meta["experiment"] == config.experiment_name
            assert meta["aim_run_hash"] == result.probe["run_hash"]
            assert meta["derivation_chain_length"] == 0

    def test_resume_replays_parent_provenance(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        parent = _run(
            run_checkpoint_driver, _config(testbed, framework, stats_jsonl_path)
        )
        parent_hash = parent.probe["run_hash"]
        parent_path = _resume_target(framework, parent)

        child = _run(
            run_checkpoint_driver,
            _config(testbed, framework, stats_jsonl_path, resume_from=parent_path),
        )
        child_hash = child.probe["run_hash"]
        assert child_hash != parent_hash, "the resume opened a fresh Aim run"

        meta = child.meta_of(child.primary())
        # The failure that motivated this test: a resumed checkpoint
        # claiming the parent's identity as its own, so every eval run
        # attributes to the wrong training.
        assert meta["aim_run_hash"] == child_hash
        assert meta["derived_from"] == parent_hash
        assert meta["derivation_chain_length"] == 1

    def test_resume_from_malformed_block_does_not_fail_training(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """Provenance is never worth killing a run over."""
        parent = _run(
            run_checkpoint_driver, _config(testbed, framework, stats_jsonl_path)
        )
        child = _run(
            run_checkpoint_driver,
            _config(
                testbed,
                framework,
                stats_jsonl_path,
                resume_from=_resume_target(framework, parent),
                driver_flags={"TESTBED_CORRUPT_PARENT_META": "1"},
            ),
        )

        meta = child.meta_of(child.primary())
        assert meta["aim_run_hash"] == child.probe["run_hash"]
        # Unreadable parent means no link — but a stamp all the same.
        assert "derived_from" not in meta

    def test_composer_block_lands_under_our_class_qualname(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """Composer-specific: it keys callback state by
        ``type(obj).__qualname__``. Renaming our class would silently
        orphan every previously-written checkpoint."""
        pytest.importorskip("composer")
        result = _run(
            run_checkpoint_driver, _config(testbed, "composer", stats_jsonl_path)
        )

        checkpoint = result.primary()
        assert checkpoint["composer_callback_keys"] is not None
        assert "AstrolabeComposerCheckpointer" in checkpoint["composer_callback_keys"]
        assert checkpoint["composer_block"] is not None
        assert checkpoint["composer_block"]["submit_id"] == SUBMIT_ID
        # Reading the top level instead would have found nothing.
        assert "_astrolabe_meta" not in checkpoint["top_level_keys"]

    def test_lightning_mutation_survives_its_own_hooks(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """Lightning-specific: other callbacks also mutate the same dict
        in ``on_save_checkpoint``. Ours must not be clobbered by, or
        clobber, a co-attached callback."""
        pytest.importorskip("lightning")
        result = _run(
            run_checkpoint_driver, _config(testbed, "lightning", stats_jsonl_path)
        )

        # The driver attaches a co-mutator on each side of ours, so this
        # covers both "they ran first" and "they ran after".
        checkpoint = result.primary()
        assert "co_attached_callback_state" in checkpoint["top_level_keys"]
        assert "_astrolabe_meta" in checkpoint["top_level_keys"]
        assert result.meta_of(checkpoint)["submit_id"] == SUBMIT_ID


class TestSurvivesSchemaFinalize:
    def test_hash_still_correct_after_finalize(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """maybe_finalize_schema closes + reopens the Run. It already
        dropped run.name once (v2.0.0-rc1). A checkpoint written after a
        finalize must carry the same hash as one written before."""
        config = _config(
            testbed, framework, stats_jsonl_path, steps=4, new_metrics_at=[2]
        )
        result = _run(run_checkpoint_driver, config)

        assert _finalize_count(result) >= 1, (
            "no finalize fired; the test would pass without exercising the reopen"
        )
        hashes = {
            result.meta_of(c)["aim_run_hash"] for c in result.checkpoints()
        }
        assert hashes == {result.probe["run_hash"]}

    def test_checkpoint_after_multiple_finalizes(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        config = _config(
            testbed, framework, stats_jsonl_path, steps=6, new_metrics_at=[1, 3, 5]
        )
        result = _run(run_checkpoint_driver, config)

        assert _finalize_count(result) >= 2, (
            f"expected repeated finalizes, saw {_finalize_count(result)}"
        )
        last = result.checkpoints()[-1]
        assert result.meta_of(last)["aim_run_hash"] == result.probe["run_hash"]


class TestExports:
    def test_safetensors_export_readable_by_safetensors_lib(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """Our header write has to be legible to the real library, not
        just to our own reader."""
        pytest.importorskip("safetensors")
        result = _run(
            run_checkpoint_driver,
            _config(
                testbed, framework, stats_jsonl_path, export_formats=["safetensors"]
            ),
        )

        exported = _host_path(result, _export(result, ".safetensors"))
        from safetensors import safe_open

        with safe_open(exported, framework="pt") as handle:
            metadata = handle.metadata()
            tensor_names = list(handle.keys())
        assert metadata is not None and "_astrolabe_meta" in metadata, (
            f"the real library sees no astrolabe metadata in {exported} "
            f"(header metadata: {metadata})"
        )
        block = json.loads(metadata["_astrolabe_meta"])
        assert block["submit_id"] == SUBMIT_ID
        assert block["aim_run_hash"] == result.probe["run_hash"]
        # The block belongs in the header, not smuggled in as a tensor.
        assert "_astrolabe_meta" not in tensor_names
        assert tensor_names, "export dropped every weight"

    def test_exported_copy_carries_same_identity_as_primary(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        # ``pt`` is not usable here: the raw-PyTorch primary is already a
        # .pt, so a .pt export resolves to the same path.
        result = _run(
            run_checkpoint_driver,
            _config(
                testbed, framework, stats_jsonl_path, export_formats=["safetensors"]
            ),
        )

        primary = result.meta_of(result.primary())
        exported = result.meta_of(_export(result, ".safetensors"))
        for field in ("submit_id", "experiment", "version", "aim_run_hash"):
            assert exported[field] == primary[field], f"{field} diverged"


class TestFirstCheckpointMarker:
    def test_marker_touched_on_first_checkpoint(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """Engine-side probe closes the `until: first_checkpoint` healing
        window on this file's existence."""
        result = _run(
            run_checkpoint_driver, _config(testbed, framework, stats_jsonl_path)
        )

        marker = result.probe["marker"]
        assert marker["existed_at_start"] is False
        assert marker["exists_at_end"] is True, (
            "the healing window never closes without this file"
        )

    def test_marker_not_rewritten_on_later_checkpoints(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        result = _run(
            run_checkpoint_driver,
            _config(
                testbed,
                framework,
                stats_jsonl_path,
                steps=4,
                driver_flags={"TESTBED_PROBE_MARKER_LATCH": "1"},
            ),
        )

        primaries = [c for c in result.checkpoints() if c["role"] == "primary"]
        assert len(primaries) >= 2, "need repeated saves for this to mean anything"
        assert result.probe["marker"]["mtime_ns"] < primaries[-1]["mtime_ns"], (
            "marker is newer than the last checkpoint, so it was re-touched"
        )
        # The latch is per marker path, so a second call in the same
        # process must not bring the file back after we remove it.
        assert result.probe["marker"]["recreated_after_unlink"] is False

    def test_training_survives_unwritable_marker_path(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        result = _run(
            run_checkpoint_driver,
            _config(
                testbed,
                framework,
                stats_jsonl_path,
                marker_path="/no-such-directory/first-checkpoint.tag",
            ),
        )

        assert result.probe["marker"]["exists_at_end"] is False
        # Training finished and the checkpoint is still stamped — losing
        # the healing bound must not cost the provenance too.
        assert result.meta_of(result.primary())["submit_id"] == SUBMIT_ID


class TestHuggingFaceBufferMechanism:
    """HF-specific. The buffer was validated against plain nn.Module +
    safetensors, not through a real Trainer — that gap closes here."""

    def test_buffer_survives_a_real_trainer_save(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """The load-bearing claim. If HF's save path drops or renames
        non-parameter buffers, the whole mechanism is void and we owe
        the design question a second look."""
        pytest.importorskip("transformers")
        result = _run(
            run_checkpoint_driver, _config(testbed, "huggingface", stats_jsonl_path)
        )

        checkpoint = result.primary()
        assert checkpoint["path"].endswith("model.safetensors"), (
            "not HF's own save path"
        )
        # Named, not renamed, and sitting among the weights the Trainer
        # wrote rather than beside them.
        assert "_astrolabe_meta" in checkpoint["tensor_names"]
        assert len(checkpoint["tensor_names"]) > 1, "no weights were saved"
        assert checkpoint["buffer_meta"] is not None
        assert checkpoint["buffer_meta"]["submit_id"] == SUBMIT_ID
        assert checkpoint["buffer_meta"]["aim_run_hash"] == result.probe["run_hash"]
        # And the public reader resolves it from the path alone, which is
        # all an eval script is given.
        assert result.meta_of(checkpoint) == checkpoint["buffer_meta"]

    def test_buffer_present_in_safetensors_shard(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """HF shards large models. Provenance must not land in only one
        shard, or a partial load loses it."""
        pytest.importorskip("transformers")
        result = _run(
            run_checkpoint_driver,
            _config(
                testbed,
                "huggingface",
                stats_jsonl_path,
                driver_flags={"TESTBED_HF_SHARD_SAVE": "1"},
            ),
        )

        shard = result.probe["hf_shard"]
        assert len(shard["shard_files"]) > 1, "the save did not actually shard"
        assert shard["index_exists"], "no index means no way to find the buffer"
        # A tensor lives in exactly one shard; what matters is that the
        # index still points at it so a full load reassembles it.
        assert "_astrolabe_meta" in shard["weight_map_keys"]
        assert shard["buffer_shard"] in shard["shard_files"]
        assert shard["merged_meta"] is not None
        assert shard["merged_meta"]["submit_id"] == SUBMIT_ID

    def test_from_pretrained_warns_but_loads(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """The documented cost. from_pretrained is non-strict — assert
        it stays a warning and never becomes an error."""
        pytest.importorskip("transformers")
        result = _run(
            run_checkpoint_driver,
            _config(
                testbed,
                "huggingface",
                stats_jsonl_path,
                driver_flags={"TESTBED_HF_LOAD_PROBE": "1"},
            ),
        )

        load = result.probe["hf_load"]
        assert load["from_pretrained_error"] is None
        assert load["from_pretrained_ok"] is True
        assert load["from_pretrained_unexpected_keys"] == ["_astrolabe_meta"]
        # Our key must not have displaced a real weight.
        assert load["from_pretrained_missing_keys"] == []

    def test_strict_load_fails_and_strip_fixes_it(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """Pins the footgun AND its escape hatch. If a future HF version
        changes this, we want to know from a test, not a user."""
        pytest.importorskip("transformers")
        result = _run(
            run_checkpoint_driver,
            _config(
                testbed,
                "huggingface",
                stats_jsonl_path,
                driver_flags={"TESTBED_HF_LOAD_PROBE": "1"},
            ),
        )

        load = result.probe["hf_load"]
        assert load["strict_load_error"] is not None, (
            "strict load stopped failing; the documented footgun is stale"
        )
        assert "_astrolabe_meta" in load["strict_load_error"]
        assert load["strict_load_after_strip_error"] is None

    def test_embed_in_weights_false_still_touches_marker(
        self,
        testbed: "TestbedHandle",
        stats_jsonl_path: Path,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """Opting out of eval linkage must not cost you
        `until: first_checkpoint`."""
        pytest.importorskip("transformers")
        result = _run(
            run_checkpoint_driver,
            _config(
                testbed,
                "huggingface",
                stats_jsonl_path,
                embed_in_weights=False,
            ),
        )

        assert result.probe["marker"]["exists_at_end"] is True
        checkpoint = result.primary()
        assert "_astrolabe_meta" not in checkpoint["tensor_names"]
        assert checkpoint["meta"] is None, "opting out still embedded provenance"


def _resume_target(framework: str, result: CheckpointDriverResult) -> str:
    """Container path a child run resumes from.

    HF resumes from the checkpoint *directory*, every other framework
    from the file.
    """
    primaries = [c for c in result.checkpoints() if c["role"] == "primary"]
    assert primaries, f"parent run wrote nothing to resume from: {result.probe}"
    path = primaries[-1]["path"]
    return str(Path(path).parent) if framework == "huggingface" else path


def _export(result: CheckpointDriverResult, suffix: str) -> dict:
    exports = [
        c
        for c in result.checkpoints()
        if c["role"] == "export" and c["path"].endswith(suffix)
    ]
    assert exports, (
        f"no {suffix} export was written (saw: "
        f"{[(c['role'], c['path']) for c in result.checkpoints()]})"
    )
    return exports[0]


def _host_path(result: CheckpointDriverResult, checkpoint: dict) -> str:
    """Re-root a container path onto the copy the harness pulled back."""
    relative = Path(checkpoint["path"]).relative_to(result.probe["workdir"])
    return str(result.host_workdir / relative)


class TestDerivedCheckpointsKeepTheirOrigin:
    def test_two_logger_free_transforms_still_resolve_to_the_training_run(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
        stats_jsonl_path: Path,
        framework: str,
        run_checkpoint_driver: RunCheckpointFixture,
    ) -> None:
        """The GLUE-probe shape: train, then transform outside the run.

        Surgery and quantization are preprocessing steps, so no logger
        is live when they write. Before copy-forward the second hop saw
        a parent with no hash of its own, gave up, and produced a file
        indistinguishable from one trained from scratch — the eval then
        attached to nothing and the Eval tab was silently empty.

        Unit coverage of this monkeypatches the run registry. Here the
        run is closed for real and the hash has to survive two hops of
        genuine file I/O.
        """
        config = _config(
            testbed,
            framework,
            stats_jsonl_path,
            driver_flags={"TESTBED_DERIVE_CHAIN": "1"},
        )
        result = _run(run_checkpoint_driver, config)

        derivation = result.probe["derivation"]
        assert "error" not in derivation, derivation["error"]
        assert derivation["live_run_at_derive"] is None, (
            "a run was still registered while deriving, so inheritance was "
            "never exercised — this test would pass for the wrong reason"
        )

        run_hash = result.probe["run_hash"]
        assert get_metric_series(aim_repo, run_hash, "metric_0"), (
            f"origin run {run_hash!r} carries no metrics; inheriting it "
            f"would point evals at a run the dashboard cannot resolve"
        )

        hops = derivation["hops"]
        assert len(hops) == 2
        for depth, hop in enumerate(hops, start=1):
            meta = result.meta_of(hop)
            # .get, not []: to_dict() omits None fields, so a lost link
            # is an absent key. Indexing would raise KeyError and bury
            # the diagnostic.
            assert meta.get("aim_run_hash") == run_hash, (
                f"hop {depth} lost the origin run "
                f"(got {meta.get('aim_run_hash')!r}, want {run_hash!r})"
            )
            assert meta.get("derivation_chain_length") == depth
