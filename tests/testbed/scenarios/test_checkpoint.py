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

**HuggingFace is deliberately absent.** ``TrainerCallback`` exposes no
checkpoint-dict hook at all, so HF needs a different mechanism entirely
(monkey-patching ``Trainer._save`` or a post-save header rewrite) — an
unresolved design question, not an oversight. Adding it here would mean
designing that mechanism inside a test file. Tracked as the next
checkpointer to land; see the PR description.

Each framework skips if its extra is not installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


# Mechanism differs per framework; behavior must not. Any test taking
# this fixture asserts a cross-framework contract.
FRAMEWORKS = ["composer", "lightning", "pytorch"]


@pytest.fixture(params=FRAMEWORKS)
def framework(request):
    """Drives one training run per framework with its checkpointer
    attached. Skips when the framework's extra is absent."""
    pytest.importorskip(
        {"composer": "composer", "lightning": "lightning", "pytorch": "torch"}[
            request.param
        ]
    )
    return request.param


class TestCheckpointCarriesLiveRunIdentity:
    def test_embedded_hash_matches_the_opened_aim_run(self, testbed, framework):
        """The join key the dashboard's Eval tab uses. If it drifts, evals
        attach to a run that was never created and the tab is silently
        empty."""
        raise NotImplementedError("stage3")

    def test_embedded_submit_id_is_full_fidelity(self, testbed, framework):
        """Truncation belongs in log strings, never a data channel. The
        stats jsonl already got this wrong once."""
        raise NotImplementedError("stage3")

    def test_propagated_identity_present_without_a_logger(self, testbed, framework):
        """Checkpointer attached, logger absent: the stamp still happens
        from env, with aim_run_hash simply None."""
        raise NotImplementedError("stage3")


class TestFrameworkSerializesOurBlock:
    """The mechanism is per-framework; that it works is not optional for
    any of them."""

    def test_block_round_trips_through_a_real_save(self, testbed, framework):
        raise NotImplementedError("stage3")

    def test_resume_replays_parent_provenance(self, testbed, framework):
        raise NotImplementedError("stage3")

    def test_resume_from_malformed_block_does_not_fail_training(
        self, testbed, framework
    ):
        """Provenance is never worth killing a run over."""
        raise NotImplementedError("stage3")

    def test_composer_block_lands_under_our_class_qualname(self, testbed):
        """Composer-specific: it keys callback state by
        ``type(obj).__qualname__``. Renaming our class would silently
        orphan every previously-written checkpoint."""
        pytest.importorskip("composer")
        raise NotImplementedError("stage3")

    def test_lightning_mutation_survives_its_own_hooks(self, testbed):
        """Lightning-specific: other callbacks also mutate the same dict
        in ``on_save_checkpoint``. Ours must not be clobbered by, or
        clobber, a co-attached callback."""
        pytest.importorskip("lightning")
        raise NotImplementedError("stage3")


class TestSurvivesSchemaFinalize:
    def test_hash_still_correct_after_finalize(self, testbed, framework):
        """maybe_finalize_schema closes + reopens the Run. It already
        dropped run.name once (v2.0.0-rc1). A checkpoint written after a
        finalize must carry the same hash as one written before."""
        raise NotImplementedError("stage3")

    def test_checkpoint_after_multiple_finalizes(self, testbed, framework):
        raise NotImplementedError("stage3")


class TestExports:
    def test_safetensors_export_readable_by_safetensors_lib(self, testbed, framework):
        """Our header write has to be legible to the real library, not
        just to our own reader."""
        pytest.importorskip("safetensors")
        raise NotImplementedError("stage3")

    def test_exported_copy_carries_same_identity_as_primary(self, testbed, framework):
        raise NotImplementedError("stage3")


class TestFirstCheckpointMarker:
    def test_marker_touched_on_first_checkpoint(self, testbed, framework):
        """Engine-side probe closes the `until: first_checkpoint` healing
        window on this file's existence."""
        raise NotImplementedError("stage3")

    def test_marker_not_rewritten_on_later_checkpoints(self, testbed, framework):
        raise NotImplementedError("stage3")

    def test_training_survives_unwritable_marker_path(self, testbed, framework):
        raise NotImplementedError("stage3")
