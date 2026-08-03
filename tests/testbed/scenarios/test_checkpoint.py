"""Integration tests for checkpoint provenance against a real Aim server.

Unit tests (``tests/test_checkpoint.py``) cover the meta block, format
sniffing, and the failure modes — all of which are answerable without
Aim. This file covers the three things that are NOT:

1. **Composer actually writes our block.** Unit tests can assert
   ``state_dict()`` returns the right dict; only a real Composer save
   proves Composer serializes it under our class qualname and replays
   it through ``load_state_dict`` on resume. The whole design rests on
   that behavior, and it's a framework contract we don't own.

2. **The embedded hash matches the run Aim actually opened.** The hash
   is minted at run-open, so a mock proves nothing. Mismatch here is
   silent — the eval attaches to a run that doesn't exist — which is
   exactly the class of bug the testbed exists for.

3. **Provenance survives schema-finalize.** ``maybe_finalize_schema``
   closes and reopens the Run mid-training. It already lost ``run.name``
   once (the v2.0.0-rc1 regression that broke CoLA probe). If a
   checkpoint written after a finalize carries a stale or empty hash,
   the same class of bug returns wearing a different hat.

Skips if the ``composer`` extra is not installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.testbed.harness.driver import DriverConfig

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


class TestCheckpointCarriesLiveRunIdentity:
    def test_embedded_hash_matches_the_opened_aim_run(self, testbed):
        """The join key the dashboard's Eval tab uses. If this drifts,
        evals attach to a run that was never created and the tab is
        silently empty."""
        raise NotImplementedError("stage3")

    def test_embedded_submit_id_is_full_fidelity(self, testbed):
        """Truncation belongs in log strings, never in a data channel.
        The stats jsonl already got this wrong once."""
        raise NotImplementedError("stage3")

    def test_propagated_identity_present_without_a_logger(self, testbed):
        """Checkpointer attached, logger absent: the stamp must still
        happen with submit_id / experiment / version from env, and
        aim_run_hash simply None."""
        raise NotImplementedError("stage3")


class TestComposerSerializesOurBlock:
    def test_block_lands_under_our_class_qualname(self, testbed):
        """Composer writes ``{type(cb).__qualname__: cb.state_dict()}``
        into state['callbacks']. We depend on that; it is Composer's
        contract, not ours, so it gets exercised rather than assumed."""
        raise NotImplementedError("stage3")

    def test_load_state_dict_replays_parent_on_resume(self, testbed):
        raise NotImplementedError("stage3")

    def test_resume_from_malformed_block_does_not_fail_training(self, testbed):
        """Provenance is never worth killing a run over."""
        raise NotImplementedError("stage3")


class TestSurvivesSchemaFinalize:
    def test_hash_still_correct_after_finalize(self, testbed):
        """maybe_finalize_schema closes + reopens the Run. It already
        dropped run.name once (v2.0.0-rc1). A checkpoint written after a
        finalize must carry the same hash as one written before."""
        raise NotImplementedError("stage3")

    def test_checkpoint_after_multiple_finalizes(self, testbed):
        raise NotImplementedError("stage3")


class TestExports:
    def test_safetensors_export_readable_by_safetensors_lib(self, testbed):
        """Our header write has to be legible to the real library, not
        just to our own reader."""
        raise NotImplementedError("stage3")

    def test_exported_copy_carries_same_identity_as_primary(self, testbed):
        raise NotImplementedError("stage3")


class TestFirstCheckpointMarker:
    def test_marker_touched_on_first_checkpoint(self, testbed):
        """Engine-side probe closes the `until: first_checkpoint`
        healing window on this file's existence."""
        raise NotImplementedError("stage3")

    def test_marker_not_rewritten_on_later_checkpoints(self, testbed):
        raise NotImplementedError("stage3")

    def test_training_survives_unwritable_marker_path(self, testbed):
        raise NotImplementedError("stage3")
