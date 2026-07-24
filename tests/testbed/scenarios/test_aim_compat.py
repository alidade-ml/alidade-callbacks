"""External-contract tests: Aim SDK behaviors we depend on.

**This file documents the Aim SDK's contract.** Not our workaround
around it — that lives in the test file for whichever module
implements the workaround (usually ``test_core.py``). Split rationale:

- If a test in this file fails, **Aim's contract has changed** and
  our workaround may no longer apply. STOP the SDK upgrade.
- If a workaround test in ``test_core.py`` fails, **our code broke**;
  Aim is still doing what it always did.

Different failure modes, different files, different diagnostic
signal on the next AI incident.
"""
from __future__ import annotations

from pathlib import Path

import pytest

if False:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


class TestProtobufMessageFactory:
    """Regression: the ``MessageFactory.GetPrototype`` protobuf incompat.

    Historical bug: certain Aim SDK versions imported protobuf in a way
    that broke on newer protobuf releases. Symptom was AttributeError at
    Run() construction time.
    """

    def test_run_construction_does_not_raise(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """AstrolabeLogger.init() completes without protobuf-related errors."""
        raise NotImplementedError


class TestMemtableContract:
    """Aim's memtable/flush contract — the property our schema-finalize exploits.

    A read-only Aim Repo opened separately from the writing process sees
    only what's been flushed to SST. Metrics written but not flushed are
    invisible. This pair of assertions documents that contract. Our
    workaround (schema-finalize forcing a flush) is verified in
    ``test_core.py::TestSchemaFinalize::test_finalize_flushes_new_metric_to_disk``.
    """

    def test_unflushed_writes_are_invisible_to_readonly_reader(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Documents the semantic. If this test starts FAILING, Aim changed its memtable model."""
        raise NotImplementedError

    def test_writes_become_visible_after_aim_flush(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """When Aim flushes (however triggered), reader sees the metrics. The counterpart to the invisibility property."""
        raise NotImplementedError


class TestLocalVsRemoteRepoBehavior:
    """Document the semantic diff between local Aim.Run and remote (aim://).

    Local mode: writes hit disk immediately, then may or may not flush.
    Remote mode: writes go over the network to the aim server, which
    manages its own flush cadence.

    Scenarios that need to assert on-disk state must know which mode
    they're in.
    """

    def test_local_repo_writes_visible_to_same_process(
        self, aim_repo: Path
    ) -> None:
        """Within a single process, local Aim.Run reads back its own writes without finalize."""
        raise NotImplementedError

    def test_remote_repo_writes_require_finalize_for_readonly(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Remote Aim.Run writes NOT visible to a separate read-only Repo until finalize."""
        raise NotImplementedError


class TestSDKVersionSurface:
    """Pin critical attributes the callback library depends on.

    If any of these attributes disappear or change signature in an Aim
    upgrade, callback code will break at runtime. Fail fast here so we
    catch it before releasing against a new Aim pin.
    """

    def test_run_has_name_attribute(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        raise NotImplementedError

    def test_run_track_signature_stable(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Run.track(value, name, step, context) accepts our call shape."""
        raise NotImplementedError

    def test_run_close_flushes(
        self, testbed: "TestbedHandle", aim_repo: Path
    ) -> None:
        """Run.close() actually flushes; downstream state visible after return."""
        raise NotImplementedError
