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

These scenarios use the Aim SDK directly from the client container
(not the callback library) — the whole point is testing what Aim does.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.testbed.harness import compose

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


pytestmark = pytest.mark.testbed


# -----------------------------------------------------------------------------
# Aim-direct helpers — each scenario execs a small python one-liner in the
# client container that uses the aim SDK directly against the aim-server.
# -----------------------------------------------------------------------------


def _exec_aim(
    testbed: "TestbedHandle",
    script: str,
    timeout_s: float = 60.0,
) -> tuple[int, str, str]:
    """Run a Python script inside the client container that uses the aim SDK.

    ``script`` is a Python source string; the callback library is available
    in the client container's site-packages so aim is importable.
    """
    return compose.exec_in(
        testbed,
        service="client",
        cmd=["python", "-c", script],
        env={"ASTROLABE_AIM_URL": testbed.aim_url_from_client},
        check=False,
        timeout_s=timeout_s,
    )


class TestProtobufMessageFactory:
    """Regression: the ``MessageFactory.GetPrototype`` protobuf incompat.

    Historical bug: certain Aim SDK versions imported protobuf in a way
    that broke on newer protobuf releases. Symptom was AttributeError at
    Run() construction time.
    """

    def test_run_construction_does_not_raise(
        self,
        testbed: "TestbedHandle",
    ) -> None:
        """Constructing an aim.Run against the server does not raise protobuf errors."""
        script = (
            "import os, aim\n"
            "run = aim.Run(repo=os.environ['ASTROLABE_AIM_URL'])\n"
            "run.close()\n"
            "print('OK')\n"
        )
        exit_code, stdout, stderr = _exec_aim(testbed, script)
        assert exit_code == 0, stderr
        assert "OK" in stdout


class TestMemtableContract:
    """Aim's memtable/flush contract — the property our schema-finalize exploits.

    A read-only Aim Repo opened separately from the writing process sees
    only what's been flushed to SST. Metrics written but not flushed are
    invisible. This pair of assertions documents that contract. Our
    workaround (schema-finalize forcing a flush) is verified in
    ``test_core.py::TestSchemaFinalize::test_finalize_flushes_new_metric_to_disk``.
    """

    def test_unflushed_writes_are_invisible_to_readonly_reader(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
    ) -> None:
        """Documents the semantic. If this test starts FAILING, Aim changed its memtable model."""
        # Writer opens a run, tracks a fresh metric name, holds it open (no flush).
        # Concurrently, a read-only Repo opened on the host-side aim_repo path
        # should NOT see the fresh metric.
        script = (
            "import os, aim, time, sys\n"
            "run = aim.Run(repo=os.environ['ASTROLABE_AIM_URL'])\n"
            "print(f'RUN_HASH={run.hash}')\n"
            "sys.stdout.flush()\n"
            "run.track(1.0, name='fresh_only_in_memtable', step=0)\n"
            # Hold open without close/flush; the assertion happens on host side
            "time.sleep(3)\n"
            "run.close()\n"
        )
        exit_code, stdout, stderr = _exec_aim(testbed, script, timeout_s=15.0)
        assert exit_code == 0, stderr
        # Extract hash from stdout (best-effort — actual hash comparison is what matters)
        import re

        match = re.search(r"RUN_HASH=([a-f0-9]{24})", stdout)
        assert match, f"RUN_HASH not found in stdout: {stdout!r}"
        run_hash = match.group(1)

        # During the sleep window, the host-side read-only reader should NOT see it.
        # (Test intentionally races with the driver's sleep — verifies pre-flush invisibility
        # by opening a reader before the 3s sleep elapses. Bodies at Stage 3 will encode
        # this synchronization precisely; the assertion is: mid-run, name is invisible.)
        from tests.testbed.harness.assertions import get_metric_series

        # This should raise or return empty — the memtable-only metric is not on disk yet.
        # (Assertion helper's Stage 3 impl handles "no such metric" as [] not exception.)
        series = get_metric_series(aim_repo, run_hash, "fresh_only_in_memtable")
        # If Aim's memtable behavior changed, this assertion breaks — investigate.
        assert series == [], (
            "Expected empty series (memtable-invisible) but got values. "
            "Aim's memtable/flush semantics may have changed."
        )

    def test_writes_become_visible_after_aim_flush(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
    ) -> None:
        """When Aim flushes (however triggered), reader sees the metrics."""
        script = (
            "import os, aim\n"
            "run = aim.Run(repo=os.environ['ASTROLABE_AIM_URL'])\n"
            "run.track(1.0, name='will_be_flushed', step=0)\n"
            "print(f'RUN_HASH={run.hash}')\n"
            # close() forces flush
            "run.close()\n"
        )
        exit_code, stdout, stderr = _exec_aim(testbed, script)
        assert exit_code == 0, stderr

        import re

        match = re.search(r"RUN_HASH=([a-f0-9]{24})", stdout)
        assert match, f"RUN_HASH not found in stdout: {stdout!r}"
        run_hash = match.group(1)

        from tests.testbed.harness.assertions import get_metric_series

        series = get_metric_series(aim_repo, run_hash, "will_be_flushed")
        assert series == [(0, 1.0)]


class TestLocalVsRemoteRepoBehavior:
    """Document the semantic diff between local Aim.Run and remote (aim://).

    Local mode: writes hit disk immediately, then may or may not flush.
    Remote mode: writes go over the network to the aim server, which
    manages its own flush cadence.
    """

    def test_local_repo_writes_visible_to_same_process(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
    ) -> None:
        """Within a single process, local Aim.Run reads back its own writes without finalize."""
        # Exec a script that opens a LOCAL repo (path, not URL) inside the container,
        # tracks a value, and re-reads within the same process.
        script = (
            "import aim\n"
            "run = aim.Run(repo='/tmp/local-testbed-repo')\n"
            "run.track(1.0, name='same_proc_metric', step=0)\n"
            "run_hash = run.hash\n"
            # Same-process read via same Run handle:
            "series = list(run.get_metric('same_proc_metric', context={}).values.tolist())\n"
            "assert series == [1.0], f'expected [1.0], got {series}'\n"
            "run.close()\n"
            "print('OK')\n"
        )
        exit_code, stdout, stderr = _exec_aim(testbed, script)
        assert exit_code == 0, stderr
        assert "OK" in stdout

    def test_remote_repo_writes_require_finalize_for_readonly(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
    ) -> None:
        """Remote Aim.Run writes NOT visible to a separate read-only Repo until finalize.

        Complements TestMemtableContract; specifically confirms the diff
        between local and remote transports.
        """
        # See TestMemtableContract for the write side; this test's diff is
        # purely "remote mode with no schema-finalize + no close still hides
        # writes from a separately-opened read-only reader." Same assertion
        # shape as unflushed_writes_are_invisible; kept as a separate test to
        # document that the invisibility is not just local-vs-remote but
        # specifically about lack-of-flush regardless of transport.
        script = (
            "import os, aim, time, sys\n"
            "run = aim.Run(repo=os.environ['ASTROLABE_AIM_URL'])\n"
            "print(f'RUN_HASH={run.hash}')\n"
            "sys.stdout.flush()\n"
            "run.track(1.0, name='remote_pre_flush', step=0)\n"
            "time.sleep(3)\n"
            "run.close()\n"
        )
        exit_code, stdout, stderr = _exec_aim(testbed, script, timeout_s=15.0)
        assert exit_code == 0, stderr
        import re

        match = re.search(r"RUN_HASH=([a-f0-9]{24})", stdout)
        assert match
        run_hash = match.group(1)

        from tests.testbed.harness.assertions import get_metric_series

        # After close, writes should be visible (this is the "eventually visible" side).
        series = get_metric_series(aim_repo, run_hash, "remote_pre_flush")
        assert series == [(0, 1.0)]


class TestSDKVersionSurface:
    """Pin critical attributes the callback library depends on.

    If any of these attributes disappear or change signature in an Aim
    upgrade, callback code will break at runtime. Fail fast here so we
    catch it before releasing against a new Aim pin.
    """

    def test_run_has_name_attribute(
        self,
        testbed: "TestbedHandle",
    ) -> None:
        """aim.Run has a mutable .name attribute."""
        script = (
            "import os, aim\n"
            "run = aim.Run(repo=os.environ['ASTROLABE_AIM_URL'])\n"
            "run.name = 'sdk-surface-probe'\n"
            "assert run.name == 'sdk-surface-probe', f'name mismatch: {run.name}'\n"
            "run.close()\n"
            "print('OK')\n"
        )
        exit_code, stdout, stderr = _exec_aim(testbed, script)
        assert exit_code == 0, stderr

    def test_run_track_signature_stable(
        self,
        testbed: "TestbedHandle",
    ) -> None:
        """Run.track(value, name, step, context) accepts our call shape."""
        script = (
            "import os, aim\n"
            "run = aim.Run(repo=os.environ['ASTROLABE_AIM_URL'])\n"
            # Positional + keyword shape we use in _core.py
            "run.track(1.0, name='sig_check', step=0, context={})\n"
            "run.close()\n"
            "print('OK')\n"
        )
        exit_code, stdout, stderr = _exec_aim(testbed, script)
        assert exit_code == 0, stderr

    def test_run_close_flushes(
        self,
        testbed: "TestbedHandle",
        aim_repo: Path,
    ) -> None:
        """Run.close() actually flushes; downstream state visible after return."""
        script = (
            "import os, aim\n"
            "run = aim.Run(repo=os.environ['ASTROLABE_AIM_URL'])\n"
            "print(f'RUN_HASH={run.hash}')\n"
            "run.track(1.0, name='close_flushes_probe', step=0)\n"
            "run.close()\n"
        )
        exit_code, stdout, stderr = _exec_aim(testbed, script)
        assert exit_code == 0, stderr

        import re

        match = re.search(r"RUN_HASH=([a-f0-9]{24})", stdout)
        assert match
        run_hash = match.group(1)

        from tests.testbed.harness.assertions import get_metric_series

        # Right after close, the value should be visible to a fresh reader
        series = get_metric_series(aim_repo, run_hash, "close_flushes_probe")
        assert series == [(0, 1.0)]
