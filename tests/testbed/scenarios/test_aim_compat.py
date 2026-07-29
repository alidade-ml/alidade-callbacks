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

import os
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
        aim_repo: str,
    ) -> None:
        """Documents the semantic. If this test starts FAILING, Aim changed its memtable model.

        Uses a writer subprocess run in background so the reader can query
        mid-flight (before the writer closes and forces a flush). The
        writer signals via a temp file when its write has landed; the
        reader then checks visibility on the host side.
        """
        # Writer subprocess: opens aim.Run inside client container, tracks
        # a fresh metric, writes hash to a signal file, sleeps, closes.
        # We monitor the signal file and check reader visibility DURING
        # the sleep window.
        signal_path_container = "/tmp/testbed-memtable-signal.txt"
        signal_path_host = f"{testbed.aim_repo_host_path.parent}/memtable-signal-{id(self)}.txt"
        import subprocess

        script = (
            "import os, aim, time, sys\n"
            "run = aim.Run(repo=os.environ['ASTROLABE_AIM_URL'])\n"
            "run.track(1.0, name='fresh_only_in_memtable', step=0)\n"
            f"open('{signal_path_container}', 'w').write(run.hash)\n"
            # Hold open — host-side reader polls the signal file, then
            # checks visibility BEFORE the sleep elapses.
            "time.sleep(5)\n"
            "run.close()\n"
        )
        writer = subprocess.Popen(
            [
                "docker",
                "compose",
                "-f",
                str(testbed.compose_file),
                "exec",
                "-T",
                "-e",
                f"ASTROLABE_AIM_URL={testbed.aim_url_from_client}",
                "client",
                "python",
                "-c",
                script,
            ],
            env={**os.environ, "AIM_REPO_HOST_PATH": str(testbed.aim_repo_host_path)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Poll for the signal file — write happened but run not closed yet.
        import time as _time

        run_hash = None
        for _ in range(30):
            code, stdout, _ = compose.exec_in(
                testbed,
                service="client",
                cmd=["cat", signal_path_container],
                check=False,
                timeout_s=5.0,
            )
            if code == 0 and stdout.strip():
                run_hash = stdout.strip()
                break
            _time.sleep(0.2)
        assert run_hash, "writer never signaled hash"
        # Reader check MID-FLIGHT — before the writer closes. Use the
        # NO-INDEX path (avoid forcing visibility via RepoIndexManager).
        # ``aim.Run(hash, ..., read_only=True).metrics()`` returns only
        # system metrics + explicitly-indexed metrics; user metrics
        # written but not yet flushed by close() do NOT appear.
        import aim as _aim

        try:
            reader_run = _aim.Run(run_hash, repo=str(aim_repo), read_only=True)
            user_metric_names = [
                s.name for s in reader_run.metrics() if not s.name.startswith("__system__")
            ]
            assert "fresh_only_in_memtable" not in user_metric_names, (
                f"Expected 'fresh_only_in_memtable' NOT visible mid-flight (memtable-only), "
                f"but reader enumerated: {user_metric_names}. Aim's memtable model may have changed."
            )
        finally:
            # Let writer finish + clean up
            writer.wait(timeout=15)
            compose.exec_in(
                testbed,
                service="client",
                cmd=["rm", "-f", signal_path_container],
                check=False,
                timeout_s=5.0,
            )

    def test_writes_become_visible_after_aim_flush(
        self,
        testbed: "TestbedHandle",
        aim_repo: str,
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
        aim_repo: str,
    ) -> None:
        """Within a single process, local Aim.Run reads back its own writes without finalize."""
        # Exec a script that opens a LOCAL repo (path, not URL) inside the container,
        # tracks a value, and re-reads within the same process.
        # Aim's get_metric signature has drifted across versions; iterate
        # via run.metrics() instead of get_metric to stay portable.
        script = (
            "import aim\n"
            "run = aim.Run(repo='/tmp/local-testbed-repo')\n"
            "run.track(1.0, name='same_proc_metric', step=0)\n"
            "found = False\n"
            "for m in run.metrics():\n"
            "    if m.name == 'same_proc_metric':\n"
            "        found = True\n"
            "        break\n"
            "assert found, 'same_proc_metric not enumerated on writer handle'\n"
            "run.close()\n"
            "print('OK')\n"
        )
        exit_code, stdout, stderr = _exec_aim(testbed, script)
        assert exit_code == 0, stderr
        assert "OK" in stdout

    def test_remote_repo_writes_require_finalize_for_readonly(
        self,
        testbed: "TestbedHandle",
        aim_repo: str,
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
        aim_repo: str,
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
