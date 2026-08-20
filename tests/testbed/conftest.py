"""Testbed fixtures — docker-compose lifecycle + driver invocation helpers.

Session-scope ``testbed`` brings up two containers (aim server + client)
via docker-compose. Function-scope ``aim_repo`` returns the host-side
path where the aim server is writing, so per-test assertions can read
what landed without touching the containers.

Function-scope ``run_driver`` and ``run_eval_driver`` fixtures wrap the
compose.exec_in dance so scenarios stay concise: build a config, call
the fixture, get a result with (exit_code, run_hash, stats).

Tests isolate on run hashes / experiment names, not on repo state. If
scenarios need cleaner isolation, call ``compose.reset_repo(testbed)``
in a per-test fixture.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator

import pytest

from tests.testbed.harness import compose
from tests.testbed.harness.compose import TestbedHandle
from tests.testbed.harness.sample_driver import (
    SampleDriverConfig,
    sample_config_to_env,
)
from tests.testbed.harness.driver import (
    DriverConfig,
    DriverResult,
    config_to_env as driver_config_to_env,
)
from tests.testbed.harness.checkpoint_driver import (
    PROBE_PREFIX,
    CheckpointDriverConfig,
    CheckpointDriverResult,
    config_to_env as checkpoint_config_to_env,
)
from tests.testbed.harness.eval_driver import (
    EvalDriverConfig,
    EvalDriverResult,
    config_to_env as eval_config_to_env,
)


__all__ = [
    "testbed",
    "aim_repo",
    "stats_jsonl_path",
    "run_driver",
    "run_eval_driver",
    "run_sample_driver",
    "run_checkpoint_driver",
]


TESTBED_DIR = Path(__file__).parent
COMPOSE_FILE = TESTBED_DIR / "docker-compose.yml"

_RUN_HASH_RE = re.compile(r"ASTROLABE_RUN_HASH=([a-f0-9]{24})")
_EVAL_RUN_HASH_RE = re.compile(r"ASTROLABE_EVAL_RUN_HASH=([a-f0-9]{24})")
_EVAL_LINKED_RE = re.compile(r"ASTROLABE_EVAL_LINKED=(true|false)")
_SAMPLE_RUN_HASH_RE = re.compile(r"ASTROLABE_SAMPLE_RUN_HASH=([a-f0-9]{24})")
_MARKER_TOUCHED_RE = re.compile(r"ASTROLABE_MARKER_TOUCHED=(true|false)")


@pytest.fixture(scope="session")
def testbed(tmp_path_factory: pytest.TempPathFactory) -> Generator[TestbedHandle, None, None]:
    """Session-scope docker-compose testbed.

    Fast-iteration knob: ``TESTBED_KEEP=1`` skips teardown so subsequent
    pytest invocations reuse the running stack. Combined with
    ``compose.up`` also skipping ``--build`` when this flag is set,
    typical iteration drops from ~60s per pytest run to ~5-15s.

    Repo dir: when ``TESTBED_KEEP=1``, pins to a stable location so
    successive runs share the same repo; scenarios stay isolated by
    unique run_name / experiment_name / stats_jsonl_path. Otherwise
    mints a fresh tmp dir per session (default).
    """
    import os as _os

    if _os.environ.get("TESTBED_KEEP") == "1":
        # Stable repo path across invocations. Scenarios isolate via
        # unique run identifiers, not repo state.
        aim_repo_dir = Path("/tmp/testbed-aim-repo-persistent")
        aim_repo_dir.mkdir(exist_ok=True)
    else:
        aim_repo_dir = tmp_path_factory.mktemp("aim-repo")

    handle = compose.up(COMPOSE_FILE, aim_repo_dir)
    try:
        yield handle
    finally:
        if _os.environ.get("TESTBED_KEEP") != "1":
            compose.down(handle)


@pytest.fixture
def aim_repo(testbed: TestbedHandle) -> Path:
    """Host-side aim repo path — same directory the aim server writes to."""
    return testbed.aim_repo_host_path


@pytest.fixture
def stats_jsonl_path(testbed: TestbedHandle, tmp_path: Path) -> Path:
    """Host-visible path where the client container writes stats jsonl.

    Bind-mounted from a per-test tmp_path so each scenario gets a clean
    file. The client container sees this at ``/host-stats/<name>`` via a
    docker exec-time bind (docker-compose can't bind per-test paths at
    startup, so we mount tmp_path via ``docker cp`` after each test's
    driver invocation instead — see run_driver below).

    For the assertion side, this fixture just returns the host path; the
    driver invocation is responsible for populating it.
    """
    return tmp_path / "callback-stats.jsonl"


def _parse_run_hash(stdout: str, pattern: re.Pattern) -> str | None:
    match = pattern.search(stdout)
    return match.group(1) if match else None


def _watch_and_restart_aim(testbed: TestbedHandle, stop: threading.Event) -> None:
    """Poll the client container for the driver's restart-signal file.

    Once seen, ``docker restart`` the aim container and wait for it to
    accept connections again. Idempotent: fires at most once per
    invocation. Silently exits if ``stop`` is set before the signal
    ever appears (driver finished without triggering restart).
    """
    signal_path = "/tmp/testbed-restart-signal.txt"
    poll_deadline = time.monotonic() + 120.0
    while not stop.is_set() and time.monotonic() < poll_deadline:
        code, _, _ = compose.exec_in(
            testbed,
            service="client",
            cmd=["test", "-f", signal_path],
            check=False,
            timeout_s=5.0,
        )
        if code == 0:
            break
        time.sleep(0.1)
    else:
        return
    if stop.is_set():
        return

    subprocess.run(
        ["docker", "restart", testbed.aim_container],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Wait for aim to be accepting connections + actually serving.
    # ``wait_healthy`` uses the same probe the initial startup relied
    # on; it returns as soon as an aim.Run open/close round-trip
    # succeeds from the client container. The drainer's next retry
    # after this point should succeed.
    compose.wait_healthy(testbed, timeout_s=60.0)


def _read_stats_events(host_path: Path) -> list[dict]:
    if not host_path.exists():
        return []
    events: list[dict] = []
    with host_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


@pytest.fixture
def run_driver(
    testbed: TestbedHandle,
    stats_jsonl_path: Path,
) -> Callable[[DriverConfig], DriverResult]:
    """Return a callable that runs the driver inside the client container."""

    def _run(config: DriverConfig) -> DriverResult:
        env = driver_config_to_env(config)
        # Clear the container-side stats path before invocation. Under
        # TESTBED_KEEP=1 the container persists across pytest sessions
        # and the same per-test tmp-path basename can recur, so
        # residual events would accumulate. rm -f is safe (missing
        # file is fine).
        compose.exec_in(
            testbed,
            service="client",
            cmd=["rm", "-f", config.stats_jsonl_container_path],
            check=False,
            timeout_s=10.0,
        )
        # Aim-server restart hook (TESTBED_RESTART_AIM_AT). Driver writes
        # /tmp/testbed-restart-signal.txt inside the client container at
        # step N. We poll for it here on the host and issue docker
        # restart on the aim container once it appears. The container's
        # bridge network + restart="no" means the container comes back
        # under the same name and rejoins the same network.
        restart_stop = threading.Event()
        restart_thread: threading.Thread | None = None
        if "TESTBED_RESTART_AIM_AT" in config.driver_flags:
            # Clear any stale signal from a prior test.
            compose.exec_in(
                testbed,
                service="client",
                cmd=["rm", "-f", "/tmp/testbed-restart-signal.txt"],
                check=False,
                timeout_s=10.0,
            )
            restart_thread = threading.Thread(
                target=_watch_and_restart_aim,
                args=(testbed, restart_stop),
                daemon=True,
            )
            restart_thread.start()

        # Container-side stats path is bind-mounted per exec; driver writes to
        # the path in `config.stats_jsonl_container_path` and we retrieve it
        # after via `docker cp`.
        try:
            exit_code, stdout, stderr = compose.exec_in(
                testbed,
                service="client",
                cmd=["python", "-m", "tests.testbed.harness.driver"],
                env=env,
                check=False,
                timeout_s=600.0,
            )
        finally:
            restart_stop.set()
            if restart_thread is not None:
                restart_thread.join(timeout=10.0)
        run_hash = _parse_run_hash(stdout, _RUN_HASH_RE)

        # Copy container-side stats jsonl back to host tmp_path for assertions.
        container_stats = config.stats_jsonl_container_path.lstrip("/")
        try:
            import subprocess

            subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{testbed.client_container}:/{container_stats}",
                    str(stats_jsonl_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            # Stats file may not exist if driver died early — that's OK, tests
            # that assert on stats will fail with a clear message.
            pass

        stats_events = _read_stats_events(stats_jsonl_path)

        marker_match = _MARKER_TOUCHED_RE.search(stdout)
        marker_touched = marker_match.group(1) == "true" if marker_match else False

        return DriverResult(
            exit_code=exit_code,
            run_hash=run_hash,
            stats_events=stats_events,
            stdout=stdout,
            stderr=stderr,
            marker_touched=marker_touched,
        )

    return _run


@pytest.fixture
def run_checkpoint_driver(
    testbed: TestbedHandle,
    stats_jsonl_path: Path,
    tmp_path: Path,
) -> Callable[[CheckpointDriverConfig], CheckpointDriverResult]:
    """Run the checkpoint driver in the client container and pull its output back.

    Copies the driver's whole workdir to the host so scenarios can read
    the written checkpoints with the real ``safetensors`` /
    ``transformers`` libraries rather than only through the reader whose
    correctness is under test.
    """

    def _run(config: CheckpointDriverConfig) -> CheckpointDriverResult:
        for path in (config.workdir, config.stats_jsonl_container_path):
            compose.exec_in(
                testbed,
                service="client",
                cmd=["rm", "-rf", path],
                check=False,
                timeout_s=30.0,
            )
        exit_code, stdout, stderr = compose.exec_in(
            testbed,
            service="client",
            cmd=["python", "-m", "tests.testbed.harness.checkpoint_driver"],
            env=checkpoint_config_to_env(config),
            check=False,
            timeout_s=900.0,
        )
        host_workdir = tmp_path / "ckpt-workdir"
        _copy_from_container(testbed, config.workdir, host_workdir)
        _copy_from_container(
            testbed, config.stats_jsonl_container_path, stats_jsonl_path
        )
        return CheckpointDriverResult(
            exit_code=exit_code,
            probe=_parse_probe(stdout),
            stdout=stdout,
            stderr=stderr,
            stats_events=_read_stats_events(stats_jsonl_path),
            host_workdir=host_workdir,
        )

    return _run


def _copy_from_container(
    testbed: TestbedHandle, container_path: str, host_path: Path
) -> None:
    host_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "docker",
            "cp",
            f"{testbed.client_container}:{container_path}",
            str(host_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _parse_probe(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith(PROBE_PREFIX):
            return json.loads(line[len(PROBE_PREFIX) :])
    return {}


@dataclass
class SampleDriverResult:
    exit_code: int
    sample_run_hash: str | None
    stdout: str
    stderr: str


@pytest.fixture
def run_sample_driver(
    testbed: TestbedHandle,
) -> Callable[[SampleDriverConfig], SampleDriverResult]:
    """Return a callable that runs the sample driver inside the client container."""

    def _run(config: SampleDriverConfig) -> SampleDriverResult:
        exit_code, stdout, stderr = compose.exec_in(
            testbed,
            service="client",
            cmd=["python", "-m", "tests.testbed.harness.sample_driver"],
            env=sample_config_to_env(config),
            check=False,
            timeout_s=120.0,
        )
        return SampleDriverResult(
            exit_code=exit_code,
            sample_run_hash=_parse_run_hash(stdout, _SAMPLE_RUN_HASH_RE),
            stdout=stdout,
            stderr=stderr,
        )

    return _run


@pytest.fixture
def run_eval_driver(
    testbed: TestbedHandle,
) -> Callable[[EvalDriverConfig], EvalDriverResult]:
    """Return a callable that runs the eval driver inside the client container."""

    def _run(config: EvalDriverConfig) -> EvalDriverResult:
        env = eval_config_to_env(config)
        exit_code, stdout, stderr = compose.exec_in(
            testbed,
            service="client",
            cmd=["python", "-m", "tests.testbed.harness.eval_driver"],
            env=env,
            check=False,
            timeout_s=120.0,
        )
        eval_hash = _parse_run_hash(stdout, _EVAL_RUN_HASH_RE)
        linked_match = _EVAL_LINKED_RE.search(stdout)
        linked = linked_match.group(1) == "true" if linked_match else False

        return EvalDriverResult(
            exit_code=exit_code,
            eval_run_hash=eval_hash,
            linked=linked,
            stdout=stdout,
            stderr=stderr,
        )

    return _run
