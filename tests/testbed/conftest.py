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
from pathlib import Path
from typing import Callable, Generator

import pytest

from tests.testbed.harness import compose
from tests.testbed.harness.compose import TestbedHandle
from tests.testbed.harness.driver import (
    DriverConfig,
    DriverResult,
    config_to_env as driver_config_to_env,
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
]


TESTBED_DIR = Path(__file__).parent
COMPOSE_FILE = TESTBED_DIR / "docker-compose.yml"

_RUN_HASH_RE = re.compile(r"ASTROLABE_RUN_HASH=([a-f0-9]{24})")
_EVAL_RUN_HASH_RE = re.compile(r"ASTROLABE_EVAL_RUN_HASH=([a-f0-9]{24})")
_EVAL_LINKED_RE = re.compile(r"ASTROLABE_EVAL_LINKED=(true|false)")


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
        # Container-side stats path is bind-mounted per exec; driver writes to
        # the path in `config.stats_jsonl_container_path` and we retrieve it
        # after via `docker cp`.
        exit_code, stdout, stderr = compose.exec_in(
            testbed,
            service="client",
            cmd=["python", "-m", "tests.testbed.harness.driver"],
            env=env,
            check=False,
            timeout_s=600.0,
        )
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

        return DriverResult(
            exit_code=exit_code,
            run_hash=run_hash,
            stats_events=stats_events,
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
