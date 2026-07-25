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

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator

import pytest

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle
    from tests.testbed.harness.driver import DriverConfig, DriverResult
    from tests.testbed.harness.eval_driver import (
        EvalDriverConfig,
        EvalDriverResult,
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


@pytest.fixture(scope="session")
def testbed(tmp_path_factory: pytest.TempPathFactory) -> Generator["TestbedHandle", None, None]:
    """Session-scope docker-compose testbed.

    Brings both containers up once for the whole testbed run, tears them
    down at session exit. Scenarios talk to the containers via
    ``compose.exec_in(testbed, service="client", cmd=[...])``.
    """
    raise NotImplementedError


@pytest.fixture
def aim_repo(testbed: "TestbedHandle") -> Path:
    """Host-side aim repo path.

    Same directory the aim server container is writing to (via bind
    mount). Assertions read this via ``aim.Repo(path, read_only=True)``.

    Not per-test isolated — tests should use unique run hashes /
    experiment names for isolation. If a scenario needs a clean repo,
    it can call ``compose.reset_repo(testbed)`` explicitly.
    """
    raise NotImplementedError


@pytest.fixture
def stats_jsonl_path(testbed: "TestbedHandle", tmp_path: Path) -> Path:
    """Host-visible path where the client container writes stats jsonl.

    Bind-mounted from a per-test tmp_path so each scenario gets a clean
    file. Driver scripts inside the client container write to the same
    path via a container-side mount.
    """
    raise NotImplementedError


@pytest.fixture
def run_driver(
    testbed: "TestbedHandle",
    stats_jsonl_path: Path,
) -> Callable[["DriverConfig"], "DriverResult"]:
    """Return a callable that runs the driver inside the client container.

    Serializes ``config`` to env vars, invokes
    ``python -m tests.testbed.harness.driver`` inside the client
    container via ``compose.exec_in``, parses stdout for the Aim run hash
    marker, reads and parses the stats jsonl. Returns a ``DriverResult``.

    For ``framework="raw"`` the driver directly exercises AstrolabeRun.
    For ``framework="composer" | "lightning" | "hf"`` the driver runs
    real (tiny, CPU, seeded) training with the corresponding
    framework's Trainer and AstrolabeCallback.
    """
    raise NotImplementedError


@pytest.fixture
def run_eval_driver(
    testbed: "TestbedHandle",
) -> Callable[["EvalDriverConfig"], "EvalDriverResult"]:
    """Return a callable that runs the eval driver inside the client container.

    Same pattern as ``run_driver`` but for the eval-helper surface.
    Returns an ``EvalDriverResult`` with the eval run's Aim hash + parent
    linkage state (linked / warn / raise, per the on_missing_parent
    setting on the eval config).
    """
    raise NotImplementedError
