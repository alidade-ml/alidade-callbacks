"""Testbed fixtures — docker-compose lifecycle.

Session-scope ``testbed`` brings up two containers (aim server + client)
via docker-compose. Function-scope ``aim_repo`` returns the host-side
path where the aim server is writing, so per-test assertions can read
what landed without touching the containers.

Tests isolate on run hashes / experiment names, not on repo state. If
scenarios need cleaner isolation, call ``compose.reset_repo(testbed)``
in a per-test fixture.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import pytest

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle


__all__ = ["testbed", "aim_repo", "stats_jsonl_path"]


TESTBED_DIR = Path(__file__).parent
COMPOSE_FILE = TESTBED_DIR / "docker-compose.yml"


@pytest.fixture(scope="session")
def testbed(tmp_path_factory: pytest.TempPathFactory) -> Generator["TestbedHandle", None, None]:
    """Session-scope docker-compose testbed.

    Brings both containers up once for the whole testbed run, tears them
    down at session exit. Scenarios talk to the containers via
    ``compose.exec_in(testbed, service="client", cmd=[...])``.

    Yields
    ------
    TestbedHandle
        Live handle covering container names, aim URLs, and the
        host-side bind-mount path for the aim repo.
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
    file. Mock training scripts inside the client container write to
    the same path via a container-side mount.
    """
    raise NotImplementedError
