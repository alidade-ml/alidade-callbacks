"""Docker-compose lifecycle for the callback testbed.

Two containers on a shared bridge network:

* ``aim-server`` — runs ``aim server`` (simulates the NUC's Aim endpoint)
* ``client`` — python env with the callback source bind-mounted from the
  repo; pytest invokes mock training/eval scripts inside this container
  via ``exec_in``. Simulates a compute host.

Bridge network gives real TCP between callback (client) and Aim (server) —
one step closer to production than subprocess-on-loopback would give.
The aim repo is bind-mounted from a per-session host directory so
host-side assertions can read what the server wrote directly.

Same design shape as ``astrolabe/tests/testbed/harness/compose.py``;
scenarios that graduate from one testbed to the other should feel
familiar.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "TestbedHandle",
    "up",
    "down",
    "wait_healthy",
    "reset_repo",
    "logs",
    "exec_in",
]


@dataclass
class TestbedHandle:
    """Live handle to a running callback testbed.

    Attributes
    ----------
    compose_file : pathlib.Path
        Path to the docker-compose.yml this instance runs against.
    aim_container : str
        Full container name of the aim server service.
    client_container : str
        Full container name of the client service.
    aim_repo_host_path : pathlib.Path
        Host-side path bind-mounted into the aim container at
        ``/var/lib/aim``. Host-side assertions read this directory via
        ``aim.Repo(path, read_only=True)``.
    aim_url_from_client : str
        The URL clients inside the ``client`` container use to reach the
        aim server. Bridge network resolves service names, so this is
        ``aim://aim-server:43800``.
    aim_url_from_host : str
        Optional URL for host-side connections to the aim server (via
        published port). ``aim://localhost:<published_port>``. Used by
        assertions that prefer connecting over the aim SDK rather than
        reading the repo directly.
    """

    compose_file: Path
    aim_container: str
    client_container: str
    aim_repo_host_path: Path
    aim_url_from_client: str
    aim_url_from_host: str


def up(compose_file: Path, aim_repo_host_path: Path, timeout_s: float = 60.0) -> TestbedHandle:
    """Bring the testbed containers up and return a live handle.

    Runs ``docker compose -f <compose_file> up -d`` after exporting
    ``AIM_REPO_HOST_PATH`` so the aim container bind-mounts to the
    caller-supplied host directory (typically a pytest tmp_path).

    Blocks until both containers are healthy (aim server accepting TCP
    connections, client container reports ready). Raises
    ``TestbedStartupError`` on timeout or docker-compose failure.
    """
    raise NotImplementedError


def down(handle: TestbedHandle, purge_volumes: bool = True) -> None:
    """Tear down the testbed containers.

    ``purge_volumes=True`` runs ``docker compose down -v`` so any
    docker-managed volumes are wiped. The bind-mounted aim repo is
    host-owned so this doesn't touch it — teardown of the host-side
    tmp_path is the fixture's responsibility.
    """
    raise NotImplementedError


def wait_healthy(handle: TestbedHandle, timeout_s: float = 30.0) -> None:
    """Poll container readiness until both aim + client report healthy.

    Aim readiness: TCP connect to the aim server's published port.
    Client readiness: ``docker compose exec client python -c ...`` returns
    successfully.
    """
    raise NotImplementedError


def reset_repo(handle: TestbedHandle) -> None:
    """Purge the aim repo contents on disk without restarting containers.

    Used between scenarios in a session-scope testbed to keep test
    isolation without paying compose-restart cost per test. Aim server
    reopens the repo lazily on next write.

    Alternative pattern: tests use unique run hashes/experiment names
    for isolation and never call this. Choose based on scenario needs.
    """
    raise NotImplementedError


def logs(handle: TestbedHandle, service: str, tail: int = 200) -> str:
    """Return the last ``tail`` lines of a service's container logs.

    ``service`` is ``"aim-server"`` or ``"client"``. Used for
    post-mortem when a scenario fails.
    """
    raise NotImplementedError


def exec_in(
    handle: TestbedHandle,
    service: str,
    cmd: list[str],
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout_s: float = 300.0,
) -> tuple[int, str, str]:
    """Run ``cmd`` inside a service's container. Returns (exit, stdout, stderr).

    ``env`` merges into the container's env for this exec — used to
    parameterize mock training/eval scripts via ``MOCK_*`` env vars.
    ``check=True`` raises ``TestbedExecError`` on non-zero exit.
    Wraps ``docker compose exec``.
    """
    raise NotImplementedError


class TestbedStartupError(RuntimeError):
    """Raised when the testbed containers fail to reach healthy state."""


class TestbedExecError(RuntimeError):
    """Raised when ``exec_in(check=True)`` sees a non-zero exit."""
