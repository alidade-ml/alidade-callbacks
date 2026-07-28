"""Docker-compose lifecycle for the callback testbed.

Two containers on a shared bridge network:

* ``aim-server`` — runs ``aim server`` (simulates the NUC's Aim endpoint)
* ``client`` — python env with the callback source bind-mounted from the
  repo; pytest invokes the driver + eval-driver scripts inside this
  container via ``exec_in``. Simulates a compute host.

Bridge network gives real TCP between callback (client) and Aim (server) —
one step closer to production than subprocess-on-loopback would give.
The aim repo is bind-mounted from a per-session host directory so
host-side assertions can read what the server wrote directly.

Same design shape as ``astrolabe/tests/testbed/harness/compose.py``;
scenarios that graduate from one testbed to the other should feel
familiar.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
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
    "TestbedStartupError",
    "TestbedExecError",
]


AIM_CONTAINER = "astrolabe-callbacks-testbed-aim"
CLIENT_CONTAINER = "astrolabe-callbacks-testbed-client"
AIM_HOST_PORT = 43810  # host-published port (matches docker-compose.yml)
AIM_CLIENT_URL = "aim://aim-server:43800"  # bridge-network URL


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


def up(
    compose_file: Path,
    aim_repo_host_path: Path,
    build_timeout_s: float = 900.0,
    healthy_timeout_s: float = 120.0,
) -> TestbedHandle:
    """Bring the testbed containers up and return a live handle.

    Runs ``docker compose -f <compose_file> up -d`` after exporting
    ``AIM_REPO_HOST_PATH`` so the aim container bind-mounts to the
    caller-supplied host directory (typically a pytest tmp_path).

    ``build_timeout_s`` caps the ``compose up --build`` call. Cold GHA
    runners take ~5-10 min to build both images (torch + transformers +
    lightning + mosaicml pulls); default 15 min leaves margin.
    ``healthy_timeout_s`` caps the post-up readiness probe separately —
    once containers are up, TCP + aim.Run roundtrip should complete in
    seconds.

    Blocks until both containers are healthy (aim server accepting TCP
    connections, client container reports ready). Raises
    ``TestbedStartupError`` on timeout or docker-compose failure.
    """
    aim_repo_host_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["AIM_REPO_HOST_PATH"] = str(aim_repo_host_path)

    # TESTBED_KEEP=1 (local fast-iteration) and TESTBED_SKIP_BUILD=1 (CI,
    # images pre-built with GHA cache) both skip --build.
    up_args = ["docker", "compose", "-f", str(compose_file), "up", "-d"]
    if os.environ.get("TESTBED_KEEP") != "1" and os.environ.get("TESTBED_SKIP_BUILD") != "1":
        up_args.append("--build")

    result = subprocess.run(
        up_args,
        env=env,
        capture_output=True,
        text=True,
        timeout=build_timeout_s,
    )
    if result.returncode != 0:
        raise TestbedStartupError(
            f"docker compose up failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    handle = TestbedHandle(
        compose_file=compose_file,
        aim_container=AIM_CONTAINER,
        client_container=CLIENT_CONTAINER,
        aim_repo_host_path=aim_repo_host_path,
        aim_url_from_client=AIM_CLIENT_URL,
        aim_url_from_host=f"aim://localhost:{AIM_HOST_PORT}",
    )

    wait_healthy(handle, timeout_s=healthy_timeout_s)
    return handle


def down(handle: TestbedHandle, purge_volumes: bool = True) -> None:
    """Tear down the testbed containers.

    ``purge_volumes=True`` runs ``docker compose down -v`` so any
    docker-managed volumes are wiped. The bind-mounted aim repo is
    host-owned so this doesn't touch it — teardown of the host-side
    tmp_path is the fixture's responsibility.
    """
    env = os.environ.copy()
    env["AIM_REPO_HOST_PATH"] = str(handle.aim_repo_host_path)

    args = ["docker", "compose", "-f", str(handle.compose_file), "down"]
    if purge_volumes:
        args.append("-v")

    subprocess.run(args, env=env, capture_output=True, text=True, timeout=60)


def wait_healthy(handle: TestbedHandle, timeout_s: float = 30.0) -> None:
    """Poll container readiness until both aim + client report healthy.

    Aim readiness: TCP connect to the aim server's published port.
    Client readiness: ``docker compose exec client python -c ...`` returns
    successfully.
    """
    deadline = time.monotonic() + timeout_s

    # Aim: TCP connect to the published port
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", AIM_HOST_PORT), timeout=1.0):
                break
        except (OSError, socket.timeout):
            time.sleep(0.5)
    else:
        raise TestbedStartupError(
            f"aim server not accepting connections on port {AIM_HOST_PORT} within {timeout_s}s"
        )

    # Client: python -c 'print("ok")' returns cleanly
    while time.monotonic() < deadline:
        code, _, _ = exec_in(
            handle, service="client", cmd=["python", "-c", "print('ok')"], check=False, timeout_s=5.0
        )
        if code == 0:
            break
        time.sleep(0.5)
    else:
        raise TestbedStartupError(f"client container not ready within {timeout_s}s")

    # Real Aim server readiness: TCP being open doesn't mean the server is
    # actually serving. Confirm by opening + closing an aim.Run from the
    # client container against the bridge-network URL. This catches
    # aim-server-still-initializing races that cause "Failed to connect"
    # errors even after TCP-connect works.
    readiness_script = (
        "import aim; r = aim.Run(repo='" + AIM_CLIENT_URL + "'); r.close()"
    )
    while time.monotonic() < deadline:
        code, _, stderr = exec_in(
            handle,
            service="client",
            cmd=["python", "-c", readiness_script],
            check=False,
            timeout_s=10.0,
        )
        if code == 0:
            return
        time.sleep(1.0)
    raise TestbedStartupError(
        f"aim server not serving from bridge network within {timeout_s}s (last stderr: {stderr})"
    )


def reset_repo(handle: TestbedHandle) -> None:
    """Purge the aim repo contents on disk without restarting containers.

    Uses ``docker exec`` on the aim container to rm -rf and recreate the
    repo directory. Aim server reopens the repo lazily on next write.
    """
    subprocess.run(
        [
            "docker",
            "exec",
            handle.aim_container,
            "sh",
            "-c",
            "rm -rf /var/lib/aim/* /var/lib/aim/.aim",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def logs(handle: TestbedHandle, service: str, tail: int = 200) -> str:
    """Return the last ``tail`` lines of a service's container logs."""
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(handle.compose_file),
            "logs",
            "--tail",
            str(tail),
            service,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout + result.stderr


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
    parameterize the driver + eval-driver scripts via ``TESTBED_*`` env vars.
    ``check=True`` raises ``TestbedExecError`` on non-zero exit.
    Wraps ``docker compose exec``.
    """
    args = ["docker", "compose", "-f", str(handle.compose_file), "exec", "-T"]
    for key, value in (env or {}).items():
        args.extend(["-e", f"{key}={value}"])
    args.append(service)
    args.extend(cmd)

    # docker-compose interpolates ${AIM_REPO_HOST_PATH} in the compose file
    # every invocation — even `exec` reads the file. Set it from the handle
    # so subprocess doesn't inherit a bare shell without the variable.
    subprocess_env = os.environ.copy()
    subprocess_env["AIM_REPO_HOST_PATH"] = str(handle.aim_repo_host_path)

    result = subprocess.run(
        args,
        env=subprocess_env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if check and result.returncode != 0:
        raise TestbedExecError(
            f"exec_in({service}, {cmd}) failed (exit={result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.returncode, result.stdout, result.stderr


class TestbedStartupError(RuntimeError):
    """Raised when the testbed containers fail to reach healthy state."""


class TestbedExecError(RuntimeError):
    """Raised when ``exec_in(check=True)`` sees a non-zero exit."""
