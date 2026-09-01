"""Drive ``log_samples`` inside the client container, against a real Aim server.

Mocking Aim proves nothing about Aim. The failure modes that matter here live
in its storage layer — whether a tracked ``aim.Text`` comes back as text, and
whether two sequences sharing a step index still pair after a round trip — and
a mock returns whatever you tell it to.

Invoked as::

    compose.exec_in(testbed, service="client",
                    cmd=["python", "-m", "tests.testbed.harness.sample_driver"],
                    env=sample_config_to_env(config))

Prints ``ALIDADE_SAMPLE_RUN_HASH=<hash>`` on success; exits 43 on
``MissingParentError``, matching the eval driver's convention so a scenario can
assert "refused before writing anything" by exit code.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SampleDriverConfig:
    aim_url: str
    sample_set: str
    # [[input_or_null, output], ...] — a list, not a mapping, because sample
    # inputs are not unique. An element is either a string (logged as text) or
    # ``{"image": {"w": W, "h": H, "seed": S}}``, which the driver turns into a
    # real image inside the container.
    samples: list[list[object]]
    model_run_hash: str | None = None
    checkpoint_path: str | None = None
    external_name: str | None = None
    # Ask the driver to write a checkpoint before logging. Values are the
    # aim_run_hash to embed; the driver creates the file in the container.
    create_pt_with_hash: str | None = None
    create_safetensors_with_hash: str | None = None
    # Env the engine would have exported into a sample step. Passed verbatim:
    # the helpers read the real env var names, so anything else tests a
    # stand-in.
    submit_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "SampleDriverConfig":
        def _req(key: str) -> str:
            v = os.environ.get(key)
            if v is None:
                print(f"missing required env var {key}", file=sys.stderr)
                raise SystemExit(2)
            return v

        return cls(
            aim_url=_req("TESTBED_SAMPLE_AIM_URL"),
            sample_set=_req("TESTBED_SAMPLE_SET"),
            samples=json.loads(_req("TESTBED_SAMPLE_SAMPLES")),
            model_run_hash=os.environ.get("TESTBED_SAMPLE_MODEL_RUN_HASH") or None,
            checkpoint_path=os.environ.get("TESTBED_SAMPLE_CHECKPOINT_PATH") or None,
            external_name=os.environ.get("TESTBED_SAMPLE_EXTERNAL_NAME") or None,
            create_pt_with_hash=os.environ.get("TESTBED_SAMPLE_CREATE_PT") or None,
            create_safetensors_with_hash=(
                os.environ.get("TESTBED_SAMPLE_CREATE_ST") or None
            ),
        )


def make_pattern(width: int, height: int, seed: int):
    """Deterministic RGB pattern, shared by the driver and the scenario.

    The driver logs it; the scenario regenerates it and compares. Random
    rather than a solid colour on purpose — a solid image survives a
    transpose or a channel swap unchanged, so it would assert nothing about
    the round trip actually preserving the image.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _materialize(value: object) -> object:
    """A JSON sample element becomes the payload log_samples will receive."""
    if isinstance(value, dict) and "image" in value:
        spec = value["image"]
        return make_pattern(spec["w"], spec["h"], spec["seed"])
    return value


def sample_config_to_env(config: SampleDriverConfig) -> dict[str, str]:
    env = {
        "TESTBED_SAMPLE_AIM_URL": config.aim_url,
        "TESTBED_SAMPLE_SET": config.sample_set,
        "TESTBED_SAMPLE_SAMPLES": json.dumps(config.samples),
        "ALIDADE_AIM_URL": config.aim_url,
    }
    if config.model_run_hash:
        env["TESTBED_SAMPLE_MODEL_RUN_HASH"] = config.model_run_hash
    if config.checkpoint_path:
        env["TESTBED_SAMPLE_CHECKPOINT_PATH"] = config.checkpoint_path
    if config.external_name:
        env["TESTBED_SAMPLE_EXTERNAL_NAME"] = config.external_name
    if config.create_pt_with_hash:
        env["TESTBED_SAMPLE_CREATE_PT"] = config.create_pt_with_hash
    if config.create_safetensors_with_hash:
        env["TESTBED_SAMPLE_CREATE_ST"] = config.create_safetensors_with_hash
    env.update(config.submit_env)
    return env


def _prepare_checkpoint(config: SampleDriverConfig) -> None:
    """Write the checkpoint the helper will read, if the scenario asked for one.

    Created here rather than shipped as a fixture: the point of the exercise is
    the real embed-then-read round trip, so the file has to be produced by the
    same library version that reads it.
    """
    if config.checkpoint_path is None:
        return
    if not (config.create_pt_with_hash or config.create_safetensors_with_hash):
        return

    from astrolabe_callbacks.checkpoint import CheckpointMeta, export_checkpoint

    path = Path(config.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped_at = "2026-08-19T00:00:00Z"
    if config.create_pt_with_hash:
        export_checkpoint(
            {}, path, fmt="pt",
            meta=CheckpointMeta(
                aim_run_hash=config.create_pt_with_hash, created_at=stamped_at
            ),
        )
    if config.create_safetensors_with_hash:
        export_checkpoint(
            {}, path, fmt="safetensors",
            meta=CheckpointMeta(
                aim_run_hash=config.create_safetensors_with_hash,
                created_at=stamped_at,
            ),
        )


def run_sample_driver(config: SampleDriverConfig) -> str:
    from astrolabe_callbacks import Sample, log_samples

    _prepare_checkpoint(config)

    samples = [
        Sample(input=_materialize(pair[0]), output=_materialize(pair[1]))
        for pair in config.samples
    ]
    return log_samples(
        sample_set=config.sample_set,
        samples=samples,
        model_run_hash=config.model_run_hash,
        checkpoint=config.checkpoint_path,
        external_name=config.external_name,
        aim_url=config.aim_url,
    )


def main() -> None:
    config = SampleDriverConfig.from_env()
    try:
        run_hash = run_sample_driver(config)
    except Exception as e:
        if type(e).__name__ == "MissingParentError":
            raise SystemExit(43)
        raise
    print(f"ALIDADE_SAMPLE_RUN_HASH={run_hash}")


if __name__ == "__main__":
    main()
