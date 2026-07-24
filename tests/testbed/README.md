# Callback testbed

Integration tests for `astrolabe-callbacks` against a **docker-compose environment**: two containers on a shared bridge network — `aim-server` (simulates the NUC's Aim endpoint) and `client` (simulates a compute host running the callback library). Scenarios exercise the callback across a real TCP hop into a real Aim server, in the shape that a customer's Lambda instance would see.

Companion testbed: `astrolabe/tests/testbed/`. Same design pattern, different scope. Astrolabe testbed simulates the full NUC-plus-compute environment (SSH, sidecar, engine, dashboard). This one simulates only what the callback sees: bridge network + Aim server.

## Running

Default unit-test runs skip the testbed:

```
pytest                    # unit tests only
```

Opt in explicitly:

```
pytest -m testbed         # integration scenarios — brings compose up, runs against a real aim server
pytest -m testbed_scale   # sustained-load scenarios — hours-long, testbed_scale marker
```

Requires docker + docker-compose installed and running. Pytest owns the compose lifecycle via the `testbed` fixture — you don't run `docker-compose up` yourself.

CI runs the integration scenarios on every PR (`.github/workflows/testbed.yml`, job `fast`). Sustained-load scenarios run nightly on schedule (job `scale`).

## Layout

```
tests/testbed/
├── README.md               (this file)
├── docker-compose.yml      two services: aim-server + client
├── Dockerfile.aim          aim server image (simulates NUC)
├── Dockerfile.client       python + callback source (simulates compute)
├── conftest.py             session-scope `testbed` fixture; per-test `aim_repo` path
├── harness/
│   ├── compose.py          docker-compose lifecycle: up, down, exec_in, logs
│   ├── assertions.py       Aim SDK query helpers, host-side reads
│   ├── mock_training.py    env-var-driven emitter — runs INSIDE the client container
│   └── mock_eval.py        eval-helper driver — runs INSIDE the client container
└── scenarios/
    ├── test_core.py            _core.py — Logger, buffer, drainer, schema-finalize, tags, name, first_metric marker, hash fidelity
    ├── test_distributed.py     _distributed.py — rank gating
    ├── test_composer.py        composer.py — Composer adapter
    ├── test_lightning.py       lightning.py — Lightning adapter
    ├── test_huggingface.py     huggingface.py — HF adapter
    ├── test_pytorch.py         pytorch.py — raw PyTorch helper
    ├── test_eval_results.py    eval_results.py — log_eval_table + start_eval_run + from_checkpoint
    ├── test_aim_compat.py      external contract: Aim SDK behaviors we depend on
    └── test_sustained.py       operational mode: sustained load (testbed_scale marker)
```

## Design opinions

- **Two containers on a bridge network, not one host.** The callback in production talks to Aim over a network hop (reverse SSH tunnel on the NUC). Loopback-on-host hides bugs that only surface across a real socket. Bridge network is one step closer to production.
- **Bind-mount the aim repo to the host.** The aim container writes to `/var/lib/aim`; the host mounts a per-session tmp dir to the same path. Host-side pytest assertions read the same directory the aim server writes to via `aim.Repo(read_only=True)` — no `docker exec` gymnastics for reads.
- **Bind-mount `src/` and `tests/` into the client container.** Live-edit: any local change to the callback library is immediately visible to the client without a rebuild. Callback PRs iterate against the tree they're editing.
- **Session-scope `testbed` fixture.** Compose up is slow (~30-60s); scenarios amortize it across the whole testbed run. Tests isolate on run hashes / experiment names rather than on repo state.
- **Real Aim server, not FakeAimRun.** Unit tests use FakeAimRun to catch API-shape bugs. The testbed's whole point is what FakeAimRun cannot catch: memtable flush semantics, RocksDB chunk visibility, protobuf serialization, drainer thread races under real transport.
- **One scenario file per src module.** Not organized by phase — organized by what code owns the behavior. Prevents cross-file duplication. See the plan's "Test file structure" section for the placement rules.
- **Framework adapters share class-name convention.** `TestTraining` / `TestValidation` / `TestTeardown` in every framework test file, so failures across frameworks are easy to correlate at a glance. Framework-specific quirks (like PyTorch's context-manager entry point) live in dedicated classes.
- **Frameworks skip when their extra is missing.** `pytest.importorskip("composer")` in the framework-specific tests. A lean install sees framework scenarios as skips, not errors.

## Related plans

- `plans/callback-testbed.md` — this testbed's design plan
- `plans/eval-linkage-and-checkpoint-callbacks.md` — checkpoint metadata + eval-linkage work; adds CLI scenarios to this testbed when the CLIs land
- `astrolabe/plans/healing-and-failure-hooks-testing.md` — sibling testbed in the astrolabe repo
