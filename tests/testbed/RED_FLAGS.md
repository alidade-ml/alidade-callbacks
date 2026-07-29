# Callback library red flags — surfaced by the testbed

Findings from Stage 3 review of PR #12 that point at bugs or gaps in
the callback library itself (not the testbed). **Not fixing here** —
this file logs them for follow-up.

Each entry: date + scenario that surfaced it + observed behavior + why
it's a package issue rather than a testbed issue.

## Open entries

### 2026-07-27 — ``aim.Run.track``'s ``@noexcept`` decorator hides errors from the callback library's retry logic

**Surfaced by**: trying to implement fault-injection scenarios (drainer death, transient retry) in the callback testbed. Injected exceptions inside ``aim.Run.track`` never reach the callback library's ``_drain_loop_inner`` retry loop.

**Why**: Aim decorates ``aim.Run.track`` with ``@noexcept`` (see ``aim/ext/exception_resistant.py``). In default "safe mode" this decorator catches every ``Exception`` inside ``track`` and calls ``_SafeModeConfig.log_exception`` (which just logs — no re-raise). From the caller's POV, ``track`` always returns cleanly.

**Impact on the callback library**: ``_MetricBuffer._drain_loop_inner`` has a retry loop wrapped in ``except Exception``:

    try:
        self._run.track(value, name=name, step=step, context=...)
        self._drained += 1
        break
    except Exception as exc:
        self._retried += 1
        ...

Because ``track`` never raises (noexcept swallows), the ``except`` branch is unreachable. This means:
- ``_retried`` counter is dead — never increments regardless of Aim server health
- ``_dropped_failed`` counter is dead — never fires
- ``drainer_died`` event only fires on non-``Exception`` errors inside ``_drain_loop_inner`` (e.g. queue.get raising, or the drainer's own bookkeeping)
- Silent Aim write failures cannot be detected by the callback layer

**Not a testbed bug**: fault-injection scenarios can't be built at the ``aim.Run.track`` layer because of this decorator. The retry logic exists in code but is effectively dead.

**Fix (out of scope for this PR)**: either
1. Callback library calls ``aim.ext.exception_resistant.disable_safe_mode()`` on init so its own retry logic actually gets to see exceptions, OR
2. Callback library documents that ``run.track`` failures are swallowed and its retry logic is best-effort against non-Aim-internal failures only (e.g. drainer thread OS errors).

### 2026-07-27 — ``[hf]`` extra is missing ``accelerate``

**Surfaced by**: ``tests/testbed/scenarios/test_huggingface.py::TestTeardown::test_on_train_end_sets_end_time`` — the driver's real HF Trainer instantiation fails with:

    ImportError: Using the `Trainer` with `PyTorch` requires
    `accelerate>=1.1.0`: Please run `pip install transformers[torch]`
    or `pip install 'accelerate>=1.1.0'`.

**Why it's a package issue**: current ``transformers`` (≥4.42-ish) requires ``accelerate`` for the PyTorch backend of ``Trainer``. Our ``[hf]`` extra declares only ``transformers>=4.30``; anyone doing ``pip install astrolabe-callbacks[hf]`` will hit this same error the first time they instantiate a Trainer with our callback attached.

**Not a testbed bug**: the testbed correctly exercises ``[dev]`` (which pulls the ``[hf]`` chain). The error is what a real customer would see.

**Fix (out of scope for this PR)**: add ``accelerate>=1.1.0`` to the ``[hf]`` and ``[all]`` extras in ``pyproject.toml``.
