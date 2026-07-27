# Callback library red flags — surfaced by the testbed

Findings from Stage 3 review of PR #12 that point at bugs or gaps in
the callback library itself (not the testbed). **Not fixing here** —
this file logs them for follow-up.

Each entry: date + scenario that surfaced it + observed behavior + why
it's a package issue rather than a testbed issue.

## Open entries

### 2026-07-27 — ``[hf]`` extra is missing ``accelerate``

**Surfaced by**: ``tests/testbed/scenarios/test_huggingface.py::TestTeardown::test_on_train_end_sets_end_time`` — the driver's real HF Trainer instantiation fails with:

    ImportError: Using the `Trainer` with `PyTorch` requires
    `accelerate>=1.1.0`: Please run `pip install transformers[torch]`
    or `pip install 'accelerate>=1.1.0'`.

**Why it's a package issue**: current ``transformers`` (≥4.42-ish) requires ``accelerate`` for the PyTorch backend of ``Trainer``. Our ``[hf]`` extra declares only ``transformers>=4.30``; anyone doing ``pip install astrolabe-callbacks[hf]`` will hit this same error the first time they instantiate a Trainer with our callback attached.

**Not a testbed bug**: the testbed correctly exercises ``[dev]`` (which pulls the ``[hf]`` chain). The error is what a real customer would see.

**Fix (out of scope for this PR)**: add ``accelerate>=1.1.0`` to the ``[hf]`` and ``[all]`` extras in ``pyproject.toml``.
