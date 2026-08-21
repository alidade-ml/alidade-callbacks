"""Astrolabe contract between engine and callback.

Source of truth: this file in the engine repo. Vendored verbatim into
``astrolabe-callbacks`` (and any third-party callback library) via
``tools/vendor-contract.py``.

This file holds the **names** (env vars, Aim tag keys, metric
namespaces, default values) AND the **canonical formatters/parsers**
for any value with a non-trivial encoding. Both sides of the contract
go through the same helpers so the wire format can't drift across the
engine/callback split.

Behavioral expectations live in ``docs/callback-contract.md``.

Rules enforced by CI:

- Stdlib-only imports. Adding any third-party import is a contract
  violation; ``tools/check-contract-stdlib-only.py`` blocks merge.
- Modifying this file requires bumping ``CONTRACT_VERSION``;
  ``tools/check-contract-bump.py`` blocks merge.
- Every ``ASTROLABE_*``/``AIM_*`` env var the engine sets and every
  ``astrolabe.*`` Aim tag the engine reads must appear as a constant
  here; ``tests/test_contract_completeness.py`` enforces.
- Bare contract-literal strings outside this file are a violation;
  the engine must route through ``contract.ENV_*``/``contract.TAG_*``
  + the format/parse helpers (no inline ``json.dumps`` of contract
  values, no inline ``"astrolabe.user"`` keys).
"""

from __future__ import annotations

# Contract version (semver).
#
# Bump pattern:
#   MAJOR — breaking change (rename or remove an env var / tag, change
#           the wire format of a value)
#   MINOR — backward-compatible addition (new optional env var / tag,
#           new helper)
#   PATCH — doc-only / comment-only edit
#
# A callback library's vendored copy carries the engine version it was
# vendored from; the engine refuses submits whose pinned callback was
# vendored against a contract older than what this engine version
# requires.
CONTRACT_VERSION = "1.8.0"

# --- Env vars: ENGINE sets in the training process -------------------------
#
# The engine writes these into the training process's environment before
# the training command runs. A contract-compliant callback reads them
# (directly or via helpers) to wire itself to the orchestration.

# Unique identifier for one submit (one `astrolabe submit` invocation).
# Callbacks tag the Aim run with this so the dashboard can link the run
# back to the submit row in the state DB.
ENV_SUBMIT_ID = "ASTROLABE_SUBMIT_ID"

# Human-readable experiment name from the YAML. Callbacks may use this
# as the Aim run name when one isn't set explicitly.
ENV_EXPERIMENT_NAME = "ASTROLABE_EXPERIMENT_NAME"

# Tag dict the engine wants applied to every run produced under this
# submit. Wire format: ``key1=val1,key2=val2`` (NOT JSON — keys and
# values are pasted directly into the env var, comma-separated). Use
# :func:`format_aim_run_tags` and :func:`parse_aim_run_tags` to read
# and write this — never inline the encoding.
ENV_AIM_RUN_TAGS = "AIM_RUN_TAGS"

# Filesystem path to the local Aim repo the callback should write
# through. Set only when the NUC has ``aim_local_mode: true`` in
# ``/etc/astrolabe/config.yaml`` (v1.7.0+). When unset, callbacks fall
# back to the tunneled Aim server at ``aim://localhost:43800``.
# Engine constructs the value via :func:`format_local_aim_repo_path`.
ENV_AIM_REPO_PATH = "ASTROLABE_AIM_REPO_PATH"

# Path to a jsonl file the callback appends structured events to (run
# open/close, schema finalize, dropped batches, etc.). Used by the
# canary verifier harness for cross-checking claimed side effects.
ENV_CALLBACK_STATS_PATH = "ASTROLABE_CALLBACK_STATS_PATH"

# Directory the engine has provisioned for per-rank stdout/stderr logs
# during distributed training. Callbacks (and frameworks) write rank-N
# logs into ``$ASTROLABE_RANK_LOGS_DIR/rank-N.{stdout,stderr}``.
ENV_RANK_LOGS_DIR = "ASTROLABE_RANK_LOGS_DIR"

# Filesystem path the astrolabe-callbacks library touches when the
# first Aim metric write lands.  The engine probes this path at
# step-failure time to enforce ``until: first_metric`` healing bounds
# (:class:`astrolabe.config.StepHealingConfig`).
ENV_FIRST_METRIC_MARKER = "ASTROLABE_FIRST_METRIC_MARKER"

# Same mechanism for ``until: first_checkpoint``.  Separate marker
# because the two windows close on different events: a run can emit
# metrics for hours before its first checkpoint.
ENV_FIRST_CHECKPOINT_MARKER = "ASTROLABE_FIRST_CHECKPOINT_MARKER"

# --- Aim run tags: CALLBACK writes, ENGINE + dashboard read ----------------
#
# Callbacks apply these to the Aim run at open time. The engine reads
# them back via the Aim API (state DB lookups, dashboard rendering,
# Linear report generation, etc.). Renaming any of these is a MAJOR
# contract bump.

TAG_SUBMIT_ID = "astrolabe.submit_id"
TAG_USER = "astrolabe.user"
TAG_VERSION = "astrolabe.version"
TAG_EXPERIMENT = "astrolabe.experiment"
TAG_GPU_TYPE = "astrolabe.gpu_type"
TAG_GPU_RATE_CENTS_PER_HOUR = "astrolabe.gpu_rate_cents_per_hour"

# Final outcome — set by the engine on terminal state ("success" /
# "failure" / "cancelled"). Callbacks don't write this; it's listed
# here so engine code that reads it goes through the constant.
TAG_OUTCOME = "astrolabe.outcome"

# Eval-run identity. Written by the callback library, read by the
# dashboard; the engine touches neither. Declared here because this
# file is the registry for the ``astrolabe.*`` namespace, not only for
# the keys the engine itself handles — same reason NAMESPACE_EVAL is
# here. Nothing enforces the dashboard half of this contract: the hub
# types these strings into Go source and no check would fail if they
# drifted apart.
TAG_KIND = "astrolabe.kind"
TAG_TASK_SET = "astrolabe.task_set"
TAG_MODEL_RUN_HASH = "astrolabe.model_run_hash"

# Sample-run identity. Same ownership as the eval keys above: written by
# the callback library, read by the dashboard, untouched by the engine.
# ``sample_set`` groups one batch of outputs the way ``task_set`` groups
# one benchmark suite.
TAG_SAMPLE_SET = "astrolabe.sample_set"

# Value for TAG_KIND on a post-training benchmark run. The dashboard's
# eval discovery matches on this exact string.
KIND_EVAL = "eval"

# Value for TAG_KIND on a model astrolabe did not train — a downloaded
# checkpoint an eval scored. The run carries no metrics; it exists so
# the eval has a model to attribute to and the dashboard has a row to
# put in a leaderboard. The dashboard keys its "is this a training run"
# test on TAG_KIND being absent or "training", so any value here keeps
# the entry off the training charts.
KIND_EXTERNAL_CHECKPOINT = "external_checkpoint"

# Value for TAG_KIND on a run holding qualitative model outputs — a few
# completions, a few generated images — rather than metrics. Samples rank
# nothing and are not compared; they exist to be looked at, which is why
# they are a distinct kind rather than an eval with unusual values.
KIND_SAMPLE = "sample"

# --- Metric namespaces -----------------------------------------------------
#
# Conventions, not strictly enforceable from the engine side, but the
# dashboard groups metrics by these prefixes.

NAMESPACE_TRAIN = "train/"           # during-training metrics
NAMESPACE_VAL = "val/"               # during-training validation
NAMESPACE_EVAL = "eval/"             # post-training benchmarks

# Engine-synthesized metric: wall-clock time at each step. Callbacks
# don't write this themselves; the engine derives it from Aim's
# per-step timestamps when the Go API serves the run.
SYNTHESIZED_WALL_TIME = "wall_time"

# --- Defaults --------------------------------------------------------------

# Default Aim repo path template (local-aim mode, v1.7.0+). Substitute
# the submit_id via :func:`format_local_aim_repo_path` — do not call
# ``LOCAL_AIM_REPO_PATH_TEMPLATE.format(...)`` directly at call sites.
LOCAL_AIM_REPO_PATH_TEMPLATE = "/tmp/aim-local-{submit_id}"

# Sequence names a sample batch writes. ``log_samples`` tracks an
# ``aim.Text`` or ``aim.Image`` under ``sample/<set>/input`` and
# ``sample/<set>/output``, paired by step.
#
# Declared here because it has two consumers that must agree exactly: the
# callback builds these names, and the NUC-side exporter reads them back
# (SAMPEXP-1). A disagreement is silent — the reader finds no sequence and
# reports a batch with no samples, which is a plausible and wrong answer.
# Substitute via :func:`format_sample_sequence_name`; do not call
# ``.format(...)`` at a call site.
SAMPLE_SEQUENCE_TEMPLATE = "sample/{sample_set}/{role}"

# The two halves of a sample. ``role`` is one of these and nothing else.
SAMPLE_ROLE_INPUT = "input"
SAMPLE_ROLE_OUTPUT = "output"

# Default Aim tracking-server URL. The engine opens a reverse SSH
# tunnel from the compute host to the NUC's Aim server on port 43800
# (see ``astrolabe.engine._setup``). Training-time consumers (callbacks
# and the canary workload) connect to this URL by default; both sides
# of the contract MUST agree on the port number so the tunnel + client
# pair line up.
DEFAULT_AIM_URL = "aim://localhost:43800"

# --- Canonical formatters / parsers ---------------------------------------
#
# Both engine and callback go through these. The wire format lives in
# exactly one place, so a change to the encoding requires editing this
# file — which triggers the CONTRACT_VERSION-bump CI guard.
#
# Why these specifically: only values with non-trivial encodings need a
# helper. A constant like ``ENV_SUBMIT_ID`` is just a name; the value
# is just a string passed through, no encoding involved. ``AIM_RUN_TAGS``
# encodes a dict into a single string, and ``ASTROLABE_AIM_REPO_PATH``
# templates a submit_id into a path — both are encodings, both need
# canonical helpers.


def format_aim_run_tags(tags: dict[str, str]) -> str:
    """Encode a tag dict into the ``AIM_RUN_TAGS`` wire format.

    Wire format is ``key1=val1,key2=val2``. Keys and values are
    inserted literally — callers must not include ``=`` or ``,`` in
    keys or values. In practice astrolabe's tag keys are all
    ``astrolabe.*`` literals (no ``=`` or ``,``) and values are
    submit_ids / version labels / GPU types / integer rates (none of
    which contain those characters either).

    Parameters
    ----------
    tags : dict[str, str]
        The tag dict to encode.

    Returns
    -------
    str
        The env-var-shaped wire format. ``""`` for an empty dict.
    """
    return ",".join(f"{k}={v}" for k, v in tags.items())


def parse_aim_run_tags(raw: str | None) -> dict[str, str]:
    """Decode the ``AIM_RUN_TAGS`` wire format into a tag dict.

    Inverse of :func:`format_aim_run_tags`. Forgiving rather than
    strict — this reads a value a researcher may have pasted into a
    shell, not a machine-validated payload. Entries without ``=``,
    with empty keys, or duplicate keys (last wins) are tolerated.
    Whitespace around keys/values is stripped.

    Parameters
    ----------
    raw : str | None
        The raw env var value, or ``None``/empty.

    Returns
    -------
    dict[str, str]
        Parsed tags. ``{}`` if ``raw`` is empty or unparseable.
    """
    if not raw:
        return {}
    out: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        out[key] = value
    return out


def format_local_aim_repo_path(submit_id: str) -> str:
    """Construct the per-submit local Aim repo path.

    Substitutes ``submit_id`` into :data:`LOCAL_AIM_REPO_PATH_TEMPLATE`.
    Engine sets the env var via this helper in local-aim mode.

    Every consumer must derive the path from here rather than rebuild it.
    The callback writes to this path and the sync sidecar reads from it, and
    a disagreement between them is silent: rsync of a nonexistent source is
    not fatal to the sidecar, so training completes normally and the
    dashboard simply shows a run with no data.
    """
    return LOCAL_AIM_REPO_PATH_TEMPLATE.format(submit_id=submit_id)


def format_sample_sequence_name(sample_set: str, role: str) -> str:
    """Construct the Aim sequence name for one half of a sample batch.

    Parameters
    ----------
    sample_set : str
        The researcher's label for the batch. Must not contain ``/`` — it is a
        path segment here, and a slash forks the namespace so the name cannot
        be taken apart again. ``log_samples`` refuses one at the call site;
        this refuses it too, because the exporter reads runs Aim will happily
        have accepted from anywhere.
    role : str
        :data:`SAMPLE_ROLE_INPUT` or :data:`SAMPLE_ROLE_OUTPUT`.
    """
    if "/" in sample_set:
        raise ValueError(
            f"sample_set must not contain '/' (got {sample_set!r}) — it is a "
            "path segment in the sequence name"
        )
    if role not in (SAMPLE_ROLE_INPUT, SAMPLE_ROLE_OUTPUT):
        raise ValueError(
            f"role must be {SAMPLE_ROLE_INPUT!r} or {SAMPLE_ROLE_OUTPUT!r}, "
            f"got {role!r}"
        )
    return SAMPLE_SEQUENCE_TEMPLATE.format(sample_set=sample_set, role=role)


def format_first_metric_marker_path(submit_id: str, step_num: int) -> str:
    """Construct the per-step first-metric marker path.

    Engine sets ``ASTROLABE_FIRST_METRIC_MARKER`` to this value.  The
    astrolabe-callbacks library touches this file on the first Aim
    metric write; the engine probes it to enforce
    ``until: first_metric`` healing bounds.

    Keyed on step, not submit.  A submit-scoped marker leaks across
    steps: step 1 emits a metric, step 2 crashes before emitting one,
    and step 2's ``until: first_metric`` window is judged already
    closed by step 1's evidence.
    """
    return f"astrolabe-first-metric-{submit_id}-step{step_num}.marker"


def format_first_checkpoint_marker_path(submit_id: str, step_num: int) -> str:
    """Construct the per-step first-checkpoint marker path.

    Engine sets ``ASTROLABE_FIRST_CHECKPOINT_MARKER`` to this value.
    The astrolabe-callbacks library touches this file when the first
    checkpoint of the step is written; the engine probes it to enforce
    ``until: first_checkpoint`` healing bounds.
    """
    return f"astrolabe-first-checkpoint-{submit_id}-step{step_num}.marker"
