# Eval

Logging benchmark results — GLUE, MMLU, a held-out set you built — so they land on
astrolabe's **Eval tab** attached to the model that earned them.

For during-training metrics, see [training](training.md). The split matters:

| you logged | lands on | dashboard |
|---|---|---|
| `train/*`, `val/*` | the training run | Training tab |
| `eval/<task>/<metric>` | a **separate** eval run | Eval tab |

`val/` answers *is training converging?*. `eval/` answers *how good is the finished
model?*. They shared a prefix once and were impossible to tell apart.

---

## Scoring a model you trained

The eval run has to say which model it scored. The easy way is to let the checkpoint
answer:

```python
from astrolabe_callbacks import start_eval_run_from_checkpoint

run = start_eval_run_from_checkpoint(checkpoint="ckpt.pt", task_set="glue")

for task, metric, score in results:
    run.track(score, name=f"eval/{task}/{metric}", step=0)
run.close()
```

No hash at the call site. The checkpoint carries the training run's identity — see
[checkpoints](checkpoints.md) — and this reads it **offline**, without contacting Aim.

`checkpoint=` takes a path or an already-loaded state dict, so an eval script that has
already loaded the weights does not pay for a second read.

> **Do not look the model up by name.** Searching Aim for "the latest run in this
> experiment" is a guess that silently picks the wrong run when several exist, and it
> returns nothing at all under local-aim transport, where the compute host only sees its
> own submit. If you cannot use the checkpoint, pass `model_run_hash=` explicitly.

### If you already have the hash

```python
from astrolabe_callbacks import log_eval_table

log_eval_table(
    model_run_hash="abc123...",
    task_set="glue",
    rows={
        "cola": ("matthews",         0.822),
        "sst2": ("accuracy",         0.943),
        "mnli": ("accuracy_matched", 0.864),
        "avg":  ("mean",             0.876),
    },
)
```

One call: opens the run, tags it, writes every row, closes. `rows` maps a task to a
`(metric_label, score)` pair.

> **`log_eval_table` connects only after every score is in hand.** It validates the dict,
> *then* opens the Aim run. If the connection is down at that moment, an hour of
> benchmarking is gone. `start_eval_run_from_checkpoint` opens first and writes as you go,
> so a connection problem surfaces before you spend anything and partial results survive.
> Prefer it for anything slow.

---

## Models astrolabe never trained

Benchmarking a downloaded checkpoint — `roberta-base`, a collaborator's file. Astrolabe
has no record of it, so there is nothing for results to attach to. Name it and the library
creates that record:

```python
run = start_eval_run_from_checkpoint(
    checkpoint="~/.cache/huggingface/.../model.safetensors",
    task_set="glue",
    external_name="roberta-base",
)
```

The name becomes the row label everywhere the model appears. It is required **only** when
the file carries no provenance, so it never shows up in code evaluating your own models.

The model gets an entry in Aim under the experiment you are submitting from, which means
it appears in the runs panel and can sit in the same leaderboard as models you trained.

> **Nothing is written to the checkpoint file.** A downloaded model usually lives in a
> shared cache, possibly read-only, used by every project on the machine. Recording it
> touches Aim only.

Scoring the same model on several benchmarks from one script? Pass the name each time.
The model is recorded once and every eval attaches to that one entry:

```python
for task_set, rows in (("glue", glue_rows), ("mmlu", mmlu_rows)):
    run = start_eval_run_from_checkpoint(
        checkpoint=path, task_set=task_set, external_name="roberta-base",
    )
    ...
```

Across separate steps of a submit it gets an entry per step, because each step is its own
process and nothing is looked up. One row per benchmark run is the honest record of what
happened; if you want them under a single row, score them from one script.

External models appear in eval leaderboards but **not** in training charts — they have no
training curve to draw.

---

## Rolling evals during training

Scoring every N steps, to watch a benchmark move. Use `step=` and the dashboard renders a
**chart** instead of a table:

```python
from astrolabe_callbacks import start_eval_run

run = start_eval_run(model_run_hash="abc123...", task_set="cola-trace")
for checkpoint_step in (10_000, 20_000, 30_000):
    run.track(score_at(checkpoint_step), name="eval/cola/matthews", step=checkpoint_step)
run.close()
```

**The dispatch is on `step`:**

- every row at `step=0` → **table** (a leaderboard, one row per model)
- any row with `step > 0` → **trace** (a line chart, one line per model)

Want both views of the same benchmark? Emit two eval runs with different `task_set`
labels — a final-step table and a convergence trace.

---

## The metric path

Exactly three segments:

```
eval / <task> / <metric>
        cola      matthews
```

- segment 2 becomes a **row** in the table
- segment 3 becomes a **column**

Slashes in either field are rejected at the call site, because a stray one silently
scrambles which segment is which.

### The `avg` column

Log it as a row. The dashboard renders a row keyed `"avg"` as the last column.

```python
rows = {"cola": ("matthews", 0.822), "avg": ("mean", 0.876)}
```

**The library never computes it.** Mean, harmonic mean, the paper-canonical subset — that
is a research decision, and guessing it would put a number on your dashboard that you did
not choose. Same for multi-seed: average in your script, then log one number per task. Or
log each seed as its own `task_set` and compare them side by side.

---

## When nothing resolves

If the checkpoint has no provenance and you passed neither `model_run_hash=` nor
`external_name=`, the call **raises `MissingParentError`** before any scoring happens.

That is deliberate. The alternative — writing an eval run with no model attached — puts
results in Aim that the dashboard can never surface. The benchmark ran, the numbers
exist, and nobody can find them. Failing at the start costs you nothing; failing silently
at the end costs the whole run.

```python
run = start_eval_run_from_checkpoint(
    checkpoint=ckpt, task_set="glue", on_missing_parent="warn",
)
if not run.astrolabe_linked:
    ...   # returns an unlinked run; stamp it later
```

Use `"warn"` only if you intend to stamp the run afterwards.

> **Changed in v2.0.0-rc4**: the default was `"warn"`. If you relied on unlinked eval
> runs, pass `on_missing_parent="warn"` explicitly.

---

## Connecting to Aim

Every helper takes an optional `aim_url`, resolved the same way as everywhere else in the
library:

1. `ASTROLABE_AIM_REPO_PATH` env — a filesystem path, set in local-aim mode
2. `ASTROLABE_AIM_URL` env — set by astrolabe on provisioned instances
3. the `aim_url=` argument
4. `aim://localhost:43800`, the tunnel astrolabe opens

The repo path wins because in local-aim mode astrolabe opens no tunnel, so the
`aim://` default answers nothing.

Running as a step of an astrolabe submit, omit it. Running elsewhere, pass it — a URL or
a filesystem path both work.

### What you inherit inside a submit

An eval script launched as a submit step picks up that submit's identity automatically —
submitter, version, submit id, GPU rate — and the eval is filed under the submitting
experiment. Nothing to pass; scripts are identical inside and outside a submit.

Outside one, the run falls back to an experiment named `eval/<task_set>`.

---

## Gotchas

- **Scores must be numeric.** `bool` is rejected explicitly — `True` would otherwise log
  as `1.0`.
- **Empty `rows` is rejected.** Nothing to log means nothing to call.
- **Close what you open.** `start_eval_run` hands you the run; forgetting `.close()`
  leaves `end_time` at zero and the dashboard treats it as still running.
  `log_eval_table` closes for you, including when tracking raises.
- **Re-running an eval** makes a new run. The dashboard shows the newest per
  `(model, task_set)`; older ones stay in Aim for forensics.

---

## The API

```python
from astrolabe_callbacks import (
    start_eval_run_from_checkpoint,  # start here
    log_eval_table,                  # one-shot, when you have the hash
    start_eval_run,                  # streams and custom metric names
    EvalInputError,                  # malformed input
    MissingParentError,              # nothing to attribute to
)
```

Base install only — `pip install astrolabe-callbacks`. No framework extra; an eval script
needs to reach Aim, nothing more.

---

## See also

- [Checkpoints](checkpoints.md) — the provenance this reads
- [Training](training.md) — `train/` and `val/`
- [Contract](contract.md) — what every callback guarantees
