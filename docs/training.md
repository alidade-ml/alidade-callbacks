# Training

Streaming metrics to Aim while your model trains, so they land on astrolabe's **Training
tab**.

One line in most cases. The rest of this page is the things that surprise people.

---

## Attach a logger

<table>
<tr><td>

**Composer**

```python
from alidade_callbacks import AstrolabeComposerLogger

trainer = Trainer(
    model=...,
    train_dataloader=...,
    loggers=[AstrolabeComposerLogger()],
)
```

</td></tr>
<tr><td>

**Lightning**

```python
from alidade_callbacks import AstrolabeLightningLogger

trainer = Trainer(callbacks=[AstrolabeLightningLogger()])
```

</td></tr>
<tr><td>

**HuggingFace Trainer**

```python
from alidade_callbacks import AstrolabeHFTrainerCallback

trainer.add_callback(AstrolabeHFTrainerCallback())
```

</td></tr>
<tr><td>

**Raw PyTorch / Accelerate / JAX / anything**

```python
from alidade_callbacks import Run

with Run(experiment_name="my-experiment") as run:
    for step in range(steps):
        run.log_train(loss=loss, lr=lr, step=step)
```

</td></tr>
</table>

> **Composer takes `loggers=`, not `callbacks=`.** `AstrolabeComposerLogger` is a
> `LoggerDestination`, and Composer only broadcasts user metrics to destinations
> registered there. Composer 0.20+ rejects it in `callbacks=` with a clear error; older
> versions silently drop every metric. Note that the *checkpointer* is the opposite — it
> is a `Callback` and goes in `callbacks=`. See [checkpoints](checkpoints.md).

Install the matching extra: `pip install 'alidade-callbacks[composer]'` — or
`[lightning]`, `[hf]`, `[all]`. Raw PyTorch needs no extra.

---

## Everything you log flows through

We forward every metric you produce, under the name you chose. There is no whitelist, no
sampling, no renaming of your names.

```python
self.log("throughput", samples_per_sec)     # lands as throughput
self.log("my/custom/thing", value)          # lands as my/custom/thing
```

The only metric we synthesize is `wall_time`, so the dashboard can offer a wall-clock
x-axis. It is written at the steps your metrics are written at, and nowhere else —
a point on that axis with nothing to index would be a point the dashboard has to
either drop or pair with something that is not there.

It excludes setup and subtracts eval pauses so that runs stay comparable: two that
did the same amount of training agree at the same step no matter how often they
stopped to evaluate. Aim's own per-record wall-clock includes all of it, which is
why we synthesize this instead of reading that.

**A few framework-owned names get normalized** so the dashboard can find them across
frameworks — Composer's `loss/train/total` and HuggingFace's bare `loss` both become
`train/loss`, and each framework's validation metrics become `val/<name>`. Those are
names the framework chose, not you. Yours are never touched. Full table in
[contract](contract.md).

### `train/` vs `val/` vs `eval/`

| prefix | what | where |
|---|---|---|
| `train/` | during training | Training tab |
| `val/` | validation during training | Training tab |
| `eval/<task>/<metric>` | post-training benchmarks | **Eval tab**, on a separate run — see [eval](eval-results.md) |

`val/` answers *is it converging?*. `eval/` answers *how good is the finished model?*.

---

## Naming the run

The run name is the label on your row in the dashboard. All four take `run_name`:

```python
AstrolabeComposerLogger(run_name="LatentBERT-seed0")
AstrolabeLightningLogger(run_name="LatentBERT-seed0")
AstrolabeHFTrainerCallback(run_name="LatentBERT-seed0")
Run(run_name="LatentBERT-seed0")
```

Leave it off and each framework falls back to something sensible:

| framework | fallback |
|---|---|
| Composer | `state.run_name` |
| Lightning | `trainer.logger.name`, then your module's class name |
| HuggingFace | `args.run_name` |
| raw PyTorch | none — pass `run_name` or the run is unnamed |

> `run_name` has **no environment variable**. It is the one identity field astrolabe does
> not override — see below.

---

## Env wins over your arguments

This surprises people, so it is worth stating flatly.

| you pass | env that overrides it |
|---|---|
| `experiment_name=` | `ALIDADE_EXPERIMENT_NAME` |
| `tags=` | `AIM_RUN_TAGS` |
| `aim_url=` | `ALIDADE_AIM_URL` |
| `run_name=` | *nothing — yours always wins* |

**Inside an astrolabe submit, passing `experiment_name=` does nothing.** The engine sets
the env, and the env is authoritative because astrolabe is the one driving the run — it
has to be able to guarantee the experiment a submit's runs land in.

That is why the same script works in both places: submitted, it inherits the submit's
identity; run by hand, your arguments apply. You do not write two versions.

---

## Distributed training

Every callback gates Aim writes on rank zero, detected in this order:

1. `torch.distributed.is_initialized()` → `dist.get_rank() == 0`
2. `RANK` env — set by `torchrun` and most launchers
3. `LOCAL_RANK`
4. otherwise assume rank zero (single process)

Non-zero ranks no-op every Aim interaction. **N processes produce one Aim run**, not N.

---

## It will not crash your training

The cost of a training run is too high to lose to a logging hiccup.

| failure | default | strict mode |
|---|---|---|
| can't connect at startup | one `WARNING`, then no-op for the run | `RuntimeError` |
| a metric write fails | `DEBUG`, once per metric name; others keep flowing | re-raises |
| close fails | silent — data is already streamed | silent |

Strict mode is `ALIDADE_CALLBACK_STRICT=1`. Use it in CI, where silent metric loss is
worse than a red build.

---

## Raw PyTorch in more detail

```python
from alidade_callbacks import Run

with Run(experiment_name="my-experiment", run_name="baseline") as run:
    for step in range(steps):
        run.log_train(loss=loss, lr=lr, step=step)

        if step % eval_every == 0:
            run.log_eval(accuracy=acc, step=step)      # lands under val/

        run.log("custom/thing", value, step=step)      # exact name, no prefix
```

`log_train` prefixes `train/`, `log_eval` prefixes `val/`, `log` writes the name you give
verbatim. The context manager closes the run and records a status; without it, call
`run.close()` yourself.

`run.pause_eval()` / `run.resume()` keep evaluation time out of the wall-clock
measurement, so wall-time comparisons stay apples-to-apples.

---

## Saving checkpoints

Attach a checkpointer alongside the logger and every checkpoint carries the run's
identity — which is what lets a later eval attribute to this training without you passing
a hash. See [checkpoints](checkpoints.md).

```python
trainer = Trainer(
    loggers=[AstrolabeComposerLogger()],
    callbacks=[AstrolabeComposerCheckpointer()],
)
```

---

## Per-framework detail

Cookbooks with the framework-specific edges:

- [Composer](frameworks/composer.md)
- [Lightning](frameworks/lightning.md)
- [HuggingFace Trainer](frameworks/huggingface.md)
- [Raw PyTorch](frameworks/pytorch.md)

---

## See also

- [Checkpoints](checkpoints.md) — provenance that survives to eval time
- [Eval](eval-results.md) — post-training benchmarks
- [Contract](contract.md) — the guarantees, and the metric names we normalize
