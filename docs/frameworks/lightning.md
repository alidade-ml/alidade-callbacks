# PyTorch Lightning

Cookbook for `AstrolabeLightningLogger` — Lightning 2.x integration.

## Minimal example

```python
from lightning.pytorch import Trainer
from alidade_callbacks import AstrolabeLightningLogger

trainer = Trainer(
    callbacks=[AstrolabeLightningLogger()],
)
trainer.fit(model, train_loader, val_loader)
```

## Logging metrics from your LightningModule

Use Lightning's standard `self.log(...)`. Whatever you log lands in Aim under the same name.

**Name your training loss `train/loss`.** Lightning is the one framework where you
choose it: Composer emits `loss/train/total` and HuggingFace emits `loss`, both of
which we rename to `train/loss`, but a name you picked is never rewritten. The
dashboard's primary chart is `train/loss`, so a run that logs plain `loss` charts
correctly under its own name and leaves that chart empty.

```python
import lightning.pytorch as pl

class MyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        # All three lines below land in Aim with the names you chose:
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/perplexity", torch.exp(loss), on_step=True)
        self.log("throughput/samples_per_sec", batch_idx / time_so_far, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_loss = self.loss_fn(self(x), y)
        # val_<x> and val/<x> are re-namespaced under val/ in Aim (v1.0.0+):
        self.log("val_loss", val_loss, on_epoch=True)            # → val/loss
        self.log("val_accuracy", accuracy, on_epoch=True)        # → val/accuracy
        self.log("val/perplexity", torch.exp(val_loss), on_epoch=True)  # → val/perplexity
```

## Full example

```python
from lightning.pytorch import Trainer
from alidade_callbacks import AstrolabeLightningLogger

callback = AstrolabeLightningLogger(
    aim_url="aim://my-aim-server.example.com:43800",
    experiment_name="resnet50-imagenet",
    tags={"thesis": "vision-baselines", "model": "resnet50"},
    run_name="resnet50-bs256-lr0.1",
)

trainer = Trainer(
    max_epochs=90,
    accelerator="gpu",
    devices=8,
    strategy="ddp",
    callbacks=[callback],
)
trainer.fit(model, train_loader, val_loader)
```

## What gets logged to Aim

| Source | Aim metric name |
|---|---|
| `self.log("name", val)` in `training_step` | `name` (passed through unchanged) |
| `self.log("val_<x>", val)` or `self.log("val/<x>", val)` in `validation_step` | `val/<x>` |
| `self.log("loss", val, on_step=True)` | `loss` — your name, **not** auto-renamed to `train/loss`; the primary chart stays empty |
| Synthesized | `wall_time` (training-only elapsed seconds, eval-paused) |

## Run name resolution

If `run_name=` isn't passed at construction, the callback falls back through:

1. `trainer.logger.name` if a Lightning logger is configured
2. `pl_module.__class__.__name__` (e.g. `MyModel`)

Astrolabe's dashboard groups by `astrolabe.experiment` (the experiment name) and shows individual runs by run name, so a meaningful run name helps comparison views.

## Common patterns

### Distributed training (DDP, FSDP, DeepSpeed)

No special setup. Lightning's strategy initializes `torch.distributed`; the callback gates Aim writes on rank-zero automatically.

```python
trainer = Trainer(
    strategy="ddp",        # or "fsdp", "deepspeed_stage_3", etc.
    devices=8,
    callbacks=[AstrolabeLightningLogger()],
)
```

### Coexistence with Lightning's own loggers

`AstrolabeLightningLogger` is a `Callback`, not a `Logger`. It runs alongside whatever loggers you configure (`trainer.loggers=[TensorBoardLogger(...)]`):

```python
from lightning.pytorch.loggers import TensorBoardLogger

trainer = Trainer(
    logger=TensorBoardLogger(save_dir="logs/"),    # primary logger
    callbacks=[AstrolabeLightningLogger()],        # parallel Aim sink
)
```

Aim already ships `aim.pytorch_lightning.AimLogger` for the logger-side Aim integration. Don't combine it with `AstrolabeLightningLogger` — you'd double-write to Aim. Pick one based on which interface you prefer (this callback adds astrolabe tag conventions; `aim.pytorch_lightning.AimLogger` doesn't).

### Manual optimization

If you're using manual optimization (`self.automatic_optimization = False` and call `optimizer.step()` yourself), `training_step` may return `None` instead of a loss. The callback handles this — it just doesn't auto-extract loss in that case. Your `self.log(...)` calls still flow through.

## Gotchas

### `on_step=True` vs `on_epoch=True`

`self.log(name, value)` defaults vary by hook:

- In `training_step`: defaults to `on_step=True, on_epoch=False`
- In `validation_step`: defaults to `on_step=False, on_epoch=True`

If you want a per-batch metric in Aim's time series, pass `on_step=True` explicitly. If you only set `on_epoch=True`, Lightning waits until epoch end to populate `trainer.callback_metrics` — you'll see the metric in Aim only at epoch boundaries.

### `val_` vs `val/` prefix is up to you

Lightning users split: some prefix with underscore (`self.log("val_loss", ...)`), some with slash (`self.log("val/loss", ...)`). Both work — both are re-namespaced under `val/` in Aim. The `eval/` namespace is reserved for post-training benchmark suites and is never written during training.

### Eval-pause precision

`wall_time` is paused via Lightning's `on_validation_start` and resumed in `on_validation_end`. If you have a non-standard eval loop (e.g. custom `validate_dataloader` triggered manually), the pause/resume may not fire and eval time will leak into `wall_time`. For most users this isn't a concern.

### `on_exception` is Lightning ≥ 1.9

The `on_exception` hook for marking runs as `failed` is available from Lightning 1.9 onward. On older versions, the run still closes but with `astrolabe.status="completed"` regardless of how training ended.

### Migrating from `astrolabe-composer-callback`

If you were using `AstrolabeLogger` (a Composer callback), this is a different package. Lightning users were never targeted by `astrolabe-composer-callback`. Just install `alidade-callbacks[lightning]` fresh.
