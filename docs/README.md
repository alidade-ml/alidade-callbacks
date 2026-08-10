# Getting your data into astrolabe

`astrolabe-callbacks` is the one library your training and eval code imports. It streams
metrics to Aim with astrolabe's conventions applied, so your runs show up on the
dashboard attached to the right experiment, the right submit, and the right model.

It is deliberately small. The base install pulls `aim` and `loguru` and nothing else —
your training repo never depends on the orchestration framework.

---

## Which guide do you want

| you are… | read |
|---|---|
| logging metrics while a model trains | **[Training](training.md)** |
| saving checkpoints you will evaluate later | **[Checkpoints](checkpoints.md)** |
| logging benchmark results after training | **[Eval](eval-results.md)** |

They fit together in that order. Training writes metrics *and* stamps checkpoints with
the run that produced them; eval reads that stamp back so benchmark results attach to the
right model without you passing identifiers around.

```
   training ──writes──▶ checkpoint ──read by──▶ eval
      │                                          │
      └────── train/ val/ ──▶ Training tab       └── eval/ ──▶ Eval tab
```

---

## Install

```bash
pip install astrolabe-callbacks                 # raw PyTorch, and all eval helpers
pip install 'astrolabe-callbacks[composer]'     # MosaicML Composer
pip install 'astrolabe-callbacks[lightning]'    # PyTorch Lightning
pip install 'astrolabe-callbacks[hf]'           # HuggingFace Trainer
pip install 'astrolabe-callbacks[all]'          # everything
```

Add `[safetensors]` if you export checkpoints in that format.

Eval scripts need **only the base install** — no framework extra. They talk to Aim, not
to a trainer.

---

## The 30-second version

Training:

```python
from astrolabe_callbacks import AstrolabeComposerLogger, AstrolabeComposerCheckpointer

trainer = Trainer(
    model=...,
    loggers=[AstrolabeComposerLogger()],        # metrics
    callbacks=[AstrolabeComposerCheckpointer()],  # provenance on every save
)
```

Eval, later, possibly in a different repo:

```python
from astrolabe_callbacks import start_eval_run_from_checkpoint

run = start_eval_run_from_checkpoint(checkpoint="ckpt.pt", task_set="glue")
run.track(0.822, name="eval/cola/matthews", step=0)
run.close()
```

Nothing was passed between them but the file.

---

## Two things worth knowing early

**Inside an astrolabe submit, the environment wins.** The engine sets
`ASTROLABE_EXPERIMENT_NAME` and `AIM_RUN_TAGS`, and those override the matching
constructor arguments. That is what lets the same script work submitted and standalone —
but it means `experiment_name=` silently does nothing inside a submit. Details in
[training](training.md#env-wins-over-your-arguments).

**Logging never crashes training.** A dead Aim server costs you a warning and the
metrics, not the run. Set `ASTROLABE_CALLBACK_STRICT=1` to invert that in CI, where
losing metrics silently is the worse failure.

---

## Reference

- [Contract](contract.md) — what every callback guarantees, and which framework-owned
  metric names get normalized
- Per-framework cookbooks: [Composer](frameworks/composer.md) ·
  [Lightning](frameworks/lightning.md) · [HuggingFace](frameworks/huggingface.md) ·
  [Raw PyTorch](frameworks/pytorch.md)
