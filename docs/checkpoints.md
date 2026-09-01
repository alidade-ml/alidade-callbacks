# Checkpoints

**What this is for:** so that six months from now, someone holding a checkpoint file can
find out which run produced it.

That question is normally answered by remembering, by filename conventions, or by
guessing from a timestamp. All three fail. A checkpoint written through this library
carries the answer inside it, and [the eval guide](eval-results.md) reads it back — which
is how benchmark results attach to the model that earned them without you passing a hash
around.

This is the shared middle of the two other guides. [Training](training.md) writes
provenance; [eval](eval-results.md) reads it.

---

## The quickest version

Attach a checkpointer next to your logger. Every checkpoint the framework writes then
carries the live run's identity, with no call at the save site.

```python
from alidade_callbacks import AstrolabeComposerLogger, AstrolabeComposerCheckpointer

trainer = Trainer(
    model=...,
    train_dataloader=...,
    loggers=[AstrolabeComposerLogger()],
    callbacks=[AstrolabeComposerCheckpointer()],   # callbacks=, not loggers=
)
```

Later, from an eval script:

```python
from alidade_callbacks import start_eval_run_from_checkpoint

run = start_eval_run_from_checkpoint(checkpoint="ckpt.pt", task_set="glue")
```

You never handled a run hash. The file knew.

---

## What gets written

A `CheckpointMeta` block:

| field | what it is |
|---|---|
| `aim_run_hash` | the Aim run that was live when this was saved — **the field eval attributes on** |
| `submit_id` | the astrolabe submit |
| `experiment` | experiment name |
| `version` | `"v1"`, `"v2"`, … |
| `created_at` | ISO timestamp |
| `derived_from` | set when this came from another checkpoint (see [Transforms](#transforms)) |
| `derivation_chain_length` | how many transforms deep |

Outside an astrolabe submit most of these are `None`, which is fine — `aim_run_hash` is
the one that matters, and it comes from the live Aim run, not from the environment.

Read it back with:

```python
from alidade_callbacks import read_checkpoint_meta

meta = read_checkpoint_meta("ckpt.pt")       # a path, or an already-loaded state dict
meta.aim_run_hash if meta else None          # None when the file carries no provenance
```

`read_checkpoint_meta` is **offline** — it reads the file and never contacts Aim.

---

## Attaching a checkpointer

```python
from alidade_callbacks import AstrolabeComposerCheckpointer
trainer = Trainer(..., loggers=[AstrolabeComposerLogger()],
                        callbacks=[AstrolabeComposerCheckpointer()])
```

```python
from alidade_callbacks import AstrolabeLightningCheckpointer
trainer = Trainer(..., callbacks=[AstrolabeLightningLogger(),
                                  AstrolabeLightningCheckpointer()])
```

```python
from alidade_callbacks import AstrolabeHFCheckpointer
trainer.add_callback(AstrolabeHFTrainerCallback())
trainer.add_callback(AstrolabeHFCheckpointer())
```

Raw PyTorch has no framework to hook, so you call the save yourself:

```python
from alidade_callbacks import Run, save_checkpoint

with Run(experiment_name="my-experiment") as run:
    for step in range(steps):
        run.log_train(loss=loss)
        if step % 1000 == 0:
            save_checkpoint(model.state_dict(), f"ckpt-{step}.pt")
```

`save_checkpoint` picks up the run that is currently open in the process. There is no
argument to pass.

> **The logger has to be attached too.** The checkpointer stamps whichever Aim run is
> live. With no logger there is no run, so `aim_run_hash` is `None` and an eval on that
> file has nothing to attribute to.

### Constructor options

```python
AstrolabeComposerCheckpointer(*, export_formats=None, export_dir=None)
AstrolabeLightningCheckpointer(*, export_formats=None, export_dir=None)
AstrolabeHFCheckpointer(*, embed_in_weights=True, export_formats=None)
```

- **`export_formats`** — additionally write a copy in `"pt"` and/or `"safetensors"`,
  alongside whatever the framework saves in its own format. Use when something downstream
  wants a portable file rather than a framework checkpoint.
- **`export_dir`** — where those copies go. Defaults next to the framework's own saves.
- **`embed_in_weights`** (HuggingFace) — see [Surviving re-serialization](#surviving-re-serialization).

---

## Writing a checkpoint yourself

No framework, no callback — just a state dict and a destination:

```python
from alidade_callbacks import export_checkpoint

export_checkpoint(state, "model.pt", fmt="pt")
export_checkpoint(state, "model.safetensors", fmt="safetensors")
```

`fmt` is `"pt"` or `"safetensors"`. It takes the live run's identity automatically; pass
`meta=` only to override.

> **safetensors needs an extra**: `pip install 'alidade-callbacks[safetensors]'`.
> Without it, `export_checkpoint(..., fmt="safetensors")` raises `ImportError` naming the
> extra. `"pt"` needs only torch.

Where the block lives depends on the format. A `.pt` file gets a top-level
`_astrolabe_meta` key; safetensors gets it in the header. Either way
`read_checkpoint_meta` finds it, and neither disturbs the tensors.

---

## Transforms

Surgery, quantization, extracting a submodule — anything that produces a new checkpoint
from an old one. Use `save_derived_checkpoint` so the new file remembers where it came
from:

```python
from alidade_callbacks import save_derived_checkpoint

save_derived_checkpoint(quantized_state, "model-int8.pt", parent="model.pt")
```

The derived file carries the **original training run's** `aim_run_hash`, plus
`derived_from` and a hop count. Evaluating `model-int8.pt` attributes to the training
that produced the model it came from, rather than to nothing.

`parent` accepts a path, or a `CheckpointMeta` you already read.

> **Why it matters:** a plain `torch.save` of a transformed state dict loses the
> provenance silently. The file still loads, the eval still runs, and the results attach
> to nothing — and you find out months later when you cannot answer which pretrain a
> number came from.

---

## Back-filling provenance

For a file written before any of this existed, or by code that did not use the library:

```python
from alidade_callbacks import stamp_checkpoint

stamp_checkpoint("old-model.safetensors", aim_run_hash="abc123...")
```

This **rewrites the file in place** — the header is replaced and the tensor block is
copied through as raw bytes. Only reach for it deliberately, on a file you own.

> **Do not stamp a downloaded model.** A HuggingFace checkpoint usually lives in a shared
> cache, possibly read-only, used by every project on the machine. To benchmark a model
> astrolabe never trained, use `external_name=` — see
> [the eval guide](eval-results.md#models-astrolabe-never-trained). It records the model
> in Aim and never touches the file.

---

## Surviving re-serialization

Some frameworks rebuild the state dict when they save, which drops a top-level metadata
key. HuggingFace's `save_pretrained` is the common case, including its sharded path.

`AstrolabeHFCheckpointer(embed_in_weights=True)` — the default — handles this by storing
the block as a **tensor buffer on the model**, so it travels with the weights through any
re-serialization, sharding included.

The cost is a small extra tensor in the state dict. To read or remove it:

```python
from alidade_callbacks import read_meta_from_buffer, strip_meta_buffer

meta = read_meta_from_buffer(state_dict)
clean = strip_meta_buffer(state_dict)     # e.g. before publishing weights
```

Set `embed_in_weights=False` if a downstream consumer rejects unexpected buffers. You
then lose provenance across `save_pretrained`.

---

## When there is no provenance

`read_checkpoint_meta` returns `None`. Nothing crashes, and an eval on that file will
refuse to start rather than log results that attach to nothing — see
[the eval guide](eval-results.md#when-nothing-resolves).

Common causes, in rough order:

1. **No logger was attached during training**, so no Aim run existed to stamp.
2. **The file was transformed with a plain save** instead of `save_derived_checkpoint`.
3. **It was downloaded** — it was never ours, and `external_name=` is the answer.
4. **It predates the feature** — `stamp_checkpoint` can back-fill it.

---

## See also

- [Training](training.md) — attaching loggers and checkpointers
- [Eval](eval-results.md) — reading provenance back to attribute benchmark results
- [Contract](contract.md) — what every callback guarantees
