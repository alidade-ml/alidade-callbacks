# Samples

Storing a few actual model outputs, so the run can show what it produces.

Every other number astrolabe surfaces is a scalar: loss curves, eval scores, cost.
None of them answers the question people ask first when a run finishes, and the only
one a person outside the project understands: *what does it actually make?*

Samples are not ranked and not compared. They exist to be looked at. That is why they
are a separate run kind rather than an eval with unusual values.

| you logged | lands on | dashboard |
|---|---|---|
| `train/*`, `val/*` | the training run | Training tab |
| `eval/<task>/<metric>` | a separate eval run | Eval tab |
| `sample/<set>/{input,output}` | a separate sample run | Samples tab |

---

## The shape

Your own script does the inference. This stores the results and links them to the
model that produced them.

```python
from alidade_callbacks import Sample, log_samples

log_samples(
    checkpoint="ckpt.pt",
    sample_set="sentence-completion",
    samples=[
        Sample(input="The capital of France is", output=" Paris, which…"),
        Sample(input="def fib(n):", output="\n    return n if n < 2 else …"),
    ],
)
```

Attribution works exactly as it does for eval: `checkpoint=` reads the training run's
identity out of the file, offline, without contacting Aim. `model_run_hash=` and
`external_name=` are there for the cases eval documents. A model cannot end up
attributed one way on the Eval tab and another on Samples, because it is the same
code deciding.

---

## What a sample may be

Dispatch is on the value you pass, never on a type argument you also pass. A flag
would be a second source of truth that can disagree with the payload, and the
disagreement would show up only as a corrupt-looking sample in the tab.

| you pass | stored as |
|---|---|
| `str` | `aim.Text` |
| PIL image | `aim.Image` |
| `torch.Tensor` | `aim.Image` |
| `numpy.ndarray` (uint8) | `aim.Image` |

Both fields take either. The four useful combinations:

```python
Sample(input="a golden retriever", output=pil_image)   # prompt to image
Sample(input=noisy_tensor, output=clean_tensor)        # denoising, style transfer
Sample(input="The capital of…", output=" Paris")       # completion
Sample(output=generated_image)                         # unconditional generation
```

`input` is optional because unconditional generation has none. A set where every
input is `None` renders without an input column, and that is the only thing its
absence changes.

### Things that are refused, and why

**A `Path` is not read for you.** `aim.Image` will load a `str` as a file path, so
expecting a `Path` to work is reasonable. It is refused anyway: reading files on your
behalf means guessing at format and at what to do when the read fails, and in every
real case you already have the image in memory.

**A float array is refused with instructions.** PIL's own message is
`Cannot handle this data type: (1, 1, 3), <f4`, which does not tell you what to do.
Scale to 0-255 and cast to `uint8`.

**One `sample_set` renders as one kind.** A set that is half text and half images is
a mess to look at, and if you have both you meant two sets. The rule applies to
**outputs only**: a text prompt with an image output is the most common image sample
there is, and mixed inputs within a set are fine.

### No size ceiling

Nothing refuses, downsamples, or warns on a large image. For a model producing
512x512, that *is* the output, and downsampling would destroy the thing you are
looking at. Samples are retained the way every other run's data is retained.
Display-side scaling belongs to the dashboard, which knows the viewport.

---

## Grouping

`sample_set` groups one batch the way `task_set` groups one benchmark suite. Call
`log_samples` again with a different set to add another batch:

```python
log_samples(checkpoint="ckpt.pt", sample_set="faces", samples=[...])
log_samples(checkpoint="ckpt.pt", sample_set="landscapes", samples=[...])
```

It becomes a path segment in the metric name, so it must not contain `/`.

`samples` is a list rather than a dict because sample inputs are not unique: the same
prompt at two temperatures is a normal thing to log. Order is preserved, and input and
output share a step index, so sample *i* pairs structurally rather than by convention.

---

## When it goes wrong

`log_samples` validates everything and encodes every payload **before** it opens an
Aim run, so a bad call leaves nothing behind. A half-tagged sample run would still
appear in the dashboard's discovery query and render nothing, which is worse than an
error at your call site.

| exception | means |
|---|---|
| `SampleInputError` | malformed arguments or an unsupported payload |
| `MissingParentError` | nothing to attribute the samples to |
