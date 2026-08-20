"""Qualitative model outputs — a few completions, a few generated images.

Every number astrolabe surfaces is a scalar: loss curves, eval tables, cost.
None of them answers the question a researcher asks first when a run finishes,
and the only one a person outside the project understands — **what does it
actually produce?**

Samples rank nothing and are not compared. They exist to be looked at, which
is why they are a distinct run kind rather than an eval with unusual values.

The researcher's own script does inference; this stores and links the results:

    from astrolabe_callbacks import Sample, log_samples

    log_samples(
        checkpoint="ckpt.pt",
        sample_set="sentence-completion",
        samples=[
            Sample(input="The capital of France is", output=" Paris, which…"),
        ],
    )

Attribution — ``checkpoint`` / ``model_run_hash`` / ``external_name`` — is
resolved by the same code eval uses, so a model cannot be attributed one way
in the Eval tab and another in Examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrolabe_callbacks import contract
from astrolabe_callbacks._attribution import AttributionInputError, resolve_parent
from astrolabe_callbacks._identity import ambient_identity, resolve_aim_url

__all__ = ["Sample", "SampleInputError", "log_samples"]


class SampleInputError(ValueError):
    """Raised when a sample helper receives malformed input.

    Surfaces at the researcher's call site rather than writing a half-formed
    Aim run the dashboard cannot make sense of later.
    """


@dataclass(frozen=True, kw_only=True)
class Sample:
    """One model output, and optionally what produced it.

    Both fields are ``Any`` because the pathway serves text and images
    equally: a completion is a ``str``, a generated image is a PIL image /
    tensor / ndarray, and a denoising or style-transfer sample has an *image*
    input.

    Keyword-only is load-bearing rather than stylistic. With both fields
    untyped, ``Sample(prompt, completion)`` binding ``output=prompt`` would
    raise nothing — it would silently record the prompt as the model's output,
    and the error would be visible only to a human reading the tab.

    ``input`` is optional because unconditional generation has none. A set
    where every input is ``None`` renders without an input column, and that is
    the only thing absence changes.
    """

    output: Any
    input: Any | None = None


def _validate(sample_set: str, samples: Any) -> None:
    if not isinstance(sample_set, str) or not sample_set:
        raise SampleInputError("sample_set must be a non-empty string")
    if "/" in sample_set:
        # It becomes a path segment in the metric name, so a slash forks the
        # namespace the dashboard discovers sets by.
        raise SampleInputError(
            f"sample_set must not contain '/' (got {sample_set!r}) — it is a "
            "path segment in the metric name"
        )
    if not isinstance(samples, (list, tuple)) or not samples:
        raise SampleInputError("samples must be a non-empty list of Sample")
    for i, sample in enumerate(samples):
        if not isinstance(sample, Sample):
            raise SampleInputError(
                f"samples[{i}] is {type(sample).__name__}, not a Sample. Wrap "
                "each output with Sample(input=..., output=...) so the pairing "
                "is explicit — a bare list reads as a list of outputs."
            )
        if not isinstance(sample.output, str):
            raise SampleInputError(
                f"samples[{i}].output is {type(sample.output).__name__}; only "
                "text is supported so far. Image payloads are coming."
            )
        if sample.input is not None and not isinstance(sample.input, str):
            raise SampleInputError(
                f"samples[{i}].input is {type(sample.input).__name__}; only "
                "text is supported so far. Image inputs are coming."
            )


def log_samples(
    *,
    sample_set: str,
    samples: list[Sample],
    checkpoint: str | Path | dict[str, Any] | None = None,
    model_run_hash: str | None = None,
    external_name: str | None = None,
    aim_url: str | None = None,
) -> str:
    """Log one batch of model outputs against the run that produced them.

    Returns the hash of the Aim run created.

    Call it again with a different ``sample_set`` to add another batch —
    completions in one, images in another. ``sample_set`` groups one batch the
    way ``task_set`` groups one benchmark suite.

    Parameters
    ----------
    sample_set : str
        Human label for this batch (``"sentence-completion"``, ``"faces"``).
        Becomes ``astrolabe.sample_set`` and a segment of the metric name, so
        it must not contain ``/``.
    samples : list[Sample]
        The outputs, in the order you want them shown. A list rather than a
        dict because sample inputs are not unique — the same prompt at two
        temperatures is a normal thing to log.
    checkpoint : str | Path | dict, optional
        The file you sampled from. Its embedded provenance names the training
        run, so there is no hash at the call site. Read offline; Aim is never
        queried.
    model_run_hash : str, optional
        Explicit parent, when you have it and the artifact does not.
    external_name : str, optional
        Name for a model astrolabe never trained. Registers an entry and
        attributes the samples to it. A checkpoint carrying provenance wins.
    aim_url : str, optional
        Aim tracking URL. ``ASTROLABE_AIM_URL`` wins over this argument.

    Raises
    ------
    SampleInputError
        Malformed arguments — validated before any Aim run is opened, so a bad
        call leaves nothing behind.
    MissingParentError
        Nothing to attribute the samples to. Raised before any run is created:
        an unattributed run lands in Aim and is invisible to the dashboard
        forever.
    """
    from aim import Run, Text

    _validate(sample_set, samples)
    try:
        parent = resolve_parent(
            checkpoint=checkpoint,
            model_run_hash=model_run_hash,
            external_name=external_name,
            aim_url=aim_url,
            what="sample batch",
        )
    except AttributionInputError as exc:
        raise SampleInputError(str(exc)) from None

    identity = ambient_identity()
    run = Run(
        experiment=identity.get(contract.TAG_EXPERIMENT) or f"sample/{sample_set}",
        repo=resolve_aim_url(aim_url),
    )
    try:
        # Identity first, contract tags second: the discovery tags are the only
        # reason the dashboard can see this run, so an unexpected key in the
        # ambient payload must not be able to shadow one.
        for key, value in identity.items():
            run[key] = value
        run[contract.TAG_KIND] = contract.KIND_SAMPLE
        run[contract.TAG_SAMPLE_SET] = sample_set
        run[contract.TAG_MODEL_RUN_HASH] = parent

        # One step per sample, so order and input↔output pairing are structural
        # rather than conventional.
        for i, sample in enumerate(samples):
            if sample.input is not None:
                run.track(
                    Text(sample.input),
                    name=f"sample/{sample_set}/input",
                    step=i,
                )
            run.track(
                Text(sample.output),
                name=f"sample/{sample_set}/output",
                step=i,
            )
        return run.hash
    finally:
        # An unclosed run leaves end_time at zero and the dashboard treats it
        # as in-flight indefinitely.
        run.close()
