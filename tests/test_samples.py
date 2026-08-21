"""`log_samples` — qualitative model outputs, stored and linked.

Two properties carry most of the weight here, and both are about failing
before anything is written:

- **Validation runs before any Aim run is opened.** A half-tagged sample run
  appears in the dashboard's discovery query and renders nothing, which is
  worse than a crash at the call site.
- **Attribution is eval's**, not a second implementation. The tests that
  matter for that live in `test_eval_results.py`; here we check that the
  wiring reaches it and that its errors surface as `SampleInputError`.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from astrolabe_callbacks import contract
from astrolabe_callbacks._attribution import MissingParentError
from astrolabe_callbacks.samples import Sample, SampleInputError, log_samples


def _run_mock(run_hash: str = "s-hash") -> MagicMock:
    run = MagicMock()
    run.hash = run_hash
    return run


def _log(samples, *, sample_set="completions", **kw):
    """Call log_samples with aim.Run patched; return (result, run mock)."""
    run = _run_mock()
    with patch("aim.Run", return_value=run), patch("aim.Text", side_effect=lambda v: v):
        result = log_samples(
            sample_set=sample_set,
            samples=samples,
            model_run_hash=kw.pop("model_run_hash", "parent-hash"),
            **kw,
        )
    return result, run


def _tracked(run) -> list[tuple[str, int, object]]:
    """(name, step, value) for every track() call, in order."""
    return [
        (c.kwargs["name"], c.kwargs["step"], c.args[0])
        for c in run.track.call_args_list
    ]


class TestTheShapeTheDataclassExistsToEnforce:
    """A bare `samples=[...]` reads as a list of outputs. It must not work."""

    @pytest.fixture(autouse=True)
    def _no_run(self):
        with patch("aim.Run") as run:
            self.run = run
            yield
            assert not run.called, "an Aim run was created despite invalid input"

    def test_a_list_of_tuples_names_Sample(self):
        with pytest.raises(SampleInputError, match="Sample"):
            log_samples(
                sample_set="x",
                samples=[("prompt", "completion")],
                model_run_hash="h",
            )

    def test_a_list_of_strings_names_Sample(self):
        with pytest.raises(SampleInputError, match="Sample"):
            log_samples(sample_set="x", samples=["just an output"], model_run_hash="h")

    def test_an_empty_list_raises(self):
        with pytest.raises(SampleInputError, match="non-empty"):
            log_samples(sample_set="x", samples=[], model_run_hash="h")

    def test_a_slash_in_sample_set_raises(self):
        """It is a path segment in the metric name; a slash forks the
        namespace the dashboard discovers sets by."""
        with pytest.raises(SampleInputError, match="'/'"):
            log_samples(
                sample_set="a/b",
                samples=[Sample(output="x")],
                model_run_hash="h",
            )

    def test_an_empty_sample_set_raises(self):
        with pytest.raises(SampleInputError, match="non-empty string"):
            log_samples(sample_set="", samples=[Sample(output="x")], model_run_hash="h")

    def test_a_non_text_output_says_images_are_coming(self):
        """Not 'invalid type', which would read as permanent."""
        with pytest.raises(SampleInputError, match="[Ii]mage"):
            log_samples(
                sample_set="x",
                samples=[Sample(output=object())],
                model_run_hash="h",
            )


class TestSampleIsKeywordOnly:
    """With both fields untyped, a positional mixup raises nothing — it
    records the prompt as the model's output, visible only to a human."""

    def test_positional_construction_is_rejected(self):
        with pytest.raises(TypeError):
            Sample("prompt", "completion")

    def test_input_defaults_to_none(self):
        assert Sample(output="x").input is None


class TestWhatLandsInAim:

    def test_order_is_preserved_and_pairing_is_the_step(self):
        _, run = _log([
            Sample(input="in-0", output="out-0"),
            Sample(input="in-1", output="out-1"),
        ])
        assert _tracked(run) == [
            ("sample/completions/input", 0, "in-0"),
            ("sample/completions/output", 0, "out-0"),
            ("sample/completions/input", 1, "in-1"),
            ("sample/completions/output", 1, "out-1"),
        ]

    def test_absent_input_tracks_no_input_sequence(self):
        _, run = _log([Sample(output="only-output")])
        assert _tracked(run) == [("sample/completions/output", 0, "only-output")]

    def test_duplicate_inputs_stay_distinct(self):
        """The reason this takes a list where log_eval_table takes a dict:
        the same prompt at two temperatures is a normal thing to log."""
        _, run = _log([
            Sample(input="same", output="first"),
            Sample(input="same", output="second"),
        ])
        outputs = [v for n, _, v in _tracked(run) if n.endswith("/output")]
        assert outputs == ["first", "second"]

    def test_the_run_carries_the_discovery_tags(self):
        _, run = _log([Sample(output="x")], model_run_hash="parent-123")
        tags = {c.args[0]: c.args[1] for c in run.__setitem__.call_args_list}
        assert tags[contract.TAG_KIND] == contract.KIND_SAMPLE
        assert tags[contract.TAG_SAMPLE_SET] == "completions"
        assert tags[contract.TAG_MODEL_RUN_HASH] == "parent-123"

    def test_the_run_is_closed(self):
        """An unclosed run leaves end_time at zero and the dashboard treats
        it as in-flight indefinitely."""
        _, run = _log([Sample(output="x")])
        run.close.assert_called_once()

    def test_two_sets_are_independent(self):
        _, a = _log([Sample(output="x")], sample_set="completions")
        _, b = _log([Sample(output="y")], sample_set="images")
        assert _tracked(a)[0][0] == "sample/completions/output"
        assert _tracked(b)[0][0] == "sample/images/output"

    def test_it_returns_the_run_hash(self):
        result, _ = _log([Sample(output="x")])
        assert result == "s-hash"


class TestAttributionIsEvals:
    """Wiring, not behaviour — the resolver's own tests live with eval."""

    def test_nothing_to_attribute_to_raises_before_any_run(self):
        with patch("aim.Run") as run:
            with pytest.raises(MissingParentError):
                log_samples(sample_set="x", samples=[Sample(output="y")])
            assert not run.called

    def test_a_checkpoint_resolves_without_a_hash_at_the_call_site(self):
        with patch(
            "astrolabe_callbacks._attribution._parent_run_hash",
            return_value="from-ckpt",
        ):
            _, run = _log(
                [Sample(output="x")], model_run_hash=None, checkpoint="ckpt.pt"
            )
        tags = {c.args[0]: c.args[1] for c in run.__setitem__.call_args_list}
        assert tags[contract.TAG_MODEL_RUN_HASH] == "from-ckpt"

    def test_a_malformed_attribution_argument_is_a_SampleInputError(self):
        """The resolver raises AttributionInputError; this surface owns its
        own error type, so a researcher catches one thing."""
        with patch("aim.Run") as run:
            with pytest.raises(SampleInputError):
                log_samples(
                    sample_set="x",
                    samples=[Sample(output="y")],
                    model_run_hash="",
                )
            assert not run.called


def _make_pil(size=(4, 3), color=(10, 200, 30)):
    from PIL import Image as PILImage

    return PILImage.new("RGB", size, color=color)


def _make_ndarray(shape=(3, 4, 3)):
    import numpy as np

    arr = np.zeros(shape, dtype=np.uint8)
    arr[:, :, 1] = 128
    return arr


def _log_real_payloads(samples, *, sample_set="faces", **kw):
    """Like ``_log`` but leaves aim.Text/aim.Image unpatched.

    Dispatch is the thing under test here, so the real encoders have to run —
    a patched ``aim.Image`` would make these tests measure the mock.
    """
    run = _run_mock()
    with patch("aim.Run", return_value=run):
        result = log_samples(
            sample_set=sample_set,
            samples=samples,
            model_run_hash=kw.pop("model_run_hash", "parent-hash"),
            **kw,
        )
    return result, run


class TestOneSampleSetRendersAsOneKind:
    def test_mixed_output_types_raise_naming_both(self):
        with pytest.raises(SampleInputError, match=r"samples\[1\].output is image"):
            _log_real_payloads([
                Sample(output="a completion"),
                Sample(output=_make_pil()),
            ])

    def test_the_message_names_the_earlier_sample_too(self):
        with pytest.raises(SampleInputError, match=r"samples\[0\].output is text"):
            _log_real_payloads([
                Sample(output="a completion"),
                Sample(output=_make_pil()),
            ])

    def test_image_first_then_text_also_raises(self):
        with pytest.raises(SampleInputError, match="One sample_set renders as one kind"):
            _log_real_payloads([
                Sample(output=_make_pil()),
                Sample(output="a completion"),
            ])

    def test_a_text_input_with_an_image_output_is_accepted(self):
        # The prompt-to-image case: the single most common image sample there
        # is. The mixed-output rule must not reach inputs, and this test is
        # separate from the mixed-output ones precisely because reading
        # `input` too would reject the majority case while still passing them.
        from aim import Image as AimImage
        from aim import Text as AimText

        _, run = _log_real_payloads([
            Sample(input="a golden retriever", output=_make_pil()),
        ])
        tracked = _tracked(run)
        kinds = {name: type(value) for name, _step, value in tracked}
        assert kinds["sample/faces/input"] is AimText
        assert kinds["sample/faces/output"] is AimImage

    def test_mixed_input_types_are_allowed(self):
        # Deliberate: one set may pair a text prompt with one image and an
        # image prompt with another. Only the outputs must agree.
        _, run = _log_real_payloads([
            Sample(input="a prompt", output=_make_pil()),
            Sample(input=_make_pil(), output=_make_pil()),
        ])
        assert len(_tracked(run)) == 4


class TestWhatCountsAsAPayload:
    def test_an_image_input_and_image_output_both_encode(self):
        # The denoising / style-transfer shape.
        from aim import Image as AimImage

        _, run = _log_real_payloads([
            Sample(input=_make_ndarray(), output=_make_ndarray()),
        ])
        tracked = _tracked(run)
        assert len(tracked) == 2
        assert all(type(v) is AimImage for _n, _s, v in tracked)

    def test_unsupported_type_raises_without_promising_images(self):
        with pytest.raises(SampleInputError) as exc:
            _log_real_payloads([Sample(output=object())])
        message = str(exc.value)
        assert "not text and not an image" in message
        # 02 said "Image payloads are coming." They are here; the promise
        # would now be a lie.
        assert "coming" not in message

    def test_a_path_is_refused_rather_than_read(self):
        # aim.Image loads a str as a file path, so expecting Path to work is
        # coherent. Refused deliberately: reading files on the researcher's
        # behalf guesses format and failure handling.
        from pathlib import Path

        with pytest.raises(SampleInputError, match="does not read files"):
            _log_real_payloads([Sample(output=Path("/tmp/nope.png"))])

    def test_a_float_array_says_what_to_do_about_it(self):
        # PIL's own message is "Cannot handle this data type: (1, 1, 3), <f4",
        # which does not tell a researcher to cast to uint8. A float array is
        # a very common way to be holding an image.
        import numpy as np

        with pytest.raises(SampleInputError, match="uint8"):
            _log_real_payloads([Sample(output=np.zeros((4, 4, 3), dtype="float32"))])

    def test_nothing_is_written_when_a_payload_is_bad(self):
        # The property 02 established, still holding for image dispatch:
        # encoding happens before the Run is opened.
        run = _run_mock()
        with patch("aim.Run", return_value=run) as run_ctor:
            with pytest.raises(SampleInputError):
                log_samples(
                    sample_set="faces",
                    samples=[Sample(output=_make_pil()), Sample(output="text")],
                    model_run_hash="parent-hash",
                )
        run_ctor.assert_not_called()
        run.track.assert_not_called()


class TestTextStillBehavesAsBefore:
    def test_text_only_batches_are_unchanged_by_dispatch(self):
        from aim import Text as AimText

        _, run = _log_real_payloads(
            [
                Sample(input="prompt one", output="completion one"),
                Sample(output="completion two"),
            ],
            sample_set="completions",
        )
        tracked = _tracked(run)
        assert [(n, s) for n, s, _v in tracked] == [
            ("sample/completions/input", 0),
            ("sample/completions/output", 0),
            ("sample/completions/output", 1),
        ]
        assert all(type(v) is AimText for _n, _s, v in tracked)


class TestTheBaseInstallStaysThin:
    def test_image_payloads_do_not_require_torch(self, monkeypatch):
        # The base install is aim + loguru. A top-level `import torch` behind
        # an isinstance check would break every training repo without it.
        # Poison the module so any import of it raises.
        monkeypatch.setitem(sys.modules, "torch", None)
        _, run = _log_real_payloads([
            Sample(input="a prompt", output=_make_pil()),
            Sample(output=_make_ndarray()),
        ])
        assert len(_tracked(run)) == 3


class TestTheSequenceNameComesFromTheContract:
    """The names in the assertions above are the wire contract, and they are
    deliberately spelled out there so a change to them fails loudly.

    These tests answer the other half: that ``log_samples`` *derives* the name
    from ``contract.format_sample_sequence_name`` rather than reproducing it. A
    hardcoded f-string satisfies every literal assertion in this file forever,
    including after the engine renames the template.
    """

    def test_the_tracked_name_follows_the_contract_template(self, monkeypatch):
        # If samples.py builds the name itself, the tracked names ignore this
        # and the assertion fails on the original "sample/..." strings.
        monkeypatch.setattr(
            contract, "SAMPLE_SEQUENCE_TEMPLATE", "moved/{sample_set}/{role}"
        )
        _, run = _log([Sample(input="in", output="out")])
        assert [name for name, _, _ in _tracked(run)] == [
            "moved/completions/input",
            "moved/completions/output",
        ]

    def test_the_roles_are_the_contract_constants(self, monkeypatch):
        # Two derivations can agree on the template and disagree on the role.
        monkeypatch.setattr(contract, "SAMPLE_ROLE_INPUT", "prompt")
        monkeypatch.setattr(contract, "SAMPLE_ROLE_OUTPUT", "completion")
        _, run = _log([Sample(input="in", output="out")])
        assert [name for name, _, _ in _tracked(run)] == [
            "sample/completions/prompt",
            "sample/completions/completion",
        ]

    def test_a_slash_in_the_sample_set_raises_the_producers_error(self):
        # The formatter also rejects this, but later and with a worse message.
        # The producer's validation must stay the thing a user sees.
        with pytest.raises(SampleInputError) as exc:
            _log([Sample(output="x")], sample_set="a/b")
        assert "sample_set" in str(exc.value)

    def test_the_vendored_contract_is_current(self):
        # The hash pin proves the vendored file is untampered, not that it is
        # current — it compares the copy against its own recorded hash. This
        # pins the version the sample sequence contract arrived in.
        assert contract.CONTRACT_VERSION >= "1.8.0"
        assert hasattr(contract, "format_sample_sequence_name")
