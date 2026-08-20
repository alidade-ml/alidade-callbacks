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
