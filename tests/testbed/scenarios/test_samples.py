"""``log_samples`` against a real Aim server.

Unit tests assert what gets *tracked*. They cannot answer the question that
matters here: does an ``aim.Text`` written on one sequence come back as text,
and do two sequences sharing a step index still pair after a round trip
through Aim's storage layer? A mock returns whatever you tell it to.

This is also the only place the sample run is proved discoverable — the tags
the dashboard queries by are meaningless until something reads them back off
a real repo.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from astrolabe_callbacks import contract
from tests.testbed.harness.assertions import (
    assert_run_closed,
    assert_run_experiment,
    get_run_tags,
    get_images,
    get_texts,
)
from tests.testbed.harness.sample_driver import (
    SampleDriverConfig,
    make_pattern,
)

if TYPE_CHECKING:
    from tests.testbed.harness.compose import TestbedHandle

pytestmark = pytest.mark.testbed

FAKE_PARENT_HASH = "0" * 24


def _config(testbed, *, samples, sample_set="completions", **kw) -> SampleDriverConfig:
    return SampleDriverConfig(
        aim_url=testbed.aim_url_from_client,
        sample_set=sample_set,
        samples=samples,
        model_run_hash=kw.pop("model_run_hash", FAKE_PARENT_HASH),
        **kw,
    )


class TestSamplesSurviveARealRoundTrip:

    def test_pairing_and_order_survive(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """Input and output share a step, so sample *i* pairs structurally.

        The property under test is not "two sequences exist" — it is that
        after Aim has written and re-read them, step *i* on one still means
        the same sample as step *i* on the other.
        """
        result = run_sample_driver(
            _config(
                testbed,
                samples=[
                    ["The capital of France is", " Paris"],
                    ["def fib(n):", "\n    return n"],
                ],
            )
        )
        assert result.exit_code == 0, result.stderr
        assert result.sample_run_hash is not None

        ins = get_texts(aim_repo, result.sample_run_hash, "sample/completions/input")
        outs = get_texts(aim_repo, result.sample_run_hash, "sample/completions/output")
        assert [s for s, _ in ins] == [0, 1]
        assert [s for s, _ in outs] == [0, 1]
        assert dict(ins)[0] == "The capital of France is"
        assert dict(outs)[0] == " Paris"
        assert dict(ins)[1] == "def fib(n):"

    def test_an_absent_input_writes_no_input_sequence(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """Unconditional generation. The output sequence is unaffected."""
        result = run_sample_driver(
            _config(testbed, samples=[[None, "an unprompted output"]])
        )
        assert result.exit_code == 0, result.stderr
        outs = get_texts(aim_repo, result.sample_run_hash, "sample/completions/output")
        assert dict(outs)[0] == "an unprompted output"
        assert get_texts(aim_repo, result.sample_run_hash, "sample/completions/input") == []

    def test_the_run_is_discoverable_and_closed(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """The tags are the only reason the dashboard can find this run.

        Closed matters for a second reason: an open run keeps ``end_time`` at
        zero and the dashboard treats it as in-flight indefinitely.
        """
        result = run_sample_driver(_config(testbed, samples=[["in", "out"]]))
        assert result.exit_code == 0, result.stderr

        tags = get_run_tags(aim_repo, result.sample_run_hash)
        assert tags.get(contract.TAG_KIND) == contract.KIND_SAMPLE
        assert tags.get(contract.TAG_SAMPLE_SET) == "completions"
        assert tags.get(contract.TAG_MODEL_RUN_HASH) == FAKE_PARENT_HASH
        assert_run_closed(aim_repo, result.sample_run_hash)

    def test_two_sets_do_not_collide(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """A second batch of something different is a second call."""
        a = run_sample_driver(
            _config(testbed, samples=[["p", "completion"]], sample_set="completions")
        )
        b = run_sample_driver(
            _config(testbed, samples=[["p", "caption"]], sample_set="captions")
        )
        assert a.exit_code == 0 and b.exit_code == 0
        assert a.sample_run_hash != b.sample_run_hash
        assert dict(get_texts(aim_repo, a.sample_run_hash, "sample/completions/output"))[0] == "completion"
        assert dict(get_texts(aim_repo, b.sample_run_hash, "sample/captions/output"))[0] == "caption"


class TestNothingIsWrittenWithoutAParent:

    def test_no_attribution_exits_43_and_writes_nothing(
        self, testbed: "TestbedHandle", run_sample_driver
    ) -> None:
        """An unattributed run lands in Aim and is invisible to the dashboard
        forever, so the refusal has to happen before anything is created."""
        result = run_sample_driver(
            _config(testbed, samples=[["in", "out"]], model_run_hash=None)
        )
        assert result.exit_code == 43, result.stderr
        assert result.sample_run_hash is None


class TestSubmitIdentity:
    """A sample run produced inside a submit inherits that submit's identity.

    The engine exports ``AIM_RUN_TAGS`` and ``ASTROLABE_EXPERIMENT_NAME`` into
    every step env. ``log_samples`` reads them — and slice 02 shipped that with
    a comment explaining why it matters and no test that it happens.

    It cannot be a unit test. Filing is a property Aim resolves at run-open, so
    a mocked ``Run`` cannot show which experiment a run actually landed in.
    Eval learned this as a whole merged milestone; this is the same lesson,
    borrowed rather than re-paid.
    """

    SUBMIT_ENV = {
        "ASTROLABE_EXPERIMENT_NAME": "latent-bert",
        "AIM_RUN_TAGS": (
            "astrolabe.submit_id=s-testbed-1,astrolabe.version=v3,"
            "astrolabe.user=nathan,astrolabe.experiment=latent-bert"
        ),
    }

    def test_samples_land_in_the_submitting_experiment(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """The whole point: a submit's samples sit on the submit's page.

        Filed under ``sample/<set>`` they land on a page named for the batch,
        which no experiment view queries.
        """
        result = run_sample_driver(
            replace(
                _config(testbed, samples=[["in", "out"]]),
                submit_env=dict(self.SUBMIT_ENV),
            )
        )
        assert result.exit_code == 0, result.stderr
        assert_run_experiment(aim_repo, result.sample_run_hash, "latent-bert")

    def test_the_submits_tags_land_on_the_run(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """Without these a sample is unattributable — no submitter to filter
        on, and no submit to bill the GPU time against."""
        result = run_sample_driver(
            replace(
                _config(testbed, samples=[["in", "out"]]),
                submit_env=dict(self.SUBMIT_ENV),
            )
        )
        assert result.exit_code == 0, result.stderr
        tags = get_run_tags(aim_repo, result.sample_run_hash)
        assert tags.get("astrolabe.submit_id") == "s-testbed-1"
        assert tags.get("astrolabe.version") == "v3"
        assert tags.get("astrolabe.user") == "nathan"

    def test_identity_does_not_shadow_the_discovery_tags(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """Ambient identity is applied first, contract tags second.

        An unexpected key in the env payload must not be able to overwrite the
        tags the dashboard finds this run by — those are the only reason it is
        visible at all.
        """
        hostile = dict(self.SUBMIT_ENV)
        hostile["AIM_RUN_TAGS"] += ",astrolabe.kind=training"
        result = run_sample_driver(
            replace(
                _config(testbed, samples=[["in", "out"]]), submit_env=hostile
            )
        )
        assert result.exit_code == 0, result.stderr
        tags = get_run_tags(aim_repo, result.sample_run_hash)
        assert tags.get(contract.TAG_KIND) == contract.KIND_SAMPLE

    def test_outside_a_submit_it_falls_back_to_the_batch_name(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """Ad-hoc sampling has no experiment to inherit. The fallback is
        correct behaviour, not an accident, and must not become a crash."""
        result = run_sample_driver(
            _config(testbed, samples=[["in", "out"]], sample_set="adhoc")
        )
        assert result.exit_code == 0, result.stderr
        assert_run_experiment(aim_repo, result.sample_run_hash, "sample/adhoc")


class TestCheckpointResolutionAgainstRealFiles:
    """``checkpoint=`` is the headline ergonomic — no hash at the call site.

    Slice 02 asserted it with ``_parent_run_hash`` patched, which proves the
    wiring and nothing about whether a real file's provenance can be read.
    """

    PT_HASH = "a" * 24
    ST_HASH = "b" * 24

    def test_a_real_pt_checkpoint_resolves(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        result = run_sample_driver(
            replace(
                _config(testbed, samples=[["in", "out"]], model_run_hash=None),
                checkpoint_path="/tmp/sample-parity/model.pt",
                create_pt_with_hash=self.PT_HASH,
            )
        )
        assert result.exit_code == 0, result.stderr
        tags = get_run_tags(aim_repo, result.sample_run_hash)
        assert tags.get(contract.TAG_MODEL_RUN_HASH) == self.PT_HASH

    def test_a_real_safetensors_checkpoint_resolves(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        result = run_sample_driver(
            replace(
                _config(testbed, samples=[["in", "out"]], model_run_hash=None),
                checkpoint_path="/tmp/sample-parity/model.safetensors",
                create_safetensors_with_hash=self.ST_HASH,
            )
        )
        assert result.exit_code == 0, result.stderr
        tags = get_run_tags(aim_repo, result.sample_run_hash)
        assert tags.get(contract.TAG_MODEL_RUN_HASH) == self.ST_HASH

    def test_an_explicit_hash_wins_over_the_file(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """Resolution order, proved where the two sources actually disagree."""
        result = run_sample_driver(
            replace(
                _config(
                    testbed, samples=[["in", "out"]], model_run_hash="c" * 24
                ),
                checkpoint_path="/tmp/sample-parity/conflict.pt",
                create_pt_with_hash=self.PT_HASH,
            )
        )
        assert result.exit_code == 0, result.stderr
        tags = get_run_tags(aim_repo, result.sample_run_hash)
        assert tags.get(contract.TAG_MODEL_RUN_HASH) == "c" * 24


def _image(w: int, h: int, seed: int) -> dict:
    """A sample element the driver turns into a real image in the container."""
    return {"image": {"w": w, "h": h, "seed": seed}}


class TestImagesSurviveARealRoundTrip:
    """The half a mock cannot show.

    ``aim.Image`` encodes to PNG on the way in and decodes on the way out.
    Whether the pixels that come back are the pixels that went in is a fact
    about Aim's storage layer, and a patched ``aim.Image`` returns whatever
    the test told it to.
    """

    def test_an_image_output_returns_as_an_image_with_the_same_pixels(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        import numpy as np

        result = run_sample_driver(
            _config(
                testbed,
                sample_set="faces",
                samples=[["a golden retriever", _image(8, 6, seed=7)]],
            )
        )
        assert result.exit_code == 0, result.stderr

        images = get_images(aim_repo, result.sample_run_hash, "sample/faces/output")
        assert [step for step, _ in images] == [0]
        got = dict(images)[0]
        expected = make_pattern(8, 6, seed=7)
        # Shape first: a transpose would otherwise fail the value comparison
        # with an unreadable broadcast error rather than naming the problem.
        assert got.shape == expected.shape == (6, 8, 3)
        assert np.array_equal(got, expected), "pixels changed across the round trip"

    def test_a_text_prompt_and_an_image_output_land_in_their_own_sequences(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """The prompt-to-image case, end to end.

        The input is text and the output is an image *on the same sample*, so
        this is the scenario that would break if the one-kind-per-set rule
        ever reached inputs.
        """
        result = run_sample_driver(
            _config(
                testbed,
                sample_set="faces",
                samples=[["a golden retriever", _image(4, 4, seed=1)]],
            )
        )
        assert result.exit_code == 0, result.stderr

        texts = get_texts(aim_repo, result.sample_run_hash, "sample/faces/input")
        assert dict(texts)[0] == "a golden retriever"
        assert len(get_images(aim_repo, result.sample_run_hash, "sample/faces/output")) == 1
        # And the output is NOT retrievable as text — i.e. it was stored as an
        # image, not as a repr of one, which is the failure this slice exists
        # to prevent.
        assert get_texts(aim_repo, result.sample_run_hash, "sample/faces/output") == []

    def test_an_image_input_and_image_output_both_round_trip(
        self, testbed: "TestbedHandle", aim_repo: Path, run_sample_driver
    ) -> None:
        """The denoising shape: image in, image out, paired by step."""
        import numpy as np

        result = run_sample_driver(
            _config(
                testbed,
                sample_set="denoise",
                samples=[
                    [_image(5, 4, seed=11), _image(5, 4, seed=12)],
                    [_image(5, 4, seed=13), _image(5, 4, seed=14)],
                ],
            )
        )
        assert result.exit_code == 0, result.stderr

        ins = dict(get_images(aim_repo, result.sample_run_hash, "sample/denoise/input"))
        outs = dict(get_images(aim_repo, result.sample_run_hash, "sample/denoise/output"))
        assert sorted(ins) == sorted(outs) == [0, 1]
        # Pairing is the point: step 1's input must be seed 13, not seed 11.
        assert np.array_equal(ins[1], make_pattern(5, 4, seed=13))
        assert np.array_equal(outs[1], make_pattern(5, 4, seed=14))
