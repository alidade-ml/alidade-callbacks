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

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from astrolabe_callbacks import contract
from tests.testbed.harness.assertions import (
    assert_run_closed,
    get_run_tags,
    get_texts,
)
from tests.testbed.harness.sample_driver import SampleDriverConfig

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
