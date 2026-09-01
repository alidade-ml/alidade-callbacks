import pytest
from alidade_callbacks import _attribution
from alidade_callbacks import contract
from unittest.mock import MagicMock


class TestExternalModelIdentityIsSubmitScoped:
    """One entry per submit, not per call.

    Scoring one downloaded model on GLUE, MMLU and BEIR as three steps
    should produce three results about one model. Before the registry it
    produced three models with one result each, and ``--include`` by name
    then resolved to whichever entry was newest and silently returned a
    third of the evidence.
    """

    @pytest.fixture
    def registry(self, tmp_path, monkeypatch):
        path = tmp_path / "external-models.json"
        monkeypatch.setenv(contract.ENV_EXTERNAL_MODELS, str(path))
        return path

    def test_a_second_call_reuses_the_first_entry(self, registry, monkeypatch):
        minted = []

        def fake_run(**kwargs):
            r = MagicMock()
            r.hash = f"hash-{len(minted)}"
            minted.append(r.hash)
            return r

        monkeypatch.setattr("aim.Run", fake_run)
        first = _attribution.mint_model_entry("roberta-base", None)
        second = _attribution.mint_model_entry("roberta-base", None)

        assert first == second, "a second call minted a new identity"
        assert len(minted) == 1, f"minted {len(minted)} entries, want 1"

    def test_different_names_get_different_entries(self, registry, monkeypatch):
        counter = {"n": 0}

        def fake_run(**kwargs):
            r = MagicMock()
            counter["n"] += 1
            r.hash = f"hash-{counter['n']}"
            return r

        monkeypatch.setattr("aim.Run", fake_run)
        a = _attribution.mint_model_entry("roberta-base", None)
        b = _attribution.mint_model_entry("bert-base", None)
        assert a != b, "two different models collapsed to one entry"

    def test_no_registry_outside_a_submit_still_mints(self, monkeypatch):
        # Outside orchestration there is no submit to scope to, and every
        # call minting its own entry is the honest answer rather than an
        # error.
        monkeypatch.delenv(contract.ENV_EXTERNAL_MODELS, raising=False)
        monkeypatch.setattr("aim.Run", lambda **kw: MagicMock(hash="h"))
        assert _attribution.mint_model_entry("roberta-base", None) == "h"

    def test_a_corrupt_registry_costs_a_duplicate_not_the_result(
        self, registry, monkeypatch
    ):
        # Best-effort in every failure mode. Losing the file costs a
        # duplicate entry; raising would cost the benchmark.
        registry.write_text("{ this is not json")
        monkeypatch.setattr("aim.Run", lambda **kw: MagicMock(hash="fresh"))
        assert _attribution.mint_model_entry("roberta-base", None) == "fresh"

    def test_an_unwritable_registry_does_not_fail_the_call(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(
            contract.ENV_EXTERNAL_MODELS, str(tmp_path / "nope" / "x.json")
        )
        monkeypatch.setattr("aim.Run", lambda **kw: MagicMock(hash="h2"))
        assert _attribution.mint_model_entry("roberta-base", None) == "h2"

    def test_a_reader_never_sees_a_half_written_registry(self, registry, monkeypatch):
        # Written to a temp file and renamed. Asserted by checking no
        # partial file is left behind after a write.
        monkeypatch.setattr("aim.Run", lambda **kw: MagicMock(hash="h3"))
        _attribution.mint_model_entry("roberta-base", None)
        leftovers = list(registry.parent.glob("*.tmp"))
        assert leftovers == [], f"a temp file survived the write: {leftovers}"
