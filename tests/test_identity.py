"""Where a result connects.

`resolve_aim_url` decides which Aim a non-training process writes to. It has
one job and it got it wrong for an entire transport mode, so the ordering is
pinned here rather than left to a docstring.
"""

from __future__ import annotations

import pytest

from astrolabe_callbacks import contract
from astrolabe_callbacks._identity import resolve_aim_url


class TestTheEngineDecidesTheTransport:
    """The engine picks a transport and says so through the environment.

    Reaching a hardcoded default ahead of what it said is what broke every
    eval and sample write under local-aim mode (AIMURL-1): the engine opens no
    tunnel there, and the local ``aim server`` dies with the training process,
    so a later step found nothing listening on ``aim://localhost:43800``.
    """

    def test_the_repo_path_wins_over_everything(self, monkeypatch):
        """The regression. Before the fix this returned the aim:// default and
        a separate process had nothing to connect to."""
        monkeypatch.setenv(contract.ENV_AIM_REPO_PATH, "/tmp/aim-local-s-1")
        monkeypatch.setenv("ASTROLABE_AIM_URL", "aim://elsewhere:9999")
        assert resolve_aim_url("aim://argument:1234") == "/tmp/aim-local-s-1"

    def test_tunnel_mode_is_untouched(self, monkeypatch):
        """The default transport. With no repo path exported, the aim:// URL
        is the reverse tunnel and remains correct."""
        monkeypatch.delenv(contract.ENV_AIM_REPO_PATH, raising=False)
        monkeypatch.delenv("ASTROLABE_AIM_URL", raising=False)
        assert resolve_aim_url(None) == contract.DEFAULT_AIM_URL

    def test_an_empty_repo_path_is_not_a_transport_choice(self, monkeypatch):
        """Exported-but-empty must not win. An env var set to "" is how a
        shell says nothing, not how the engine says local-aim."""
        monkeypatch.setenv(contract.ENV_AIM_REPO_PATH, "")
        monkeypatch.delenv("ASTROLABE_AIM_URL", raising=False)
        assert resolve_aim_url(None) == contract.DEFAULT_AIM_URL

    def test_the_env_url_still_beats_the_argument(self, monkeypatch):
        monkeypatch.delenv(contract.ENV_AIM_REPO_PATH, raising=False)
        monkeypatch.setenv("ASTROLABE_AIM_URL", "aim://env:1")
        assert resolve_aim_url("aim://arg:2") == "aim://env:1"

    def test_the_argument_beats_the_default(self, monkeypatch):
        monkeypatch.delenv(contract.ENV_AIM_REPO_PATH, raising=False)
        monkeypatch.delenv("ASTROLABE_AIM_URL", raising=False)
        assert resolve_aim_url("aim://arg:2") == "aim://arg:2"


def test_the_default_has_exactly_one_definition():
    """`_core` used to restate the value instead of re-exporting it.

    A contract literal with two homes is the drift the contract file exists to
    prevent, and the copy `_identity` imported was the non-contract one.
    """
    from astrolabe_callbacks import _core

    assert _core.DEFAULT_AIM_URL is contract.DEFAULT_AIM_URL
