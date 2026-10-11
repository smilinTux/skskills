"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _no_agent_links(monkeypatch):
    """Never project test installs into the real ~/.claude, ~/.codex, etc.

    Tests that exercise agent linking opt back in with SKSKILLS_AGENT_LINKS=1
    and a fake HOME.
    """
    monkeypatch.setenv("SKSKILLS_AGENT_LINKS", "0")
