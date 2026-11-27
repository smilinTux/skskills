"""Catalog entries backed by a Git repo install by adapting SKILL.md-only trees."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import skskills.installer as installer
from skskills.catalog import CatalogEntry


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
        cwd=cwd, check=True, capture_output=True,
    )


@pytest.fixture
def skill_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "upstream"
    skill = repo / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: A SKILL.md-only skill\n---\n# Demo\n"
    )
    _git("init", "-q", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "init", cwd=repo)
    return repo


def test_skill_md_only_git_skill_is_adapted_to_a_manifest(skill_repo, monkeypatch):
    seen: dict = {}

    def fake_install(source, agent="global", force=True):
        seen["manifest"] = (Path(source) / "skill.yaml").read_text()
        seen["has_skill_md"] = (Path(source) / "SKILL.md").exists()
        seen["agent"] = agent
        return "installed"

    monkeypatch.setattr(installer, "install_from_local", fake_install)
    entry = CatalogEntry(name="demo", git=str(skill_repo), git_path="skills/demo")
    assert installer._install_git_entry(entry, "demo", "global", True) == "installed"
    assert "name: demo-skill" in seen["manifest"]
    assert "description: A SKILL.md-only skill" in seen["manifest"]
    assert "path: SKILL.md" in seen["manifest"]
    assert seen["has_skill_md"] is True


def test_missing_git_path_is_a_clear_error(skill_repo):
    entry = CatalogEntry(name="demo", git=str(skill_repo), git_path="skills/nope")
    with pytest.raises(FileNotFoundError, match="Git skill path not found"):
        installer._install_git_entry(entry, "demo", "global", True)


def test_git_entry_without_source_is_refused():
    with pytest.raises(ValueError, match="no Git source"):
        installer._install_git_entry(CatalogEntry(name="x"), "x", "global", True)
