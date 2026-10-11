"""Tests for skskills.agent_links: projecting installed skills into agent frameworks."""

from pathlib import Path

import pytest

from skskills import agent_links
from skskills.registry import SkillRegistry


def make_skill(root: Path, name: str, skill_md: bool = True) -> Path:
    src = root / "src" / name
    (src / "references").mkdir(parents=True)
    (src / "skill.yaml").write_text(f"name: {name}\nversion: 1.0.0\n", encoding="utf-8")
    (src / "references" / "ref.md").write_text("# ref\n", encoding="utf-8")
    if skill_md:
        (src / "SKILL.md").write_text(f"---\nname: {name}\n---\n# {name}\n", encoding="utf-8")
    return src


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake HOME with Claude Code, Codex, pi, and Hermes present."""
    h = tmp_path / "home"
    for d in (".claude", ".codex", ".pi/agent", ".hermes/skills"):
        (h / d).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.setattr(agent_links.shutil, "which", lambda _: None)
    monkeypatch.setenv("SKSKILLS_AGENT_LINKS", "1")
    return h


def registry_for(home: Path) -> SkillRegistry:
    return SkillRegistry(home / ".skskills", agent_links=True)


def test_detects_only_present_frameworks(home):
    targets = agent_links.detect_targets(home)
    assert targets == {
        "claude-code": home / ".claude" / "skills",
        "codex": home / ".codex" / "skills",
        "pi": home / ".pi" / "agent" / "skills",
        "hermes": home / ".hermes" / "skills" / "skskills",
    }


def test_opencode_alone_gets_its_own_dir(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / ".config" / "opencode").mkdir(parents=True)
    monkeypatch.setattr(agent_links.shutil, "which", lambda _: None)
    assert agent_links.detect_targets(h) == {"opencode": h / ".config" / "opencode" / "skills"}


def test_opencode_skipped_when_claude_present(home):
    """OpenCode reads ~/.claude/skills itself; a second copy would duplicate names."""
    (home / ".config" / "opencode").mkdir(parents=True)
    assert "opencode" not in agent_links.detect_targets(home)


def test_codex_home_and_pi_dir_env_respected(home, monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "cx"))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    (tmp_path / "cx").mkdir()
    (tmp_path / "pi").mkdir()
    targets = agent_links.detect_targets(home)
    assert targets["codex"] == tmp_path / "cx" / "skills"
    assert targets["pi"] == tmp_path / "pi" / "skills"


def test_install_links_whole_skill_dir_into_every_framework(home, tmp_path):
    reg = registry_for(home)
    reg.install(make_skill(tmp_path, "demo"))
    installed = home / ".skskills" / "installed" / "demo"
    for target in agent_links.detect_targets(home).values():
        link = target / "demo"
        assert link.is_symlink()
        assert link.resolve() == installed.resolve()
        assert (link / "references" / "ref.md").is_file()


def test_hermes_category_gets_description(home, tmp_path):
    registry_for(home).install(make_skill(tmp_path, "demo"))
    assert (home / ".hermes" / "skills" / "skskills" / "DESCRIPTION.md").is_file()


def test_skill_without_skill_md_is_not_linked(home, tmp_path):
    registry_for(home).install(make_skill(tmp_path, "toolsonly", skill_md=False))
    assert not (home / ".claude" / "skills" / "toolsonly").exists()


def test_unmanaged_dir_is_never_overwritten(home, tmp_path):
    manual = home / ".claude" / "skills" / "demo"
    manual.mkdir(parents=True)
    (manual / "SKILL.md").write_text("hand-made\n", encoding="utf-8")
    reg = registry_for(home)
    reg.install(make_skill(tmp_path, "demo"))
    assert not manual.is_symlink()
    assert (manual / "SKILL.md").read_text(encoding="utf-8") == "hand-made\n"
    assert agent_links.status(reg)["demo"]["claude-code"] == "conflict"


def test_foreign_symlink_is_never_replaced(home, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    link = home / ".claude" / "skills" / "demo"
    link.parent.mkdir(parents=True)
    link.symlink_to(elsewhere)
    registry_for(home).install(make_skill(tmp_path, "demo"))
    assert link.resolve() == elsewhere.resolve()


def test_uninstall_removes_only_managed_links(home, tmp_path):
    reg = registry_for(home)
    reg.install(make_skill(tmp_path, "demo"))
    manual = home / ".codex" / "skills" / "demo"
    manual.unlink()
    manual.mkdir()
    reg.uninstall("demo")
    assert not (home / ".claude" / "skills" / "demo").exists()
    assert not (home / ".claude" / "skills" / "demo").is_symlink()
    assert manual.is_dir()


def test_sync_backfills_and_prunes(home, tmp_path):
    reg = SkillRegistry(home / ".skskills", agent_links=False)
    reg.install(make_skill(tmp_path, "old"))
    assert not (home / ".claude" / "skills" / "old").exists()

    stale = home / ".claude" / "skills" / "gone"
    stale.parent.mkdir(parents=True)
    stale.symlink_to(home / ".skskills" / "installed" / "gone")

    result = agent_links.sync(reg)
    assert (home / ".claude" / "skills" / "old").is_symlink()
    assert not stale.is_symlink()
    assert result["linked"] >= 1 and result["pruned"] == 1


def test_sync_is_idempotent(home, tmp_path):
    reg = registry_for(home)
    reg.install(make_skill(tmp_path, "demo"))
    assert agent_links.sync(reg)["linked"] == 0


def test_explicit_root_does_not_link_by_default(home, tmp_path):
    SkillRegistry(home / ".skskills").install(make_skill(tmp_path, "demo"))
    assert not (home / ".claude" / "skills" / "demo").exists()


def test_env_opt_out(home, tmp_path, monkeypatch):
    monkeypatch.setenv("SKSKILLS_AGENT_LINKS", "0")
    monkeypatch.setenv("SKSKILLS_HOME", str(home / ".skskills"))
    SkillRegistry().install(make_skill(tmp_path, "demo"))
    assert not (home / ".claude" / "skills" / "demo").exists()


def test_default_registry_links(home, tmp_path, monkeypatch):
    monkeypatch.setenv("SKSKILLS_HOME", str(home / ".skskills"))
    SkillRegistry().install(make_skill(tmp_path, "demo"))
    assert (home / ".claude" / "skills" / "demo").is_symlink()


def test_cli_sync_and_status(home, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from skskills.cli import main

    monkeypatch.setenv("SKSKILLS_HOME", str(home / ".skskills"))
    SkillRegistry(home / ".skskills").install(make_skill(tmp_path, "demo"))
    runner = CliRunner()
    result = runner.invoke(main, ["agents", "sync"])
    assert result.exit_code == 0, result.output
    assert "Linked: 4" in result.output
    status = runner.invoke(main, ["agents", "status"])
    assert status.exit_code == 0 and "demo" in status.output and "missing" not in status.output
