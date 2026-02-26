"""Tests for enhanced CLI commands: search, enable, disable, update."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from skskills.cli import main
from skskills.models import SkillStatus
from skskills.registry import SkillRegistry


SKILL_YAML = """\
name: test-skill
version: 0.1.0
description: A test skill for searching
author:
  name: Tester
tags:
  - sync
  - test
"""

SKILL_YAML_V2 = """\
name: test-skill
version: 0.2.0
description: Updated test skill
author:
  name: Tester
tags:
  - sync
  - test
"""


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def skill_dir(tmp_path):
    """Create a minimal installable skill directory."""
    d = tmp_path / "test-skill"
    d.mkdir()
    (d / "skill.yaml").write_text(SKILL_YAML)
    (d / "knowledge").mkdir()
    (d / "knowledge" / "SKILL.md").write_text("# Test")
    return d


@pytest.fixture
def skill_dir_v2(tmp_path):
    d = tmp_path / "test-skill-v2"
    d.mkdir()
    (d / "skill.yaml").write_text(SKILL_YAML_V2)
    (d / "knowledge").mkdir()
    (d / "knowledge" / "SKILL.md").write_text("# Updated")
    return d


@pytest.fixture
def registry_with_skill(tmp_path, skill_dir):
    """Install a skill into a temp registry and return (registry, home)."""
    home = tmp_path / ".skskills"
    registry = SkillRegistry(home)
    registry.install(skill_dir, agent="global")
    return registry, home


class TestSearchCommand:
    def test_search_by_name(self, runner, tmp_path, skill_dir):
        home = tmp_path / ".skskills"
        SkillRegistry(home).install(skill_dir, agent="global")

        result = runner.invoke(main, ["search", "test", "--agent", "global"],
                               env={"SKSKILLS_HOME": str(home)})
        # Should find our skill by name
        assert result.exit_code == 0
        assert "test-skill" in result.output

    def test_search_by_tag(self, runner, tmp_path, skill_dir):
        home = tmp_path / ".skskills"
        SkillRegistry(home).install(skill_dir, agent="global")

        result = runner.invoke(main, ["search", "sync"],
                               env={"SKSKILLS_HOME": str(home)})
        assert result.exit_code == 0
        assert "test-skill" in result.output

    def test_search_no_results(self, runner, tmp_path, skill_dir):
        home = tmp_path / ".skskills"
        SkillRegistry(home).install(skill_dir, agent="global")

        result = runner.invoke(main, ["search", "nonexistent-xyz"])
        assert result.exit_code == 0
        assert "No skills found" in result.output


class TestEnableDisableCommands:
    def test_disable_skill(self, runner, tmp_path, registry_with_skill):
        registry, home = registry_with_skill

        result = runner.invoke(main, ["disable", "test-skill", "--agent", "global"],
                               env={"SKSKILLS_HOME": str(home)})
        assert result.exit_code == 0
        assert "Disabled" in result.output

        # Verify in registry
        skill = registry.get("test-skill", "global")
        assert skill is not None
        assert skill.status == SkillStatus.DISABLED

    def test_enable_skill(self, runner, tmp_path, registry_with_skill):
        registry, home = registry_with_skill
        registry.set_status("test-skill", "global", SkillStatus.DISABLED)

        result = runner.invoke(main, ["enable", "test-skill", "--agent", "global"],
                               env={"SKSKILLS_HOME": str(home)})
        assert result.exit_code == 0
        assert "Enabled" in result.output

        skill = registry.get("test-skill", "global")
        assert skill is not None
        assert skill.status == SkillStatus.INSTALLED

    def test_disable_nonexistent_skill_exits_1(self, runner, tmp_path):
        home = tmp_path / ".skskills"

        result = runner.invoke(main, ["disable", "no-such-skill"],
                               env={"SKSKILLS_HOME": str(home)})
        assert result.exit_code == 1
        assert "Not found" in result.output

    def test_list_shows_disabled_status(self, runner, tmp_path, registry_with_skill):
        registry, home = registry_with_skill
        registry.set_status("test-skill", "global", SkillStatus.DISABLED)

        result = runner.invoke(main, ["list"],
                               env={"SKSKILLS_HOME": str(home)})
        assert result.exit_code == 0
        assert "disabled" in result.output


class TestUpdateCommand:
    def test_update_reinstalls_from_source(self, runner, tmp_path, registry_with_skill, skill_dir_v2):
        registry, home = registry_with_skill

        result = runner.invoke(
            main,
            ["update", "test-skill", str(skill_dir_v2), "--agent", "global"],
            env={"SKSKILLS_HOME": str(home)},
        )
        assert result.exit_code == 0
        assert "Updated" in result.output
        assert "0.2.0" in result.output

    def test_update_nonexistent_source_fails(self, runner, tmp_path, registry_with_skill):
        registry, home = registry_with_skill

        result = runner.invoke(
            main,
            ["update", "test-skill", "/nonexistent/path", "--agent", "global"],
        )
        # Click will catch the invalid path before we get to the command
        assert result.exit_code != 0
