"""Tests for the enhanced SkillAggregator.

Covers:
- auto-discovery of installed skills for an agent
- disabled skills are skipped during load
- get_loaded_skills() returns correct metadata
- _handle_skills_endpoint returns loaded skills
- _is_skill_enabled respects registry status
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from skskills.aggregator import SkillAggregator
from skskills.models import SkillStatus
from skskills.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_SKILL_YAML = """\
name: test-skill
version: 0.1.0
description: A test skill
author:
  name: Tester
tools:
  - name: greet
    description: Say hello
    entrypoint: tools/greet.py:run
    input_schema:
      type: object
      properties:
        name:
          type: string
      required: []
knowledge:
  - path: knowledge/SKILL.md
    description: Test knowledge
tags:
  - test
"""

SECOND_SKILL_YAML = """\
name: second-skill
version: 0.2.0
description: Another test skill
author:
  name: Tester
tools:
  - name: ping
    description: Ping tool
    entrypoint: tools/ping.py:run
tags:
  - test
"""


def _make_skill(root: Path, name: str, yaml_content: str, with_tool: bool = True) -> Path:
    """Create a minimal installable skill directory."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.yaml").write_text(yaml_content)
    (skill_dir / "knowledge").mkdir()
    (skill_dir / "knowledge" / "SKILL.md").write_text(f"# {name}\nTest.")
    if with_tool:
        (skill_dir / "tools").mkdir()
        tool_py = skill_dir / "tools" / "greet.py"
        tool_py.write_text("def run(**kwargs): return 'hello'")
        (skill_dir / "tools" / "ping.py").write_text("def run(**kwargs): return 'pong'")
    return skill_dir


@pytest.fixture
def registry_root(tmp_path):
    """Provide a fresh registry root for each test."""
    return tmp_path / ".skskills"


@pytest.fixture
def skill_source(tmp_path):
    """Create a source directory with two skills."""
    src = tmp_path / "skills_src"
    src.mkdir()
    _make_skill(src, "test-skill", MINIMAL_SKILL_YAML)
    _make_skill(src, "second-skill", SECOND_SKILL_YAML)
    return src


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadAllSkills:
    def test_auto_discovers_installed_skills(self, registry_root, skill_source):
        """load_all_skills discovers all skills installed for the agent."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")
        registry.install(skill_source / "second-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        count = agg.load_all_skills()

        assert count == 2
        assert agg.loader.get_server("test-skill") is not None
        assert agg.loader.get_server("second-skill") is not None

    def test_disabled_skills_are_skipped(self, registry_root, skill_source):
        """Disabled skills are not loaded by load_all_skills."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")
        registry.install(skill_source / "second-skill", agent="global")
        registry.set_status("second-skill", "global", SkillStatus.DISABLED)

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        count = agg.load_all_skills()

        assert count == 1
        assert agg.loader.get_server("test-skill") is not None
        assert agg.loader.get_server("second-skill") is None

    def test_agent_skills_override_global(self, registry_root, skill_source):
        """Agent-specific skills take priority over global skills with same name."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")
        registry.install(skill_source / "test-skill", agent="jarvis", force=True)

        agg = SkillAggregator(agent="jarvis", registry_root=registry_root)
        count = agg.load_all_skills()

        # Only one should be loaded (jarvis version wins)
        assert count == 1

    def test_empty_registry_loads_zero(self, registry_root):
        """An empty registry loads zero skills without error."""
        agg = SkillAggregator(agent="global", registry_root=registry_root)
        count = agg.load_all_skills()
        assert count == 0


class TestGetLoadedSkills:
    def test_returns_skill_metadata(self, registry_root, skill_source):
        """get_loaded_skills returns correct metadata for each loaded skill."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        skills = agg.get_loaded_skills()
        assert len(skills) == 1
        skill = skills[0]
        assert skill["name"] == "test-skill"
        assert skill["version"] == "0.1.0"
        assert "test-skill.greet" in skill["tools"]
        assert "test" in skill["tags"]

    def test_enabled_field_reflects_status(self, registry_root, skill_source):
        """enabled field is False for disabled skills that were loaded before disabling."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        # Disable after loading (simulates runtime toggle)
        registry.set_status("test-skill", "global", SkillStatus.DISABLED)

        skills = agg.get_loaded_skills()
        assert skills[0]["enabled"] is False

    def test_returns_empty_list_when_nothing_loaded(self, registry_root):
        """get_loaded_skills returns [] when no skills are loaded."""
        agg = SkillAggregator(agent="global", registry_root=registry_root)
        assert agg.get_loaded_skills() == []


class TestSkillsEndpoint:
    def test_skskills_skills_tool_lists_loaded(self, registry_root, skill_source):
        """skskills.skills tool returns JSON list of loaded skills."""
        import json

        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        result = agg._handle_skills_endpoint()
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert isinstance(data, list)
        assert data[0]["name"] == "test-skill"

    def test_skskills_run_tool_dispatcher(self, registry_root, skill_source):
        """skskills.run_tool routes to the correct skill tool."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        result = asyncio.run(agg._handle_tool_call("test-skill.greet", {}))
        assert len(result) == 1
        # The greet tool returns "hello"
        assert "hello" in result[0].text


class TestIsSkillEnabled:
    def test_installed_skill_is_enabled(self, registry_root, skill_source):
        """INSTALLED status means enabled."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        assert agg._is_skill_enabled("test-skill") is True

    def test_disabled_skill_is_not_enabled(self, registry_root, skill_source):
        """DISABLED status means not enabled."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")
        registry.set_status("test-skill", "global", SkillStatus.DISABLED)

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        assert agg._is_skill_enabled("test-skill") is False

    def test_unknown_skill_defaults_to_enabled(self, registry_root):
        """A skill not in the registry defaults to enabled (allow unknown)."""
        agg = SkillAggregator(agent="global", registry_root=registry_root)
        assert agg._is_skill_enabled("nonexistent-skill") is True
