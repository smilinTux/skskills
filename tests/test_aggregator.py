"""Tests for the enhanced SkillAggregator.

Covers:
- auto-discovery of installed skills for an agent
- disabled skills are skipped during load
- get_loaded_skills() returns correct metadata
- _handle_skills_endpoint returns loaded skills
- _is_skill_enabled respects registry status
- health monitoring (SkillHealth records, skskills.health tool)
- tool namespace collision detection (skskills.collisions tool)
- auto-discovery from ~/.skcapstone/skills/
"""

from __future__ import annotations

import asyncio
import json
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


# ---------------------------------------------------------------------------
# Health monitoring
# ---------------------------------------------------------------------------

COLLISION_SKILL_YAML = """\
name: other-skill
version: 0.1.0
description: Another skill with a colliding tool name
author:
  name: Tester
tools:
  - name: greet
    description: Also greets (same base name as test-skill.greet)
    entrypoint: tools/greet.py:run
tags:
  - test
"""


class TestHealthMonitoring:
    def test_health_populated_after_load(self, registry_root, skill_source):
        """After load_all_skills, _health contains a record for each skill."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        assert "test-skill" in agg._health
        health = agg._health["test-skill"]
        assert health.status == "ok"
        assert health.tools_resolved == 1
        assert health.tools_failed == []
        assert health.load_error is None
        assert health.source == "registry"

    def test_health_degraded_when_tool_unresolved(self, tmp_path, registry_root):
        """A skill whose tool entrypoint is missing gets 'degraded' health."""
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "skill.yaml").write_text("""\
name: bad-skill
version: 0.1.0
description: Skill with missing entrypoint
author:
  name: Tester
tools:
  - name: missing
    description: Tool that does not exist
    entrypoint: tools/missing.py:run
""")
        (skill_dir / "tools").mkdir()
        # Note: tools/missing.py does NOT exist

        registry = SkillRegistry(registry_root)
        registry.install(skill_dir, agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        health = agg._health.get("bad-skill")
        assert health is not None
        assert health.status == "degraded"
        assert "missing" in health.tools_failed

    def test_health_error_for_skcapstone_skill_with_bad_yaml(self, tmp_path):
        """A skcapstone skill with broken skill.yaml gets 'error' health.

        The skcapstone scanner only checks file existence (unlike the registry
        which silently skips parse errors), so a broken skill.yaml that
        reaches SkillLoader.load() produces an 'error' health entry.
        """
        skcapstone_home = tmp_path / ".skcapstone"
        broken_dir = skcapstone_home / "skills" / "broken-skill"
        broken_dir.mkdir(parents=True)
        # Valid YAML structure but missing required 'name' field → Pydantic raises ValueError
        (broken_dir / "skill.yaml").write_text("""\
version: 0.1.0
description: Missing required name field
author:
  name: Tester
""")

        agg = SkillAggregator(
            agent="global",
            registry_root=tmp_path / ".skskills",
            skcapstone_home=skcapstone_home,
        )
        agg.load_all_skills()

        health = agg._health.get("broken-skill")
        assert health is not None
        assert health.status == "error"
        assert health.load_error is not None
        assert health.source == "skcapstone"

    def test_skskills_health_tool_all(self, registry_root, skill_source):
        """skskills.health with no filter returns summary + per-skill records."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        result = agg._handle_health({})
        data = json.loads(result[0].text)
        assert "summary" in data
        assert "skills" in data
        assert "test-skill" in data["skills"]
        assert data["summary"]["ok"] == 1

    def test_skskills_health_tool_single(self, registry_root, skill_source):
        """skskills.health with skill= returns just that skill's record."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        result = agg._handle_health({"skill": "test-skill"})
        data = json.loads(result[0].text)
        assert data["skill"] == "test-skill"
        assert data["status"] == "ok"

    def test_skskills_health_tool_unknown_skill(self, registry_root):
        """skskills.health for an unknown skill returns an error."""
        agg = SkillAggregator(agent="global", registry_root=registry_root)

        result = agg._handle_health({"skill": "nonexistent"})
        data = json.loads(result[0].text)
        assert "error" in data

    def test_health_cleared_on_reload(self, registry_root, skill_source):
        """_health is cleared and rebuilt on each call to load_all_skills."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()
        assert len(agg._health) == 1

        # Second load (idempotent)
        agg.load_all_skills()
        assert len(agg._health) == 1

    def test_get_loaded_skills_includes_health(self, registry_root, skill_source):
        """get_loaded_skills() returns health and source fields."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        skills = agg.get_loaded_skills()
        assert skills[0]["health"] == "ok"
        assert skills[0]["source"] == "registry"


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------


class TestCollisionDetection:
    def test_no_collisions_when_names_unique(self, registry_root, skill_source):
        """No collisions when all tool base names are unique."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        assert agg._collisions == {}

    def test_collision_detected_for_shared_base_name(self, tmp_path, registry_root):
        """Two skills with a tool named 'greet' trigger a collision record."""
        # Skill A: test-skill with tools/greet.py:run
        skill_a = tmp_path / "skill-a"
        skill_a.mkdir()
        (skill_a / "skill.yaml").write_text("""\
name: skill-a
version: 0.1.0
description: Skill A
author:
  name: Tester
tools:
  - name: greet
    description: Say hello from A
    entrypoint: tools/greet.py:run
""")
        (skill_a / "tools").mkdir()
        (skill_a / "tools" / "greet.py").write_text("def run(**kw): return 'A'")

        # Skill B: different skill, also with a 'greet' tool
        skill_b = tmp_path / "skill-b"
        skill_b.mkdir()
        (skill_b / "skill.yaml").write_text("""\
name: skill-b
version: 0.1.0
description: Skill B
author:
  name: Tester
tools:
  - name: greet
    description: Say hello from B
    entrypoint: tools/greet.py:run
""")
        (skill_b / "tools").mkdir()
        (skill_b / "tools" / "greet.py").write_text("def run(**kw): return 'B'")

        registry = SkillRegistry(registry_root)
        registry.install(skill_a, agent="global")
        registry.install(skill_b, agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        assert "greet" in agg._collisions
        colliding = agg._collisions["greet"]
        assert "skill-a.greet" in colliding
        assert "skill-b.greet" in colliding

    def test_skskills_collisions_tool_no_collisions(self, registry_root, skill_source):
        """skskills.collisions returns 0 when no collisions exist."""
        registry = SkillRegistry(registry_root)
        registry.install(skill_source / "test-skill", agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        result = agg._handle_collisions()
        data = json.loads(result[0].text)
        assert data["collisions"] == 0

    def test_skskills_collisions_tool_with_collisions(self, tmp_path, registry_root):
        """skskills.collisions returns collision details when present."""
        skill_a = tmp_path / "skill-a"
        skill_a.mkdir()
        (skill_a / "skill.yaml").write_text("""\
name: skill-a
version: 0.1.0
description: A
author:
  name: T
tools:
  - name: do-thing
    description: Do it
    entrypoint: tools/do.py:run
""")
        (skill_a / "tools").mkdir()
        (skill_a / "tools" / "do.py").write_text("def run(**kw): return 'done'")

        skill_b = tmp_path / "skill-b"
        skill_b.mkdir()
        (skill_b / "skill.yaml").write_text("""\
name: skill-b
version: 0.1.0
description: B
author:
  name: T
tools:
  - name: do-thing
    description: Also do it
    entrypoint: tools/do.py:run
""")
        (skill_b / "tools").mkdir()
        (skill_b / "tools" / "do.py").write_text("def run(**kw): return 'done2'")

        registry = SkillRegistry(registry_root)
        registry.install(skill_a, agent="global")
        registry.install(skill_b, agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()

        result = agg._handle_collisions()
        data = json.loads(result[0].text)
        assert data["collisions"] == 1
        assert "do-thing" in data["details"]

    def test_collisions_cleared_on_reload(self, tmp_path, registry_root):
        """_collisions is reset on each load_all_skills call."""
        skill_a = tmp_path / "skill-a"
        skill_a.mkdir()
        (skill_a / "skill.yaml").write_text("""\
name: skill-a
version: 0.1.0
description: A
author:
  name: T
tools:
  - name: clash
    description: Clash
    entrypoint: tools/clash.py:run
""")
        (skill_a / "tools").mkdir()
        (skill_a / "tools" / "clash.py").write_text("def run(**kw): return 'a'")

        skill_b = tmp_path / "skill-b"
        skill_b.mkdir()
        (skill_b / "skill.yaml").write_text("""\
name: skill-b
version: 0.1.0
description: B
author:
  name: T
tools:
  - name: clash
    description: Clash too
    entrypoint: tools/clash.py:run
""")
        (skill_b / "tools").mkdir()
        (skill_b / "tools" / "clash.py").write_text("def run(**kw): return 'b'")

        registry = SkillRegistry(registry_root)
        registry.install(skill_a, agent="global")
        registry.install(skill_b, agent="global")

        agg = SkillAggregator(agent="global", registry_root=registry_root)
        agg.load_all_skills()
        assert len(agg._collisions) == 1

        # Reload — should still be consistent
        agg.load_all_skills()
        assert len(agg._collisions) == 1


# ---------------------------------------------------------------------------
# skcapstone skills auto-discovery
# ---------------------------------------------------------------------------


class TestSkcapstoneDiscovery:
    def test_skills_discovered_from_skcapstone_dir(self, tmp_path):
        """Skills in ~/.skcapstone/skills/ are loaded when not in registry."""
        skcapstone_home = tmp_path / ".skcapstone"
        skills_dir = skcapstone_home / "skills" / "built-in-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "skill.yaml").write_text("""\
name: built-in-skill
version: 0.1.0
description: A skcapstone built-in
author:
  name: skcapstone
tools:
  - name: builtin-tool
    description: Built-in tool
    entrypoint: tools/run.py:run
""")
        (skills_dir / "tools").mkdir()
        (skills_dir / "tools" / "run.py").write_text("def run(**kw): return 'built-in'")

        registry_root = tmp_path / ".skskills"
        agg = SkillAggregator(
            agent="global",
            registry_root=registry_root,
            skcapstone_home=skcapstone_home,
        )
        count = agg.load_all_skills()

        assert count == 1
        assert agg.loader.get_server("built-in-skill") is not None
        health = agg._health.get("built-in-skill")
        assert health is not None
        assert health.source == "skcapstone"
        assert health.status == "ok"

    def test_registry_skill_overrides_skcapstone_skill(self, tmp_path):
        """Registry skill with same name wins over skcapstone skill."""
        skcapstone_home = tmp_path / ".skcapstone"
        sc_skill_dir = skcapstone_home / "skills" / "my-skill"
        sc_skill_dir.mkdir(parents=True)
        (sc_skill_dir / "skill.yaml").write_text("""\
name: my-skill
version: 0.1.0
description: skcapstone version
author:
  name: skcapstone
tools:
  - name: run
    description: run
    entrypoint: tools/run.py:run
""")
        (sc_skill_dir / "tools").mkdir()
        (sc_skill_dir / "tools" / "run.py").write_text("def run(**kw): return 'sc'")

        registry_root = tmp_path / ".skskills"
        reg_skill_dir = tmp_path / "registry-skill"
        reg_skill_dir.mkdir()
        (reg_skill_dir / "skill.yaml").write_text("""\
name: my-skill
version: 0.2.0
description: registry version
author:
  name: registry
tools:
  - name: run
    description: run
    entrypoint: tools/run.py:run
""")
        (reg_skill_dir / "tools").mkdir()
        (reg_skill_dir / "tools" / "run.py").write_text("def run(**kw): return 'reg'")

        from skskills.registry import SkillRegistry
        registry = SkillRegistry(registry_root)
        registry.install(reg_skill_dir, agent="global")

        agg = SkillAggregator(
            agent="global",
            registry_root=registry_root,
            skcapstone_home=skcapstone_home,
        )
        count = agg.load_all_skills()

        # Only one version loaded
        assert count == 1
        health = agg._health["my-skill"]
        # Registry version wins — source should be "registry"
        assert health.source == "registry"

    def test_scan_skcapstone_skills_empty_dir(self, tmp_path):
        """_scan_skcapstone_skills returns [] when the directory is absent."""
        agg = SkillAggregator(
            agent="global",
            registry_root=tmp_path / ".skskills",
            skcapstone_home=tmp_path / ".nonexistent",
        )
        assert agg._scan_skcapstone_skills() == []

    def test_scan_skcapstone_skills_ignores_non_skills(self, tmp_path):
        """_scan_skcapstone_skills ignores dirs without skill.yaml."""
        skcapstone_home = tmp_path / ".skcapstone"
        skills_dir = skcapstone_home / "skills"
        (skills_dir / "not-a-skill").mkdir(parents=True)
        # no skill.yaml

        agg = SkillAggregator(
            agent="global",
            registry_root=tmp_path / ".skskills",
            skcapstone_home=skcapstone_home,
        )
        assert agg._scan_skcapstone_skills() == []
