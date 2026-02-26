"""Tests for per-agent skill registry isolation.

Verifies that installing a skill for agent 'jarvis' does NOT affect agent
'lumina', and that each agent has completely separate skill registries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skskills.models import SkillStatus
from skskills.registry import SkillRegistry


JARVIS_SKILL_YAML = """\
name: jarvis-skill
version: 1.0.0
description: Exclusive skill for Jarvis
author:
  name: Jarvis
tags:
  - jarvis
"""

LUMINA_SKILL_YAML = """\
name: lumina-skill
version: 1.0.0
description: Exclusive skill for Lumina
author:
  name: Lumina
tags:
  - lumina
"""

SHARED_SKILL_YAML = """\
name: shared-skill
version: 1.0.0
description: A globally installed skill
author:
  name: Both
tags:
  - global
"""


@pytest.fixture
def registry_root(tmp_path):
    return tmp_path / ".skskills"


def _make_skill_dir(base: Path, name: str, yaml_content: str) -> Path:
    d = base / name
    d.mkdir(parents=True)
    (d / "skill.yaml").write_text(yaml_content)
    (d / "knowledge").mkdir()
    (d / "knowledge" / "SKILL.md").write_text(f"# {name}")
    return d


class TestPerAgentIsolation:
    def test_jarvis_skill_invisible_to_lumina(self, registry_root, tmp_path):
        """A skill installed for jarvis does not appear in lumina's skill list."""
        src = tmp_path / "src"
        src.mkdir()
        _make_skill_dir(src, "jarvis-skill", JARVIS_SKILL_YAML)

        registry = SkillRegistry(registry_root)
        registry.install(src / "jarvis-skill", agent="jarvis")

        lumina_skills = registry.list_skills("lumina")
        jarvis_skill_names = [s.manifest.name for s in lumina_skills]
        assert "jarvis-skill" not in jarvis_skill_names

    def test_lumina_skill_invisible_to_jarvis(self, registry_root, tmp_path):
        """A skill installed for lumina does not appear in jarvis's skill list."""
        src = tmp_path / "src"
        src.mkdir()
        _make_skill_dir(src, "lumina-skill", LUMINA_SKILL_YAML)

        registry = SkillRegistry(registry_root)
        registry.install(src / "lumina-skill", agent="lumina")

        jarvis_skills = registry.list_skills("jarvis")
        lumina_skill_names = [s.manifest.name for s in jarvis_skills]
        assert "lumina-skill" not in lumina_skill_names

    def test_disabling_for_jarvis_does_not_affect_lumina(self, registry_root, tmp_path):
        """Disabling a skill for one agent does not affect the other agent's copy."""
        src = tmp_path / "src"
        src.mkdir()
        _make_skill_dir(src, "shared-skill", SHARED_SKILL_YAML)

        registry = SkillRegistry(registry_root)
        registry.install(src / "shared-skill", agent="jarvis")
        registry.install(src / "shared-skill", agent="lumina")

        # Disable only for jarvis
        registry.set_status("shared-skill", "jarvis", SkillStatus.DISABLED)

        jarvis_skill = registry.get("shared-skill", "jarvis")
        lumina_skill = registry.get("shared-skill", "lumina")

        assert jarvis_skill is not None
        assert lumina_skill is not None
        assert jarvis_skill.status == SkillStatus.DISABLED
        assert lumina_skill.status == SkillStatus.INSTALLED

    def test_uninstalling_for_jarvis_does_not_remove_luminas_copy(self, registry_root, tmp_path):
        """Uninstalling from jarvis leaves lumina's copy intact."""
        src = tmp_path / "src"
        src.mkdir()
        _make_skill_dir(src, "shared-skill", SHARED_SKILL_YAML)

        registry = SkillRegistry(registry_root)
        registry.install(src / "shared-skill", agent="jarvis")
        registry.install(src / "shared-skill", agent="lumina")

        removed = registry.uninstall("shared-skill", "jarvis")
        assert removed is True

        # Lumina's copy must survive
        lumina_skill = registry.get("shared-skill", "lumina")
        assert lumina_skill is not None

        # Jarvis's copy must be gone
        jarvis_skill = registry.get("shared-skill", "jarvis")
        assert jarvis_skill is None

    def test_agents_have_separate_install_directories(self, registry_root, tmp_path):
        """Jarvis and Lumina each get their own directory under agents/."""
        src = tmp_path / "src"
        src.mkdir()
        _make_skill_dir(src, "jarvis-skill", JARVIS_SKILL_YAML)
        _make_skill_dir(src, "lumina-skill", LUMINA_SKILL_YAML)

        registry = SkillRegistry(registry_root)
        registry.install(src / "jarvis-skill", agent="jarvis")
        registry.install(src / "lumina-skill", agent="lumina")

        jarvis_dir = registry_root / "agents" / "jarvis"
        lumina_dir = registry_root / "agents" / "lumina"

        assert (jarvis_dir / "jarvis-skill" / "skill.yaml").exists()
        assert (lumina_dir / "lumina-skill" / "skill.yaml").exists()

        # Cross-contamination check
        assert not (jarvis_dir / "lumina-skill").exists()
        assert not (lumina_dir / "jarvis-skill").exists()

    def test_global_skill_visible_to_all_agents(self, registry_root, tmp_path):
        """A globally installed skill is visible to all agents via list_skills(None)."""
        src = tmp_path / "src"
        src.mkdir()
        _make_skill_dir(src, "shared-skill", SHARED_SKILL_YAML)

        registry = SkillRegistry(registry_root)
        registry.install(src / "shared-skill", agent="global")

        all_skills = registry.list_skills(None)
        names = [s.manifest.name for s in all_skills]
        assert "shared-skill" in names

    def test_search_is_agent_scoped(self, registry_root, tmp_path):
        """Search respects agent scoping."""
        src = tmp_path / "src"
        src.mkdir()
        _make_skill_dir(src, "jarvis-skill", JARVIS_SKILL_YAML)
        _make_skill_dir(src, "lumina-skill", LUMINA_SKILL_YAML)

        registry = SkillRegistry(registry_root)
        registry.install(src / "jarvis-skill", agent="jarvis")
        registry.install(src / "lumina-skill", agent="lumina")

        jarvis_results = registry.search("jarvis", agent="jarvis")
        lumina_results = registry.search("lumina", agent="lumina")

        assert any(s.manifest.name == "jarvis-skill" for s in jarvis_results)
        assert not any(s.manifest.name == "jarvis-skill" for s in lumina_results)
        assert any(s.manifest.name == "lumina-skill" for s in lumina_results)
        assert not any(s.manifest.name == "lumina-skill" for s in jarvis_results)

    def test_aggregator_isolation_per_agent(self, registry_root, tmp_path):
        """SkillAggregator for jarvis only loads jarvis's skills."""
        from skskills.aggregator import SkillAggregator

        src = tmp_path / "src"
        src.mkdir()
        _make_skill_dir(src, "jarvis-skill", JARVIS_SKILL_YAML)
        _make_skill_dir(src, "lumina-skill", LUMINA_SKILL_YAML)

        registry = SkillRegistry(registry_root)
        registry.install(src / "jarvis-skill", agent="jarvis")
        registry.install(src / "lumina-skill", agent="lumina")

        jarvis_agg = SkillAggregator(agent="jarvis", registry_root=registry_root)
        count = jarvis_agg.load_all_skills()

        assert count == 1
        assert jarvis_agg.loader.get_server("jarvis-skill") is not None
        assert jarvis_agg.loader.get_server("lumina-skill") is None
