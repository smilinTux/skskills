"""Tests for SKSkills Registry — install, list, uninstall."""

from pathlib import Path
from textwrap import dedent

import pytest

from skskills.models import SkillStatus
from skskills.registry import SkillRegistry


@pytest.fixture
def skill_source(tmp_path: Path) -> Path:
    """Create a minimal skill directory for testing."""
    skill_dir = tmp_path / "test-skill-src"
    skill_dir.mkdir()
    (skill_dir / "knowledge").mkdir()
    (skill_dir / "knowledge" / "SKILL.md").write_text("# Test\n")

    yaml_content = dedent("""\
        name: test-skill
        version: "0.1.0"
        description: A test skill
        author:
          name: tester
        knowledge:
          - path: knowledge/SKILL.md
            description: Test context
            auto_load: true
    """)
    (skill_dir / "skill.yaml").write_text(yaml_content)
    return skill_dir


@pytest.fixture
def registry(tmp_path: Path) -> SkillRegistry:
    """Create a registry with a temp root."""
    return SkillRegistry(root=tmp_path / "skskills")


class TestInstall:
    """Test skill installation."""

    def test_install_global(self, registry: SkillRegistry, skill_source: Path):
        """Installing a skill globally should copy it to installed/."""
        installed = registry.install(skill_source)
        assert installed.manifest.name == "test-skill"
        assert installed.agent == "global"
        assert installed.status == SkillStatus.INSTALLED
        assert Path(installed.install_path).exists()

    def test_install_per_agent(self, registry: SkillRegistry, skill_source: Path):
        """Installing for a specific agent should use agents/<name>/."""
        installed = registry.install(skill_source, agent="jarvis")
        assert installed.agent == "jarvis"
        assert "agents/jarvis" in installed.install_path

    def test_install_duplicate_fails(self, registry: SkillRegistry, skill_source: Path):
        """Installing the same skill twice without force should fail."""
        registry.install(skill_source)
        with pytest.raises(ValueError, match="already installed"):
            registry.install(skill_source)

    def test_install_force_overwrites(self, registry: SkillRegistry, skill_source: Path):
        """Force install should overwrite existing."""
        registry.install(skill_source)
        installed = registry.install(skill_source, force=True)
        assert installed.manifest.name == "test-skill"


class TestUninstall:
    """Test skill removal."""

    def test_uninstall_existing(self, registry: SkillRegistry, skill_source: Path):
        """Uninstalling an installed skill should return True."""
        registry.install(skill_source)
        assert registry.uninstall("test-skill") is True

    def test_uninstall_nonexistent(self, registry: SkillRegistry):
        """Uninstalling a missing skill should return False."""
        assert registry.uninstall("nonexistent") is False


class TestList:
    """Test skill listing."""

    def test_list_empty(self, registry: SkillRegistry):
        """Empty registry should return empty list."""
        assert registry.list_skills() == []

    def test_list_after_install(self, registry: SkillRegistry, skill_source: Path):
        """Installed skills should appear in the list."""
        registry.install(skill_source)
        skills = registry.list_skills()
        assert len(skills) == 1
        assert skills[0].manifest.name == "test-skill"

    def test_list_filter_by_agent(self, registry: SkillRegistry, skill_source: Path):
        """Filtering by agent should only return that agent's skills."""
        registry.install(skill_source, agent="jarvis")
        assert len(registry.list_skills(agent="jarvis")) == 1
        assert len(registry.list_skills(agent="lumina")) == 0


class TestLink:
    """Test skill linking."""

    def test_link_global_to_agent(self, registry: SkillRegistry, skill_source: Path):
        """Linking a global skill to an agent should create a symlink."""
        registry.install(skill_source)
        link = registry.link_to_agent("test-skill", "jarvis")
        assert link.is_symlink()
        assert (link / "skill.yaml").exists()

    def test_link_nonexistent_fails(self, registry: SkillRegistry):
        """Linking a missing skill should raise FileNotFoundError."""
        registry.ensure_dirs()
        with pytest.raises(FileNotFoundError, match="not found"):
            registry.link_to_agent("nonexistent", "jarvis")


class TestAgentSkills:
    """Test agent skill resolution."""

    def test_agent_sees_global_skills(self, registry: SkillRegistry, skill_source: Path):
        """Agents should see global skills."""
        registry.install(skill_source)
        names = registry.agent_skills("jarvis")
        assert "test-skill" in names
