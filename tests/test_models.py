"""Tests for SKSkills models — skill.yaml schema validation."""

from pathlib import Path
from textwrap import dedent

import pytest

from skskills.models import (
    HookEvent,
    KnowledgePack,
    SkillAuthor,
    SkillManifest,
    SkillType,
    ToolDefinition,
    generate_skill_yaml,
    parse_skill_yaml,
)


class TestSkillManifest:
    """Test SkillManifest creation and validation."""

    def test_minimal_manifest(self):
        """A manifest with just a name should be valid."""
        m = SkillManifest(name="test-skill")
        assert m.name == "test-skill"
        assert m.version == "0.1.0"
        assert m.component_types == set()

    def test_full_manifest(self):
        """A fully-populated manifest should parse correctly."""
        m = SkillManifest(
            name="syncthing-setup",
            version="1.0.0",
            description="Auto-configure Syncthing for sovereign agents",
            author=SkillAuthor(name="smilinTux", fingerprint="CCBE9306410CF8CD"),
            knowledge=[
                KnowledgePack(path="knowledge/SKILL.md", auto_load=True),
            ],
            tools=[
                ToolDefinition(
                    name="setup",
                    description="Run full Syncthing setup",
                    entrypoint="tools.syncthing:full_setup",
                ),
            ],
            tags=["syncthing", "sovereign"],
        )
        assert m.component_types == {SkillType.KNOWLEDGE, SkillType.TOOL}
        assert m.tool_names == ["syncthing-setup.setup"]
        assert m.is_signed() is False

    def test_name_validation_rejects_spaces(self):
        """Skill names must be kebab-case."""
        with pytest.raises(ValueError, match="kebab-case"):
            SkillManifest(name="bad skill name")

    def test_name_validation_rejects_empty(self):
        """Empty skill names are rejected."""
        with pytest.raises(ValueError, match="kebab-case"):
            SkillManifest(name="")

    def test_signed_manifest(self):
        """A manifest with signature and signer should report signed."""
        m = SkillManifest(
            name="signed-skill",
            signature="-----BEGIN PGP SIGNATURE-----...",
            signed_by="CCBE9306410CF8CD",
        )
        assert m.is_signed() is True


class TestParseSkillYaml:
    """Test skill.yaml file parsing."""

    def test_parse_valid_yaml(self, tmp_path: Path):
        """A valid skill.yaml should parse into a SkillManifest."""
        yaml_content = dedent("""\
            name: test-skill
            version: "0.2.0"
            description: A test skill
            author:
              name: tester
            knowledge:
              - path: knowledge/SKILL.md
                description: Main context
                auto_load: true
            tools:
              - name: greet
                description: Say hello
                entrypoint: "tools.greet:run"
            tags:
              - test
        """)
        skill_yaml = tmp_path / "skill.yaml"
        skill_yaml.write_text(yaml_content)

        manifest = parse_skill_yaml(skill_yaml)
        assert manifest.name == "test-skill"
        assert manifest.version == "0.2.0"
        assert len(manifest.knowledge) == 1
        assert len(manifest.tools) == 1
        assert manifest.tools[0].name == "greet"

    def test_parse_missing_file(self, tmp_path: Path):
        """Missing skill.yaml should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_skill_yaml(tmp_path / "nonexistent.yaml")

    def test_parse_invalid_yaml(self, tmp_path: Path):
        """Non-mapping YAML should raise ValueError."""
        skill_yaml = tmp_path / "skill.yaml"
        skill_yaml.write_text("just a string")

        with pytest.raises(ValueError, match="YAML mapping"):
            parse_skill_yaml(skill_yaml)


class TestGenerateSkillYaml:
    """Test YAML generation from manifests."""

    def test_roundtrip(self, tmp_path: Path):
        """A manifest should survive a generate -> parse roundtrip."""
        original = SkillManifest(
            name="roundtrip-test",
            version="1.0.0",
            description="Testing YAML roundtrip",
            author=SkillAuthor(name="tester"),
            tools=[
                ToolDefinition(
                    name="ping",
                    description="Return pong",
                    entrypoint="tools.ping:run",
                ),
            ],
        )

        yaml_str = generate_skill_yaml(original)
        skill_yaml = tmp_path / "skill.yaml"
        skill_yaml.write_text(yaml_str)

        restored = parse_skill_yaml(skill_yaml)
        assert restored.name == original.name
        assert restored.version == original.version
        assert len(restored.tools) == 1
        assert restored.tools[0].name == "ping"
