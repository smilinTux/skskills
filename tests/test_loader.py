"""Tests for SKSkills Loader — skill server creation and tool routing."""

from pathlib import Path
from textwrap import dedent

import pytest

from skskills.loader import SkillLoader, SkillServer, resolve_entrypoint, socket_path_for
from skskills.models import parse_skill_yaml


@pytest.fixture
def skill_with_tool(tmp_path: Path) -> Path:
    """Create a skill with a Python tool for testing."""
    skill_dir = tmp_path / "greet-skill"
    skill_dir.mkdir()
    (skill_dir / "knowledge").mkdir()
    (skill_dir / "knowledge" / "SKILL.md").write_text("# Greet Skill\n")
    (skill_dir / "tools").mkdir()

    tool_py = skill_dir / "tools" / "greet.py"
    tool_py.write_text(dedent("""\
        def run(name: str = "world") -> str:
            return f"Hello, {name}!"

        async def async_run(name: str = "world") -> str:
            return f"Async hello, {name}!"
    """))

    yaml_content = dedent("""\
        name: greet-skill
        version: "0.1.0"
        description: Greeting skill
        author:
          name: tester
        knowledge:
          - path: knowledge/SKILL.md
            description: Greet context
        tools:
          - name: greet
            description: Say hello
            entrypoint: "tools/greet.py:run"
          - name: async-greet
            description: Say hello async
            entrypoint: "tools/greet.py:async_run"
    """)
    (skill_dir / "skill.yaml").write_text(yaml_content)
    return skill_dir


class TestSocketPath:
    """Test socket path generation."""

    def test_socket_path(self):
        """Socket path should be /tmp/skskill-{name}.sock."""
        assert socket_path_for("my-skill") == "/tmp/skskill-my-skill.sock"


class TestResolveEntrypoint:
    """Test entrypoint resolution."""

    def test_resolve_python_file(self, skill_with_tool: Path):
        """Should resolve a Python file:function entrypoint."""
        fn = resolve_entrypoint("tools/greet.py:run", skill_with_tool)
        assert fn is not None
        assert fn(name="Jarvis") == "Hello, Jarvis!"

    def test_resolve_nonexistent(self, tmp_path: Path):
        """Missing entrypoint should return None."""
        fn = resolve_entrypoint("missing/module.py:func", tmp_path)
        assert fn is None


class TestSkillServer:
    """Test SkillServer tool execution."""

    def test_load_and_resolve(self, skill_with_tool: Path):
        """Loading a skill should resolve its tools."""
        manifest = parse_skill_yaml(skill_with_tool / "skill.yaml")
        server = SkillServer(manifest, skill_with_tool)
        server.resolve_all()

        schemas = server.get_tool_schemas()
        assert len(schemas) == 2
        assert schemas[0]["name"] == "greet-skill.greet"

    @pytest.mark.asyncio
    async def test_call_sync_tool(self, skill_with_tool: Path):
        """Calling a sync tool should work via async interface."""
        manifest = parse_skill_yaml(skill_with_tool / "skill.yaml")
        server = SkillServer(manifest, skill_with_tool)
        server.resolve_all()

        result = await server.call_tool("greet", {"name": "Lumina"})
        assert result == "Hello, Lumina!"

    @pytest.mark.asyncio
    async def test_call_async_tool(self, skill_with_tool: Path):
        """Calling an async tool should work."""
        manifest = parse_skill_yaml(skill_with_tool / "skill.yaml")
        server = SkillServer(manifest, skill_with_tool)
        server.resolve_all()

        result = await server.call_tool("async-greet", {"name": "Jarvis"})
        assert result == "Async hello, Jarvis!"

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, skill_with_tool: Path):
        """Calling a nonexistent tool should raise KeyError."""
        manifest = parse_skill_yaml(skill_with_tool / "skill.yaml")
        server = SkillServer(manifest, skill_with_tool)
        server.resolve_all()

        with pytest.raises(KeyError, match="Tool not found"):
            await server.call_tool("nonexistent", {})

    def test_resource_list(self, skill_with_tool: Path):
        """Knowledge packs should appear as resources."""
        manifest = parse_skill_yaml(skill_with_tool / "skill.yaml")
        server = SkillServer(manifest, skill_with_tool)
        resources = server.get_resource_list()
        assert len(resources) == 1
        assert resources[0]["uri"] == "skill://greet-skill/knowledge/SKILL.md"

    @pytest.mark.asyncio
    async def test_read_resource(self, skill_with_tool: Path):
        """Should read knowledge pack content by URI."""
        manifest = parse_skill_yaml(skill_with_tool / "skill.yaml")
        server = SkillServer(manifest, skill_with_tool)
        content = await server.read_resource("skill://greet-skill/knowledge/SKILL.md")
        assert content == "# Greet Skill\n"


class TestSkillLoader:
    """Test the SkillLoader orchestrator."""

    def test_load_skill(self, skill_with_tool: Path):
        """Loading a skill should register it."""
        loader = SkillLoader()
        server = loader.load(skill_with_tool)
        assert server.manifest.name == "greet-skill"
        assert loader.get_server("greet-skill") is server

    def test_all_tools(self, skill_with_tool: Path):
        """All tools from loaded skills should be aggregated."""
        loader = SkillLoader()
        loader.load(skill_with_tool)
        tools = loader.all_tools()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert "greet-skill.greet" in names

    @pytest.mark.asyncio
    async def test_qualified_tool_call(self, skill_with_tool: Path):
        """Should route qualified tool calls correctly."""
        loader = SkillLoader()
        loader.load(skill_with_tool)
        result = await loader.call_tool("greet-skill.greet", {"name": "Boss"})
        assert result == "Hello, Boss!"

    @pytest.mark.asyncio
    async def test_unqualified_tool_call_fails(self, skill_with_tool: Path):
        """Unqualified tool names should raise KeyError."""
        loader = SkillLoader()
        loader.load(skill_with_tool)
        with pytest.raises(KeyError, match="qualified"):
            await loader.call_tool("greet", {"name": "test"})

    def test_unload_skill(self, skill_with_tool: Path):
        """Unloading a skill should remove it."""
        loader = SkillLoader()
        loader.load(skill_with_tool)
        assert loader.unload("greet-skill") is True
        assert loader.get_server("greet-skill") is None
        assert loader.unload("greet-skill") is False
