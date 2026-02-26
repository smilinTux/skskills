"""SKSkills Aggregator — central MCP server exposing all skill tools.

The aggregator connects to all running skill servers and presents
their tools and resources through a single unified MCP endpoint.
This is the bridge between skcapstone MCP and individual skills.

Architecture:
    Agent MCP Client
         |
    SKCapstone MCP Server
         |
    SkillAggregator (this)
         |
    +----+----+----+
    |    |    |    |
   Skill Skill Skill ...
   (unix sockets)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from .loader import SkillLoader
from .models import SkillStatus
from .registry import SkillRegistry

logger = logging.getLogger("skskills.aggregator")


class SkillAggregator:
    """Aggregates all skill MCP servers into a single endpoint.

    Discovers installed skills, loads them via SkillLoader,
    and exposes their tools/resources through the MCP protocol.

    Args:
        agent: Agent name for per-agent skill resolution.
        registry_root: Path to ~/.skskills.
    """

    def __init__(
        self,
        agent: str = "global",
        registry_root: Optional[Path] = None,
    ) -> None:
        self.agent = agent
        self.registry = SkillRegistry(registry_root)
        self.loader = SkillLoader(registry_root)
        self._mcp_server = Server("skskills-aggregator")
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self._mcp_server.list_tools()
        async def list_tools() -> list[Tool]:
            builtin = [
                Tool(
                    name="skskills.list",
                    description="List all installed skills and their status",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent": {
                                "type": "string",
                                "description": "Filter by agent (default: all)",
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="skskills.skills",
                    description="List all loaded (active) skills with their tool namespaces",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                Tool(
                    name="skskills.info",
                    description="Get detailed information about a specific skill",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Skill name to inspect",
                            },
                        },
                        "required": ["name"],
                    },
                ),
                Tool(
                    name="skskills.run_tool",
                    description="Run a specific skill tool by its qualified name (skill_name.tool_name)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tool": {
                                "type": "string",
                                "description": "Fully-qualified tool name, e.g. 'syncthing-setup.check_status'",
                            },
                            "args": {
                                "type": "object",
                                "description": "Arguments to pass to the tool",
                            },
                        },
                        "required": ["tool"],
                    },
                ),
            ]

            # Only expose tools from enabled skills
            skill_tools = []
            for schema in self.loader.all_tools():
                skill_name = schema["name"].split(".")[0]
                if self._is_skill_enabled(skill_name):
                    skill_tools.append(Tool(
                        name=schema["name"],
                        description=schema["description"],
                        inputSchema=schema["inputSchema"],
                    ))

            return builtin + skill_tools

        @self._mcp_server.list_resources()
        async def list_resources() -> list[Resource]:
            resources = []
            for res in self.loader.all_resources():
                resources.append(Resource(
                    uri=res["uri"],
                    name=res["name"],
                    description=res.get("description", ""),
                    mimeType=res.get("mimeType", "text/plain"),
                ))
            return resources

        @self._mcp_server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            return await self._handle_tool_call(name, arguments)

        @self._mcp_server.read_resource()
        async def read_resource(uri: str) -> str:
            return await self._handle_read_resource(uri)

    def _is_skill_enabled(self, skill_name: str) -> bool:
        """Check whether a skill is enabled in the registry.

        Args:
            skill_name: Skill name to check.

        Returns:
            bool: True if enabled (INSTALLED or RUNNING), False if DISABLED.
        """
        skill = self.registry.get(skill_name, self.agent)
        if skill is None:
            # Also check global
            skill = self.registry.get(skill_name, "global")
        if skill is None:
            return True  # loaded but not in registry — allow by default
        return skill.status != SkillStatus.DISABLED

    async def _handle_tool_call(self, name: str, arguments: dict) -> list[TextContent]:
        """Route a tool call to the appropriate handler.

        Args:
            name: Fully-qualified tool name.
            arguments: Tool arguments.

        Returns:
            list[TextContent]: MCP response.
        """
        if name == "skskills.list":
            return self._handle_list(arguments)

        if name == "skskills.skills":
            return self._handle_skills_endpoint()

        if name == "skskills.info":
            return self._handle_info(arguments)

        if name == "skskills.run_tool":
            tool = arguments.get("tool", "")
            args = arguments.get("args", {})
            return await self._handle_tool_call(tool, args)

        try:
            result = await self.loader.call_tool(name, arguments)
            if isinstance(result, str):
                return [TextContent(type="text", text=result)]
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except KeyError as exc:
            return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]
        except Exception as exc:
            logger.exception("Tool '%s' failed", name)
            return [TextContent(type="text", text=json.dumps({"error": f"{name} failed: {exc}"}))]

    def _handle_skills_endpoint(self) -> list[TextContent]:
        """Return metadata for all currently loaded (active) skills.

        Returns:
            list[TextContent]: JSON list of loaded skill summaries.
        """
        data = []
        for server in self.loader.all_servers():
            m = server.manifest
            data.append({
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "enabled": self._is_skill_enabled(m.name),
                "tools": [f"{m.name}.{t.name}" for t in m.tools],
                "knowledge_packs": len(m.knowledge),
                "hooks": [h.event.value for h in m.hooks],
                "tags": m.tags,
                "signed": m.is_signed(),
            })
        return [TextContent(type="text", text=json.dumps(data, indent=2))]

    def _handle_list(self, arguments: dict) -> list[TextContent]:
        """List installed skills."""
        agent = arguments.get("agent")
        skills = self.registry.list_skills(agent)
        data = [
            {
                "name": s.manifest.name,
                "version": s.manifest.version,
                "agent": s.agent,
                "types": [t.value for t in s.manifest.component_types],
                "tools": s.manifest.tool_names,
                "status": s.status.value,
                "signed": s.manifest.is_signed(),
            }
            for s in skills
        ]
        return [TextContent(type="text", text=json.dumps(data, indent=2))]

    def _handle_info(self, arguments: dict) -> list[TextContent]:
        """Get info about a specific skill."""
        name = arguments.get("name", "")
        if not name:
            return [TextContent(type="text", text=json.dumps({"error": "name is required"}))]

        server = self.loader.get_server(name)
        if server is None:
            skill = self.registry.get(name, self.agent)
            if skill is None:
                return [TextContent(type="text", text=json.dumps({"error": f"Skill not found: {name}"}))]
            m = skill.manifest
        else:
            m = server.manifest

        data = {
            "name": m.name,
            "version": m.version,
            "description": m.description,
            "author": m.author.model_dump(),
            "types": [t.value for t in m.component_types],
            "tools": [
                {"name": t.name, "description": t.description}
                for t in m.tools
            ],
            "knowledge": [
                {"path": k.path, "description": k.description}
                for k in m.knowledge
            ],
            "hooks": [
                {"event": h.event.value, "description": h.description}
                for h in m.hooks
            ],
            "dependencies": [
                {"name": d.name, "version": d.version, "type": d.type}
                for d in m.dependencies
            ],
            "tags": m.tags,
            "signed": m.is_signed(),
            "signed_by": m.signed_by,
        }
        return [TextContent(type="text", text=json.dumps(data, indent=2))]

    async def _handle_read_resource(self, uri: str) -> str:
        """Route resource read to the correct skill server.

        Args:
            uri: Resource URI (e.g., skill://syncthing-setup/SKILL.md).

        Returns:
            str: Resource content.

        Raises:
            ValueError: If resource not found.
        """
        for server in self.loader.all_servers():
            content = await server.read_resource(uri)
            if content is not None:
                return content
        raise ValueError(f"Resource not found: {uri}")

    def load_all_skills(self) -> int:
        """Discover and load all enabled installed skills for the current agent.

        Agent-specific skills take priority over global skills when both exist
        with the same name. Disabled skills are skipped.

        Returns:
            int: Number of skills successfully loaded.
        """
        # Agent-specific skills first, then global (agent overrides global)
        skills = self.registry.list_skills(self.agent)
        if self.agent != "global":
            skills.extend(self.registry.list_skills("global"))

        loaded = 0
        seen: set[str] = set()
        for skill in skills:
            if skill.manifest.name in seen:
                continue
            seen.add(skill.manifest.name)

            if skill.status == SkillStatus.DISABLED:
                logger.info("Skipping disabled skill: %s", skill.manifest.name)
                continue

            try:
                self.loader.load(Path(skill.install_path))
                loaded += 1
            except Exception as exc:
                logger.error("Failed to load skill '%s': %s", skill.manifest.name, exc)

        logger.info("Loaded %d skills for agent '%s'", loaded, self.agent)
        return loaded

    def get_loaded_skills(self) -> list[dict]:
        """Return a summary of all currently loaded skills.

        Returns:
            list[dict]: Skill summaries with name, version, tools, enabled status.
        """
        result = []
        for server in self.loader.all_servers():
            m = server.manifest
            result.append({
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "enabled": self._is_skill_enabled(m.name),
                "tools": [f"{m.name}.{t.name}" for t in m.tools],
                "knowledge_packs": len(m.knowledge),
                "hooks": [h.event.value for h in m.hooks],
                "tags": m.tags,
            })
        return result

    async def run_stdio(self) -> None:
        """Run the aggregator as an MCP server on stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self._mcp_server.run(
                read_stream,
                write_stream,
                self._mcp_server.create_initialization_options(),
            )


def main(agent: str = "global") -> None:
    """Entry point for the SKSkills aggregator MCP server.

    Args:
        agent: Agent namespace to load skills for.
    """
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    agg = SkillAggregator(agent=agent)
    count = agg.load_all_skills()
    logger.warning("SKSkills aggregator started: %d skills loaded for '%s'", count, agent)
    asyncio.run(agg.run_stdio())
