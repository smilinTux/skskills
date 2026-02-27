# SKSkills Skill
## SKILL.md - Sovereign Agent Skill Framework

**Name:** skskills
**Version:** 0.1.0
**Author:** smilinTux Team
**Category:** Agent Infrastructure & Skill Management
**License:** GPL-3.0-or-later

---

## Description

Sovereign agent skill framework built natively on MCP (Model Context Protocol). SKSkills replaces OpenClaw as the standard mechanism for packaging, installing, and running agent capabilities. Skills are self-contained MCP servers that expose Knowledge, Capabilities, and Flows to any compatible agent.

**Three Primitives:**
- **Knowledge** — MCP resources (documents, context, data the agent can read)
- **Capability** — MCP tools (actions the agent can execute)
- **Flow** — Automation sequences (multi-step processes triggered by events or schedules)

**Namespace support:** Skills can be scoped globally or per named agent.
**Config:** `~/.skskills/` or agent-specific namespace directories.

---

## Installation

### Python (recommended)

```bash
pip install skskills
```

### With CapAuth Integration

```bash
pip install "skskills[capauth]"
```

### With Development Tools

```bash
pip install "skskills[dev]"
```

### From Source

```bash
git clone https://github.com/smilinTux/skskills.git
cd skskills
pip install -e .
```

---

## Quick Start

### Create a New Skill

```bash
skskills init --name my-skill --author "Your Name" --desc "What it does"
```

### Install a Skill

```bash
skskills install ./my-skill
skskills install https://github.com/smilinTux/skmemory.git
```

### List Installed Skills

```bash
skskills list
```

### Run All Enabled Skills

```bash
skskills run
```

---

## CLI Commands

| Command | Flags | Description |
|---------|-------|-------------|
| `skskills init` | `--name NAME`, `--directory DIR`, `--author AUTHOR`, `--desc DESC` | Create a new skill scaffold in the target directory |
| `skskills install SOURCE` | `--agent AGENT`, `--force` | Install a skill from a local path or remote URL |
| `skskills list` | `--agent AGENT` | List all installed skills with status |
| `skskills info NAME` | `--agent AGENT` | Show detailed metadata and primitives for a skill |
| `skskills uninstall NAME` | `--agent AGENT`, `--yes` | Remove an installed skill (--yes skips confirmation) |
| `skskills link NAME` | `--agent AGENT` | Symlink a local directory as a skill for live development |
| `skskills search QUERY` | `--agent AGENT` | Search installed skills by name or description |
| `skskills enable NAME` | `--agent AGENT` | Enable a previously disabled skill |
| `skskills disable NAME` | `--agent AGENT` | Disable a skill without uninstalling it |
| `skskills update NAME SOURCE` | `--agent AGENT` | Update a skill from a new source path or URL |
| `skskills run` | `--agent AGENT` | Start all enabled skill MCP servers for an agent |

---

## Configuration

### Default Paths

```
~/.skskills/
  skills/
    skmemory/               # Installed skill directory
      skill.yaml            # Skill manifest
      src/                  # Skill source
    skseal/
      skill.yaml
      src/
  agents/
    opus/
      skills/               # Agent-specific skill installs
    lumina/
      skills/
  registry.json             # Local index of installed skills
```

### Environment Variables

```bash
export SKSKILLS_HOME=~/.skskills            # Override config root
export SKSKILLS_AGENT=opus                  # Default agent namespace
export SKSKILLS_REGISTRY_URL=https://...    # Remote skill registry URL
export SKSKILLS_CAPAUTH_URL=http://...      # CapAuth server for auth (optional)
```

---

## Skill Manifest (skill.yaml)

Every skill contains a `skill.yaml` at its root:

```yaml
name: my-skill
version: 1.0.0
description: What this skill does
author: Your Name
license: MIT
primitives:
  - knowledge
  - capability
entry: src/server.py
dependencies:
  - mcp
  - pydantic
```

---

## Architecture

```
~/.skskills/
  skills/
    <skill-name>/
      skill.yaml            # Manifest
      src/
        server.py           # MCP server entry point
        knowledge/          # MCP resource handlers
        capabilities/       # MCP tool handlers
        flows/              # Automation flow definitions
  agents/
    <agent-name>/
      skills/               # Agent-namespaced installs (symlinks or copies)
  registry.json             # Installed skill index
```

**Runtime model:**
- `skskills run` discovers all enabled skills for the target agent namespace
- Each skill's MCP server is launched as a subprocess
- The agent framework (Claude Code, OpenClaw, etc.) connects to each MCP server via stdio or SSE transport
- Skills communicate with each other only through the shared agent context

**CapAuth integration (optional):**
- When installed with `[capauth]`, skills can require a valid CapAuth session before exposing tools
- Capability tokens are verified per-tool, per-agent

---

## Support

- GitHub: https://github.com/smilinTux/skskills
- Discord: https://discord.gg/5767MCWbFR
- Email: support@smilintux.org

---

## Philosophy

> *"An agent without skills is just a chatbot. Skills are what make it sovereign."*

SKSkills treats every capability as a first-class citizen — versioned, namespaced, auditable, and replaceable. No monolithic plugin system. No vendor lock-in. Just MCP servers that do one thing well.

**Part of the Penguin Kingdom.**
