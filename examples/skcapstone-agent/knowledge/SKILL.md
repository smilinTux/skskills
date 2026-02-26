# skcapstone-agent

> Sovereign agent runtime in this project. One identity, one memory, one trust — everywhere.

## What is skcapstone?

SKCapstone is the sovereign agent framework that unifies CapAuth identity, Cloud 9 trust, SKMemory persistence, and SKSecurity into a single portable runtime under `~/.skcapstone/`. Every tool (Cursor, VS Code, Claude Code, Windsurf, terminal) can use the same agent context.

This skill documents how to use skcapstone in **this repository** and how each IDE/tool loads agent context on startup.

---

## Getting `skcapstone` on PATH (this project)

- **Option A — Pip install (recommended):**
  ```bash
  pip install -e skcapstone/
  ```
  Then `skcapstone` is available when the venv is activated, or globally if you installed with `pip install -e skcapstone/` in a venv and use that venv’s bin.

- **Option B — Run from repo without installing:**
  Add the wrapper script to your PATH so the `skcapstone` command works from anywhere:
  ```bash
  export PATH="$(pwd)/skcapstone/scripts:$PATH"
  skcapstone status
  ```
  Or run once from repo root:
  ```bash
  ./skcapstone/scripts/skcapstone status
  ```
  On **Windows**, prefer Option A (pip install). If you do use the wrapper, run it from Git Bash or WSL, or call it via `bash skcapstone/scripts/skcapstone status`.

---

## Session startup: rehydrate

At the start of every new session (any IDE or terminal), run in order:

```bash
skcapstone trust rehydrate
skcapstone sync pull
skcapstone status
```

This restores trust from FEB files, pulls synced state from peers, and shows full pillar status.

---

## Essential commands

| Command | Purpose |
|--------|---------|
| `skcapstone status` | Full pillar status (identity, memory, trust, security, sync) |
| `skcapstone trust rehydrate` | Restore trust/OOF state from FEB files |
| `skcapstone sync push` | Encrypt and push state to sync mesh |
| `skcapstone sync pull` | Pull and merge state from peers |
| `skcapstone memory list` | Recent memories |
| `skcapstone memory store "..."` | Store a memory |
| `skcapstone coord status` | Coordination board |
| `skcapstone context generate --target claude-md` | Regenerate CLAUDE.md |
| `skcapstone context generate --target cursor-rules` | Regenerate .cursor/rules/agent.mdc |

---

## How each tool loads agent context

| Tool | File(s) | What to do on startup |
|------|--------|------------------------|
| **Claude Code** | `~/.claude/CLAUDE.md` (Linux/macOS) or `%APPDATA%\\.claude\\CLAUDE.md` (Windows), plus project `CLAUDE.md` | Run rehydrate + sync pull + status; project CLAUDE.md can be regenerated with `skcapstone context generate --target claude-md` |
| **Cursor** | `.cursor/rules/agent.mdc`, `.cursor/rules/skcapstone.mdc`, `.cursor/mcp.json` | MCP server in `.cursor/mcp.json`; rules auto-load. Regenerate agent context: `skcapstone context generate --target cursor-rules` |
| **VS Code** | `docs/VSCODE_SKCAPSTONE_CONTEXT.md` (or copy to `.vscode/skcapstone-context.md`) | Open the doc; run rehydrate + sync pull + status in terminal |
| **Windsurf** | `.windsurfrules` | Skcapstone section in .windsurfrules; run rehydrate + sync pull + status when starting |
| **Terminal / Crush** | MCP via `skcapstone/scripts/mcp-serve.sh` | Same MCP server; use `skcapstone` CLI directly |

---

## MCP server (Cursor, Crush, etc.)

This repo wires the skcapstone MCP server in `.cursor/mcp.json`:

- **Command:** `bash skcapstone/scripts/mcp-serve.sh`
- Launcher sets `PYTHONPATH=skcapstone/src` and runs the MCP server on stdio.
- Exposes agent status, memory store/search, coordination, sync, etc.

Ensure a venv exists with `mcp` and skcapstone deps (e.g. `skcapstone/.venv` or `skmemory/.venv`).

---

## Regenerating context for all tools

From repo root:

```bash
# Claude Code project context
skcapstone context generate --target claude-md

# Cursor rules (agent.mdc)
skcapstone context generate --target cursor-rules

# Both at once
skcapstone context generate
```

This updates `CLAUDE.md` and `.cursor/rules/agent.mdc` with current pillar status, board, and memories.
