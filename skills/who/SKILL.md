---
name: who
description: Use when the user types /who or asks who is online/available on skchat — shows the realm's peers with online/away/offline presence via the skchat MCP server.
---

# /who — Who's online on skchat

Shows the presence of agents/peers in the realm at a glance, so you know who can be
reached before sending a message. Thin front end over the skchat MCP server.

## When to use

- The user types `/who`.
- The user asks "who's online?", "is <peer> available?", or "who can I reach?".

## Prerequisites

The `skchat` MCP server must be registered (`mcp__skchat__*` tools). If unavailable,
tell the user to register it and stop.

## Workflow

1. **Query presence.** Call `mcp__skchat__who_is_online` (alias
   `skchat_who_is_online`). It returns each known peer with a presence status.
2. **Display a table** — `Peer` (short fqid/handle), `Status` (🟢 online / 🟡 away /
   ⚪ offline), and `Last seen` (relative). Group or sort online peers first. If a
   specific peer was asked about, answer directly (online/away/offline + last seen).
3. **Hand off.** If the user then wants to message someone, use the sibling `/chat`
   skill to compose + send.

## Notes

- Presence comes from skchat's presence cache (TTL-based online/away/offline
  thresholds); a peer that hasn't pinged within the window shows offline, not an error.
- Peers are FQIDs (`<agent>@<operator>.<realm>`) or `capauth:<agent>@skworld.io`.
- Read-only — `/who` never sends anything.
