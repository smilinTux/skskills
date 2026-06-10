---
name: chat
description: Use when the user types /chat or asks to check/read/reply to skchat messages inline — checks the skchat inbox via MCP, shows new encrypted P2P messages in a table, and sends replies through send_message.
---

# /chat — Inline skchat from Claude Code

Lets you triage and answer encrypted P2P **skchat** messages without leaving the
session. It is a thin front end over the skchat MCP server — it never re-implements
transport, identity, or crypto.

## When to use

- The user types `/chat`.
- The user asks to "check messages", "read my skchat", "any new messages?", or
  "reply to <peer>".

## Prerequisites

The `skchat` MCP server must be registered (tools appear as `mcp__skchat__*`). If
its tools are unavailable, tell the user to register it (`claude mcp add skchat …`)
and stop — do not fall back to shelling out.

## Workflow

1. **Check the inbox.** Call `mcp__skchat__check_inbox` (alias `skchat_inbox`).
   Each message is already decrypted + signature-verified by the daemon.
2. **Display new messages** in a compact table — `From` (short fqid/handle), `Time`
   (local), `Preview` (first ~60 chars), and a `✓`/`✗` signature column. If the
   inbox is empty, say so plainly and stop.
3. **Offer to reply.** If the user wants to respond, ask which message (by sender or
   index) and what to say, then send with `mcp__skchat__send_message` (alias
   `skchat_send`) — pass the recipient as the sender's FQID/handle and the composed
   body. Confirm the send result (delivered/queued) back to the user.
4. **Stay idempotent.** Don't re-send on retry; if a send result is ambiguous, report
   it rather than sending again.

## Notes

- Identity is resolved agent-aware by skchat (capauth `resolve_agent_identity`); do
  not hardcode a `SKCHAT_IDENTITY`.
- Recipients are FQIDs of the form `<agent>@<operator>.<realm>` (e.g.
  `opus@chef.skworld`) or `capauth:<agent>@skworld.io` handles.
- For an at-a-glance list of online peers before replying, use the sibling `/who`
  skill.
