# pgp-identity skill

Sovereign PGP key management for agent identity and CapAuth integration.

## Overview

This skill provides tools for managing the PGP keys that underpin the
sovereign agent identity system (CapAuth). Every sovereign agent has a PGP
keypair — the fingerprint is the agent's canonical identity across all
services (Authentik, Forgejo, SKChat, SKSeal, etc.).

## Key concepts

- **Fingerprint** — 40-char hex identifier derived from the public key.
  This is the agent's immutable identity.
- **Keyring** — GPG keyring at `~/.gnupg/` (or `$GNUPGHOME`).
- **CapAuth** — Uses PGP signing challenges for passwordless login.
- **SKSeal** — Uses PGP signatures for document signing.

## Quick start

```bash
# List keys in keyring
skskills run pgp-identity.list_keys

# Generate a new identity key
skskills run pgp-identity.generate_key --name "Lumina" --email "lumina@skworld.io"

# Export public key for sharing
skskills run pgp-identity.export_public_key
```

## CapAuth integration

The agent's PGP key is the master credential. When logging in via CapAuth:
1. The server sends a random challenge nonce
2. The agent signs the nonce with its private key
3. The server verifies the signature against the stored public key

The fingerprint in `CLAUDE.md` (`CCBE9306410CF8CD5E393D6DEC31663B95230684`)
is this agent's identity. Never share the private key.

## Key storage

Keys live in the GPG keyring (`~/.gnupg/`). For sovereign deployments,
back up the keyring to SKData or export to a hardware token (YubiKey).
