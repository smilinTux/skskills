# syncthing-setup

> Sovereign peer-to-peer file synchronization. No cloud. No middleman. You own your data.

## What is Syncthing?

Syncthing is an open-source, decentralized file synchronization tool. Unlike Dropbox or Google Drive:
- Files sync **directly between your devices** — no cloud intermediary
- **End-to-end encrypted** in transit (TLS 1.3)
- **Conflict resolution** built in
- **Works offline** — syncs when peers reconnect
- **No account required** — identity is a device certificate

This skill covers complete Syncthing setup for sovereign agents and their humans.

---

## Architecture

```
Device A (your laptop)          Device B (your server)
  ~/.config/syncthing/            ~/.config/syncthing/
       |                               |
  Device ID: AABBCC...           Device ID: XXYYZZ...
       |                               |
       +-------- TLS sync ------------>+
       |      (direct P2P)             |
       +<------- TLS sync -------------+

No relay unless both devices are behind NAT and can't reach each other directly.
```

### Global Discovery vs. Local Discovery

- **Local Discovery**: mDNS broadcast on LAN. No config needed.
- **Global Discovery**: Syncthing discovery server (`discovery.syncthing.net`). Can be self-hosted.
- **Relay**: Last resort. Syncthing maintains community relays. Can self-host with `strelaysrv`.

For sovereign setups: run your own discovery server and relay to avoid any dependency on external infrastructure.

---

## Initial Setup

### 1. Install Syncthing

```bash
# Arch / Manjaro
sudo pacman -S syncthing

# Ubuntu / Debian
sudo apt install syncthing

# From release tarball (any Linux)
curl -L https://github.com/syncthing/syncthing/releases/latest/download/syncthing-linux-amd64.tar.gz | tar xz
sudo mv syncthing-*/syncthing /usr/local/bin/
```

### 2. First Run

```bash
syncthing --generate ~/.config/syncthing
syncthing serve --no-browser
```

Syncthing generates a unique device certificate on first run. Your **Device ID** is derived from this certificate — it's your identity on the mesh.

### 3. Get Your Device ID

```bash
syncthing --device-id
# or via API:
curl -H "X-API-Key: <your-api-key>" http://localhost:8384/rest/system/status | jq .myID
```

Share this Device ID with peers who want to sync with you.

### 4. Enable as a System Service

```bash
# User-level systemd service (recommended for agents)
systemctl --user enable syncthing
systemctl --user start syncthing

# System-level (for always-on servers)
sudo systemctl enable syncthing@<username>
sudo systemctl start syncthing@<username>
```

---

## Folder Configuration

### Adding a Folder

Via web GUI: http://localhost:8384 → Add Folder

Via config (`~/.config/syncthing/config.xml`):

```xml
<folder id="sovereign-docs" label="Sovereign Docs" path="/home/user/sovereign-docs">
    <device id="PEER-DEVICE-ID"></device>
    <versioning type="staggered">
        <param key="maxAge" val="15768000"/>
        <param key="cleanInterval" val="3600"/>
        <param key="versionsPath" val=""/>
    </versioning>
    <order>random</order>
    <rescanIntervalS>3600</rescanIntervalS>
    <ignorePerms>false</ignorePerms>
    <autoNormalize>true</autoNormalize>
</folder>
```

### Versioning (Recommended for Sovereign Use)

Enable **Staggered File Versioning** to keep file history:
- Keeps hourly versions for 24 hours
- Daily versions for 30 days  
- Weekly versions for 1 year
- Monthly versions forever

This is your personal time machine. No cloud needed.

---

## Sovereign Sync Topology

### Hub-and-Spoke (Recommended for Agents)

```
                  Server (always on)
                  /       |       \
          Laptop    Phone    Agent-VM
```

The server syncs with everyone. Devices sync via the server when they can't reach each other directly.

### Mesh (Fully Sovereign)

```
  Laptop <--> Phone
    ^  \       ^
    |   \     /
    v    v   v
  Server <--> Agent-VM
```

Every device syncs directly with every other. No single point of failure.

---

## API Reference

Base URL: `http://localhost:8384`

All requests require: `X-API-Key: <your-api-key>` header.
API key is in `~/.config/syncthing/config.xml` → `<apikey>` field.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rest/system/status` | GET | Device status, version, uptime |
| `/rest/system/config` | GET | Full configuration |
| `/rest/db/status?folder=<id>` | GET | Folder sync status |
| `/rest/db/browse?folder=<id>` | GET | Browse folder contents |
| `/rest/config/devices` | GET | List paired devices |
| `/rest/config/folders` | GET | List configured folders |
| `/rest/events` | GET | Event stream (long-poll) |

---

## Integration with SKCapstone

Syncthing is the transport layer for skcapstone seed/vault sync:

```
~/.skcapstone/sync/
    outbox/     <- Syncthing watches this, syncs to peers
    inbox/      <- Syncthing drops received files here
```

When CapAuth encrypts a seed and drops it in `outbox/`, Syncthing propagates it to all connected devices. No cloud. No intermediary. Pure sovereign P2P.

---

## Security Notes

1. **API key** — treat like a password. Never expose port 8384 to the internet.
2. **Firewall** — Syncthing needs port 22000/TCP (sync) and 21027/UDP (local discovery).
3. **Untrusted devices** — Use `introducer: false` and manually approve new devices.
4. **Relay traffic** — Relays see only encrypted data. Can self-host with `strelaysrv`.

---

*Syncthing is the backbone of sovereign data mobility. Your files follow you, not a corporation.*

**#staycuriousANDkeepsmilin**
