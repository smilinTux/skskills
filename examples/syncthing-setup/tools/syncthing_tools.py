"""Syncthing management tools for the SKSkills syncthing-setup skill.

All tools are framework-free Python functions. They can call the
Syncthing REST API or local CLI depending on what's available.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional


def _default_config_dir() -> Path:
    """Return the default Syncthing config directory."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "syncthing"
    return Path.home() / ".config" / "syncthing"


def _read_api_key(config_dir: Optional[str] = None) -> Optional[str]:
    """Extract the API key from config.xml.

    Args:
        config_dir: Path to Syncthing config directory.

    Returns:
        Optional[str]: API key string, or None if not found.
    """
    cfg_path = Path(config_dir or _default_config_dir()) / "config.xml"
    if not cfg_path.exists():
        return None
    try:
        tree = ET.parse(cfg_path)
        root = tree.getroot()
        gui = root.find("gui")
        if gui is not None:
            key_elem = gui.find("apikey")
            if key_elem is not None and key_elem.text:
                return key_elem.text.strip()
    except ET.ParseError:
        pass
    return None


def _api_get(path: str, api_key: str, port: int = 8384) -> dict[str, Any]:
    """Make a GET request to the Syncthing REST API.

    Args:
        path: API path (e.g., '/rest/system/status').
        api_key: Syncthing API key.
        port: Syncthing GUI port.

    Returns:
        dict: JSON response body.

    Raises:
        RuntimeError: On HTTP or connection error.
    """
    url = f"http://localhost:{port}{path}"
    req = urllib.request.Request(url, headers={"X-API-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        raise RuntimeError(f"Syncthing API error ({url}): {exc}") from exc


def check_status(config_dir: Optional[str] = None) -> dict[str, Any]:
    """Check whether the Syncthing daemon is running and return basic status.

    Args:
        config_dir: Syncthing config directory.

    Returns:
        dict: Status info with keys: running, version, device_id, uptime_s.
    """
    # Check if the process is running via systemctl or pgrep
    running = False
    version = "unknown"

    syncthing_bin = shutil.which("syncthing")
    if syncthing_bin:
        try:
            result = subprocess.run(
                ["syncthing", "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip().split()[1] if result.stdout else "unknown"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # Try the API
    api_key = _read_api_key(config_dir)
    if api_key:
        try:
            status = _api_get("/rest/system/status", api_key)
            return {
                "running": True,
                "version": status.get("version", version),
                "device_id": status.get("myID", "unknown"),
                "uptime_s": status.get("uptime", 0),
                "goroutines": status.get("goroutines", 0),
                "api_accessible": True,
            }
        except RuntimeError:
            pass

    return {
        "running": running,
        "version": version,
        "device_id": None,
        "uptime_s": 0,
        "api_accessible": False,
        "note": "Syncthing may not be running or API is unreachable.",
    }


def get_device_id(config_dir: Optional[str] = None) -> dict[str, Any]:
    """Get this machine's Syncthing Device ID.

    Args:
        config_dir: Syncthing config directory.

    Returns:
        dict: Device ID and cert path.
    """
    api_key = _read_api_key(config_dir)
    if api_key:
        try:
            status = _api_get("/rest/system/status", api_key)
            return {
                "device_id": status.get("myID", ""),
                "source": "api",
            }
        except RuntimeError:
            pass

    # Fall back to CLI
    syncthing_bin = shutil.which("syncthing")
    if syncthing_bin:
        cfg = str(config_dir or _default_config_dir())
        try:
            result = subprocess.run(
                ["syncthing", "--device-id", f"--home={cfg}"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return {"device_id": result.stdout.strip(), "source": "cli"}
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return {
        "device_id": None,
        "source": None,
        "error": "Syncthing not running and CLI unavailable.",
    }


def list_folders(
    api_key: Optional[str] = None,
    port: int = 8384,
) -> dict[str, Any]:
    """List all configured Syncthing folders and their sync status.

    Args:
        api_key: Syncthing API key (auto-detected from config if omitted).
        port: Syncthing GUI port.

    Returns:
        dict: List of folders with sync state info.
    """
    key = api_key or _read_api_key()
    if not key:
        return {"error": "No API key available. Syncthing may not be configured."}

    try:
        folders_cfg = _api_get("/rest/config/folders", key, port)
        result = []
        for folder in folders_cfg:
            folder_id = folder.get("id", "")
            try:
                status = _api_get(f"/rest/db/status?folder={folder_id}", key, port)
            except RuntimeError:
                status = {}

            result.append({
                "id": folder_id,
                "label": folder.get("label", folder_id),
                "path": folder.get("path", ""),
                "type": folder.get("type", "sendreceive"),
                "state": status.get("state", "unknown"),
                "need_bytes": status.get("needBytes", 0),
                "global_bytes": status.get("globalBytes", 0),
                "in_sync_bytes": status.get("inSyncBytes", 0),
            })

        return {"folders": result, "count": len(result)}

    except RuntimeError as exc:
        return {"error": str(exc)}


def generate_config(
    device_name: str,
    sync_dirs: Optional[list[str]] = None,
    gui_password: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a Syncthing configuration with sovereign defaults.

    This does NOT modify the live config — it returns the config XML
    as a string and saves it to a temp file for review.

    Args:
        device_name: Human-readable name for this device.
        sync_dirs: List of directories to configure for sync.
        gui_password: Optional password for the web GUI.

    Returns:
        dict: Generated config path and summary.
    """
    import tempfile
    import hashlib

    sync_dirs = sync_dirs or []

    # Build minimal config XML
    folders_xml = ""
    for i, d in enumerate(sync_dirs):
        safe_id = hashlib.md5(d.encode()).hexdigest()[:8]
        folder_name = Path(d).name or f"folder-{i}"
        folders_xml += f"""
    <folder id="{safe_id}" label="{folder_name}" path="{d}">
        <versioning type="staggered">
            <param key="maxAge" val="15768000"/>
            <param key="cleanInterval" val="3600"/>
        </versioning>
        <order>random</order>
        <rescanIntervalS>3600</rescanIntervalS>
        <autoNormalize>true</autoNormalize>
    </folder>"""

    password_line = ""
    if gui_password:
        import hashlib as _h
        pw_hash = _h.sha256(gui_password.encode()).hexdigest()
        password_line = f"\n        <password>{pw_hash}</password>"

    config_xml = f"""<configuration version="37">
    <device id="PLACEHOLDER-RUN-syncthing-generate" name="{device_name}">
        <address>dynamic</address>
        <introducer>false</introducer>
        <skipIntroductionRemovals>false</skipIntroductionRemovals>
        <encryptionPassword></encryptionPassword>
    </device>
    {folders_xml}
    <gui enabled="true" tls="false">
        <address>127.0.0.1:8384</address>{password_line}
        <insecureAdminAccess>false</insecureAdminAccess>
        <theme>default</theme>
    </gui>
    <options>
        <globalAnnounceEnabled>true</globalAnnounceEnabled>
        <localAnnounceEnabled>true</localAnnounceEnabled>
        <relaysEnabled>true</relaysEnabled>
        <urAccepted>-1</urAccepted>
        <crashReportingEnabled>false</crashReportingEnabled>
        <stunKeepaliveStartS>180</stunKeepaliveStartS>
        <reconnectionIntervalS>60</reconnectionIntervalS>
        <startBrowser>false</startBrowser>
        <autoUpgradeIntervalH>0</autoUpgradeIntervalH>
    </options>
</configuration>
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", prefix="syncthing-config-", delete=False
    ) as f:
        f.write(config_xml)
        tmp_path = f.name

    return {
        "config_path": tmp_path,
        "device_name": device_name,
        "folders_configured": len(sync_dirs),
        "note": (
            f"Config saved to {tmp_path}. "
            "Copy to ~/.config/syncthing/config.xml after review. "
            "Run 'syncthing --generate' first to get your actual Device ID."
        ),
    }
