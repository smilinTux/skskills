"""Syncthing on_boot hook — ensure Syncthing is running when the agent starts."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("syncthing-setup.hooks.on_boot")


def ensure_running() -> dict:
    """Start Syncthing via systemd user service if it isn't already running.

    Returns:
        dict: Result with 'started', 'was_running', and 'message' keys.
    """
    # Check if already running via systemctl
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "syncthing"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip() == "active":
            logger.info("Syncthing already running.")
            return {"started": False, "was_running": True, "message": "Syncthing already running."}
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Attempt to start it
    try:
        result = subprocess.run(
            ["systemctl", "--user", "start", "syncthing"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("Syncthing started via systemctl.")
            return {"started": True, "was_running": False, "message": "Syncthing started."}
        else:
            logger.warning("Failed to start Syncthing: %s", result.stderr)
            return {
                "started": False,
                "was_running": False,
                "message": f"Could not start Syncthing: {result.stderr.strip()}",
            }
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        return {"started": False, "was_running": False, "message": f"systemctl not available: {exc}"}
