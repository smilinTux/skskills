"""on_memory_stored hook: optionally sign high-importance memories."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("pgp-identity.on_memory_stored")


def sign_memory(memory: dict[str, Any]) -> None:
    """Sign memories with importance >= 0.9 for non-repudiation."""
    importance = memory.get("importance", 0.0)
    if importance < 0.9:
        return

    fingerprint = os.environ.get("AGENT_FINGERPRINT", "")
    if not fingerprint:
        return

    try:
        import gnupg
        gpg = gnupg.GPG()
        content = memory.get("content", "")
        signed = gpg.sign(content.encode(), keyid=fingerprint, detach=True)
        if signed:
            logger.debug(
                "Signed high-importance memory (importance=%.2f) with key %s",
                importance,
                fingerprint[:16],
            )
    except Exception as exc:
        logger.debug("Memory signing skipped: %s", exc)
