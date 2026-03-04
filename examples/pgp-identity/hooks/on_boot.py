"""on_boot hook: verify agent PGP identity key exists."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("pgp-identity.on_boot")


def verify_identity() -> None:
    """Check that the agent's fingerprint is in the keyring on startup."""
    fingerprint = os.environ.get("AGENT_FINGERPRINT", "")
    if not fingerprint:
        logger.debug("AGENT_FINGERPRINT not set — skipping PGP identity check")
        return

    try:
        import gnupg
        gpg = gnupg.GPG()
        keys = gpg.list_keys(secret=True)
        found = any(k["fingerprint"] == fingerprint.upper() for k in keys)
        if found:
            logger.info("PGP identity verified: %s", fingerprint[:16])
        else:
            logger.warning(
                "Agent fingerprint %s not found in keyring — "
                "run: skskills run pgp-identity.generate_key",
                fingerprint[:16],
            )
    except ImportError:
        logger.debug("gnupg not installed — skipping PGP identity check")
    except Exception as exc:
        logger.warning("PGP identity check failed: %s", exc)
