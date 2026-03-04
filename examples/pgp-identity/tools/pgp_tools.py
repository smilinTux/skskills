"""PGP identity tools for sovereign agents.

Wraps the gnupg Python library to manage keys and signatures.
"""

from __future__ import annotations

import os
from typing import Any, Optional


def _gpg():
    import gnupg
    return gnupg.GPG(gnupghome=os.environ.get("GNUPGHOME", os.path.expanduser("~/.gnupg")))


def list_keys(secret_only: bool = False, fingerprint: Optional[str] = None) -> dict[str, Any]:
    """List PGP keys in the keyring."""
    gpg = _gpg()
    keys = gpg.list_keys(secret=secret_only)

    result = []
    for key in keys:
        fp = key["fingerprint"]
        if fingerprint and not fp.startswith(fingerprint.upper()):
            continue
        result.append({
            "fingerprint": fp,
            "keyid": key.get("keyid", ""),
            "uids": key.get("uids", []),
            "length": key.get("length", ""),
            "algo": key.get("algo", ""),
            "expires": key.get("expires", ""),
            "trust": key.get("trust", ""),
            "type": "secret" if secret_only else "public",
        })

    return {"keys": result, "count": len(result)}


def generate_key(
    name: str,
    email: str,
    comment: str = "sovereign agent",
    key_type: str = "ed25519",
    expire: str = "0",
) -> dict[str, Any]:
    """Generate a new PGP keypair."""
    gpg = _gpg()

    input_data = gpg.gen_key_input(
        key_type="EDDSA" if key_type == "ed25519" else "RSA",
        key_length=4096 if key_type == "rsa4096" else None,
        key_curve="Ed25519" if key_type == "ed25519" else None,
        name_real=name,
        name_email=email,
        name_comment=comment,
        expire_date=expire,
        no_protection=True,  # For automated agent use; set passphrase for production
    )

    result = gpg.gen_key(input_data)
    if not result.fingerprint:
        return {"ok": False, "error": str(result)}

    return {
        "ok": True,
        "fingerprint": result.fingerprint,
        "name": name,
        "email": email,
        "key_type": key_type,
    }


def sign_message(
    message: Optional[str] = None,
    file_path: Optional[str] = None,
    fingerprint: Optional[str] = None,
    armor: bool = True,
) -> dict[str, Any]:
    """Create a detached PGP signature."""
    gpg = _gpg()

    if file_path:
        with open(file_path, "rb") as f:
            data = f.read()
    elif message:
        data = message.encode()
    else:
        return {"ok": False, "error": "Provide either 'message' or 'file_path'"}

    kwargs: dict[str, Any] = {"detach": True}
    if fingerprint:
        kwargs["keyid"] = fingerprint

    signed = gpg.sign(data, **kwargs)
    if not signed:
        return {"ok": False, "error": "Signing failed — check key availability"}

    return {
        "ok": True,
        "signature": str(signed),
        "fingerprint": signed.fingerprint,
    }


def verify_signature(
    message: str,
    signature: str,
    expected_fingerprint: Optional[str] = None,
) -> dict[str, Any]:
    """Verify a detached PGP signature."""
    import tempfile

    gpg = _gpg()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".asc", delete=False) as sig_file:
        sig_file.write(signature)
        sig_path = sig_file.name

    try:
        verified = gpg.verify_data(sig_path, message.encode())
    finally:
        os.unlink(sig_path)

    result = {
        "ok": bool(verified),
        "fingerprint": verified.fingerprint or "",
        "key_id": verified.key_id or "",
        "status": verified.status,
        "timestamp": verified.timestamp,
    }

    if expected_fingerprint and verified.fingerprint:
        result["fingerprint_match"] = verified.fingerprint.endswith(
            expected_fingerprint.upper()
        )

    return result


def export_public_key(fingerprint: Optional[str] = None) -> dict[str, Any]:
    """Export a public key as ASCII armor."""
    gpg = _gpg()

    if not fingerprint:
        keys = gpg.list_keys(secret=True)
        if not keys:
            return {"ok": False, "error": "No secret keys found in keyring"}
        fingerprint = keys[0]["fingerprint"]

    armored = gpg.export_keys(fingerprint)
    if not armored:
        return {"ok": False, "error": f"Key not found: {fingerprint}"}

    return {
        "ok": True,
        "fingerprint": fingerprint,
        "public_key": armored,
    }
