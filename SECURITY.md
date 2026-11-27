# Security Policy - skskills

## Contents

- [Reporting a vulnerability](#reporting-a-vulnerability)
  - [Safe harbour](#safe-harbour)
- [Scope](#scope)
- [Supported versions](#supported-versions)
- [The signature gap (read this first)](#the-signature-gap-read-this-first)
  - [What integrity checking does exist](#what-integrity-checking-does-exist)
- [Threat model (summary)](#threat-model-summary)
  - [Release integrity is not gated](#release-integrity-is-not-gated)
- [Secret handling](#secret-handling)
- [Dependency posture](#dependency-posture)
- [What this repo does NOT claim](#what-this-repo-does-not-claim)

`skskills` installs and executes third-party code. That is its entire job: a skill is
a manifest plus payload, and running a skill means importing a Python dotpath or
spawning an executable that came from a path, a git repo, a pip package, or a remote
registry. **Trust in a skill is trust in whoever wrote it**, and this repo does not
currently give you a cryptographic way to establish that trust. Read
[§ The signature gap](#the-signature-gap-read-this-first) before relying on it.

**Maturity-tier:** T0 (classical). skskills is **not** a crypto component: it
generates, exchanges, signs, wraps, and stores no key material. It uses SHA-256 only
for tarball integrity.

---

## Reporting a vulnerability

Report privately. Do **not** open a public issue for a security bug.

- **Primary channel:** GitHub **private vulnerability reporting** on this repo
  (`Security` then `Report a vulnerability`). This keeps the report, the fix, and the
  advisory in one place.
- **Secondary (out of band):** a PGP-encrypted report to the SKWorld maintainers via
  the smilinTux CapAuth identity, or through the SKCapstone coordination channel, for
  when GitHub is unavailable or you prefer not to use it.
- **Acknowledgement SLA: 72 hours.** If you have heard nothing in 72 hours, assume the
  report did not arrive and try the secondary channel. Silence is a delivery failure,
  not a decision.

Include the affected version (say which file you read it from, see
[SOP.md §5.2](SOP.md#52-version-bump-read-this-before-tagging)), a reproduction, and
the impact you actually observed.

Coordinated disclosure, per
[SECURITY_DISCLOSURE_STANDARD](https://github.com/smilinTux/sk-standards/blob/main/standards/SECURITY_DISCLOSURE_STANDARD.md):
embargo until a fix ships, target 90 days or sooner, active exploitation collapses the
embargo, reporter credited with consent.

### Safe harbour

Good-faith security research conducted under coordinated disclosure will not be
pursued or reported. Test against your own installation, do not access data that is
not yours, do not degrade a service for others, and give us a reasonable window before
publishing. Work within that and you are welcome here.

---

## Scope

**In scope**

- The code in `src/skskills/` and the published artifacts built from it (`skskills` on
  PyPI, `@smilintux/skskills` on npm).
- The install, load, and execution paths: `registry.py`, `loader.py`,
  `aggregator.py`, `remote.py`, `pip_bridge.py`, `installer.py`.
- The plugin compiler and its outbound scrub gate, `src/skskills/plugin/`.
- The four skills carried in this repo under `skills/`.

**Out of scope**

- **Third-party skills.** A malicious skill running its own code is the documented
  behaviour of a skill runner, not a vulnerability in the runner. A way for a skill to
  escape a boundary skskills *claims* to enforce would be in scope; today skskills
  claims no such boundary (see the threat model).
- **The remote registry service** at `skills.smilintux.org`. This repo ships only the
  HTTP client.
- **Upstream dependencies** (`click`, `mcp`, `pydantic`, `pyyaml`, `rich`). Report
  upstream; we track and bump.
- Findings already recorded in `CHANGELOG.md` or in this file as a known gap.
- Theoretical findings with no realisable impact on a supported configuration.

## Supported versions

| Version | Status | Fixes |
|---|---|---|
| 0.2.x (`pyproject.toml` / `__init__.py`) | Active | Yes |
| 0.1.x (as published to npm, and as declared in `package.json` / `skill.yaml` / `SKILL.md`) | Legacy, and a symptom of the version drift in [SOP.md §5.2](SOP.md#52-version-bump-read-this-before-tagging) | Critical only |
| Anything pre-0.1 | Unsupported | No |

The version drift is itself a reporting hazard: the npm artifact and the PyPI artifact
built from the same commit announce different versions. When you report, name the file
you read the version from.

---

## The signature gap (read this first)

**skskills stores and displays a signature claim that it never verifies.**

- `src/skskills/models.py:147` defines
  `signature: str = Field(default="", description="CapAuth detached PGP signature of skill.yaml")`
  and line 148 defines `signed_by`.
- `SkillManifest.is_signed()` (line 178) returns
  `bool(self.signature and self.signed_by)`. That is a **non-empty-string check**, not
  a cryptographic one.
- `skskills list` prints a "signed" column and the `skskills.info` MCP tool returns
  `signed` / `signed_by`, both fed from that check.
- Verified 2026-08-14: a grep of all of `src/` for `gpg`, `pgp`, `verify`, or
  `capauth` returns only docstrings, a CLI help string, and a SHA-256 checksum
  comparison. **No signature verification code exists in this repository.**

What that means in practice: any attacker who can write a `skill.yaml` can set
`signature` and `signed_by` to arbitrary strings and the skill will render as signed,
attributed to whoever they name. Do not use the "signed" column as a trust signal, in
a UI, in a policy decision, or in your head.

This is stated here as a **gap, not a mitigation**. The fix is real verification
against a capauth-held public key, at install time and at load time, with an
unverifiable signature failing closed. Until that exists, treat every skill as
unsigned.

### What integrity checking does exist

`remote.py:228-234` compares the SHA-256 of a downloaded tarball against the `sha256`
recorded in the registry entry, and refuses the install on a mismatch. That defends
against a corrupted or truncated transfer. It does **not** defend against a registry
that serves a malicious tarball together with a matching hash, because nothing signs
the hash.

---

## Threat model (summary)

| Asset | Threat | Control today |
|---|---|---|
| Arbitrary code execution | A skill's `entrypoint` runs in the agent's process (dotpath / `.py` file) or as a subprocess (executable). | **None.** No sandbox, no capability restriction, no allowlist. Installing a skill grants it everything the agent has. This is the design; know it before you install. |
| Skill provenance | A forged `signed_by` attributes a malicious skill to a trusted author. | **None.** See the signature gap above. |
| Transport integrity of a pulled skill | A corrupted or tampered tarball. | SHA-256 comparison against the registry entry (`remote.py:229`). Unsigned, so it does not stop a hostile registry. |
| Namespace confusion | Two skills exporting the same base tool name, so an agent invokes the wrong one. | Fully-qualified `skill.tool` names stay unique; overlaps are surfaced, not hidden, via the `skskills.collisions` MCP tool. |
| Cross-agent leakage | An agent seeing another agent's skills. | Namespace separation under `~/.skskills/agents/<agent>/`, resolved agent-first then global. Filesystem permissions only; not an enforced boundary. |
| Outbound disclosure at publish | A compiled plugin artifact leaking a token or a private endpoint to a public marketplace. | **Real control.** `plugin/scrub.py` scans every emitted file and blocks on `ghp_`, `github_pat_`, `xox[baprs]-`, `AKIA`, PEM private-key headers, RFC1918 addresses, CGNAT/tailnet `100.64.0.0/10`, and `*.ts.net`. Publishing additionally requires `publish: true` in `catalog.yaml` and an explicit `--i-am-chef` flag. |
| Secrets in the repo | A credential committed to a public repo. | `.github/workflows/secret-scan.yml` runs the **gitleaks binary** (8.28.0, pinned) over the **full history** on every push and pull request, with `--redact` and `--exit-code 1`. The binary rather than `gitleaks-action`, because that wrapper needs a paid licence for organization-owned repos and exits before scanning anything. |
| Release integrity | A broken or malicious build reaching PyPI and npm. | **Weak, and known.** See below. |

### Release integrity is not gated

`.github/workflows/ci.yml` runs `python -m pytest tests/ -v --tb=short || true`, so the
test job cannot fail a run. `.github/workflows/publish.yml` repeats that job and then
declares `publish-pypi` and `publish-npm` with `needs: test` **plus** `if: always()`.
A total test failure still publishes to both registries.

Tracked as coordination card **62a5256d**. It is left in place rather than silently
removed because a test currently fails for an environmental reason
([SOP.md §4](SOP.md#4-test)) and removing `|| true` first would only turn `main` red.
The correct order is: fix the test, then remove `|| true`, then remove `if: always()`
from the publish jobs.

**Do not treat a green CI badge on this repo as evidence of anything.** Only the
`lint` job and `secret-scan` can fail today.

---

## Secret handling

**This repo stores no secrets and must never store one.**

- The only credential the code touches is `SKSKILLS_TOKEN`, a CapAuth bearer token
  read from the environment for `skskills publish` (`cli.py:399`). It is never
  written to disk by this code and must never appear in a `skill.yaml`, in
  `catalog.yaml`, or in a committed config.
- CI credentials (`PYPI_API_TOKEN`, `NPM_TOKEN`) live as GitHub repository secrets.
- If a secret ever lands: **rotate first**, prove the new credential works with a call
  that genuinely authenticates rather than a list endpoint that ignores the header,
  then revoke the old one, then purge the history. Never allowlist a real finding to
  quiet the scanner.

## Dependency posture

- Runtime dependencies are pinned to major-version ranges in `pyproject.toml`:
  `click>=8.1,<9.0`, `mcp>=1.0,<2.0`, `pydantic>=2.0,<3.0`, `pyyaml>=6.0,<7.0`,
  `rich>=13.0,<14.0`. `capauth` is an **optional** extra and is not imported anywhere
  in `src/`.
- The linter is pinned (`ruff==0.15.4`) so a new upstream rule cannot redden `main`
  with no code change.
- YAML is always read with `yaml.safe_load`. Note that `safe_load` silently accepts
  duplicate keys, so a hand-edited `catalog.yaml` or `skill-roots.yaml` can lose an
  entry without any error.

## What this repo does NOT claim

- It does **not** claim skills are sandboxed, isolated, or capability-restricted.
- It does **not** claim a skill's author is verified. See the signature gap.
- It does **not** claim a published artifact was tested. See release integrity.
- It makes no post-quantum claim of any kind, because it performs no key exchange and
  no signing. SHA-256 here is a symmetric integrity primitive with a Grover-only
  attack surface, and needs no replacement.
