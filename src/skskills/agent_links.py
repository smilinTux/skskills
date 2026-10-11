"""Project installed skills into every agent framework's skill directory.

A skill installed into the registry (``~/.skskills/installed/<name>``) is only
useful to an agent that looks there. Claude Code, Codex, pi, Hermes, and
OpenCode each load ``<name>/SKILL.md`` from their own user-level directory, so
every global install that ships a ``SKILL.md`` is symlinked, whole directory,
into each framework found on this machine:

    claude-code  ~/.claude/skills/<name>
    codex        $CODEX_HOME/skills/<name>            (default ~/.codex)
    pi           $PI_CODING_AGENT_DIR/skills/<name>   (default ~/.pi/agent)
    hermes       ~/.hermes/skills/skskills/<name>     (Hermes groups by category)
    opencode     ~/.config/opencode/skills/<name>     only when Claude Code is
                                                       absent: OpenCode already
                                                       reads ~/.claude/skills and
                                                       wants unique names

Only symlinks that point into the registry are ever created, replaced, or
removed. A hand-made directory or a symlink to anywhere else is reported as a
conflict and left alone. Set ``SKSKILLS_AGENT_LINKS=0`` to turn linking off.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .registry import SkillRegistry

ENV_SWITCH = "SKSKILLS_AGENT_LINKS"
HERMES_CATEGORY = "skskills"
HERMES_DESCRIPTION = "Skills installed and kept in sync by skskills.\n"


def enabled() -> bool:
    """False when SKSKILLS_AGENT_LINKS is set to 0/false/no/off."""
    return os.environ.get(ENV_SWITCH, "1").strip().lower() not in {"0", "false", "no", "off"}


def detect_targets(home: Optional[Path] = None) -> dict[str, Path]:
    """Map each agent framework present on this machine to its skill directory."""
    home = Path(home) if home else Path.home()
    targets: dict[str, Path] = {}

    if (home / ".claude").is_dir():
        targets["claude-code"] = home / ".claude" / "skills"

    codex_home = Path(os.environ.get("CODEX_HOME") or home / ".codex")
    if codex_home.is_dir() or shutil.which("codex"):
        targets["codex"] = codex_home / "skills"

    pi_home = Path(os.environ.get("PI_CODING_AGENT_DIR") or home / ".pi" / "agent")
    if pi_home.is_dir() or shutil.which("pi"):
        targets["pi"] = pi_home / "skills"

    if (home / ".hermes").is_dir():
        targets["hermes"] = home / ".hermes" / "skills" / HERMES_CATEGORY

    opencode_present = (home / ".config" / "opencode").is_dir() or shutil.which("opencode")
    if opencode_present and "claude-code" not in targets:
        targets["opencode"] = home / ".config" / "opencode" / "skills"

    return targets


def _is_managed(link: Path, installed_dir: Path) -> bool:
    """True when link is a symlink whose target lies inside the registry."""
    if not link.is_symlink():
        return False
    raw = Path(os.readlink(link))
    if not raw.is_absolute():
        raw = link.parent / raw
    raw = Path(os.path.normpath(raw))
    return any(raw.is_relative_to(base) for base in (installed_dir, installed_dir.resolve()))


def _prepare(framework: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if framework == "hermes":
        desc = target_dir / "DESCRIPTION.md"
        if not desc.exists():
            desc.write_text(HERMES_DESCRIPTION, encoding="utf-8")


def link_skill(
    skill_dir: Path,
    installed_dir: Path,
    targets: Optional[dict[str, Path]] = None,
) -> dict[str, str]:
    """Symlink one installed skill into every target. Returns framework -> action.

    Actions: ``linked`` (created or repointed), ``ok`` (already correct),
    ``conflict`` (something unmanaged holds the name), ``skipped`` (no SKILL.md).
    """
    targets = detect_targets() if targets is None else targets
    if not (skill_dir / "SKILL.md").is_file():
        return {fw: "skipped" for fw in targets}

    results: dict[str, str] = {}
    for framework, target_dir in targets.items():
        link = target_dir / skill_dir.name
        if link.is_symlink() and link.resolve() == skill_dir.resolve():
            results[framework] = "ok"
            continue
        if link.exists() or link.is_symlink():
            if not _is_managed(link, installed_dir):
                results[framework] = "conflict"
                continue
            link.unlink()
        _prepare(framework, target_dir)
        link.symlink_to(skill_dir.resolve(), target_is_directory=True)
        results[framework] = "linked"
    return results


def unlink_skill(
    name: str,
    installed_dir: Path,
    targets: Optional[dict[str, Path]] = None,
) -> list[Path]:
    """Remove the managed links for one skill. Returns the links removed."""
    targets = detect_targets() if targets is None else targets
    removed = []
    for target_dir in targets.values():
        link = target_dir / name
        if _is_managed(link, installed_dir):
            link.unlink()
            removed.append(link)
    return removed


def sync(registry: "SkillRegistry", targets: Optional[dict[str, Path]] = None) -> dict:
    """Link every global install and prune managed links whose skill is gone."""
    targets = detect_targets() if targets is None else targets
    installed_dir = registry.installed_dir
    linked = pruned = 0
    conflicts: list[str] = []

    if installed_dir.is_dir():
        for skill_dir in sorted(p for p in installed_dir.iterdir() if p.is_dir()):
            for framework, action in link_skill(skill_dir, installed_dir, targets).items():
                if action == "linked":
                    linked += 1
                elif action == "conflict":
                    conflicts.append(f"{framework}:{skill_dir.name}")

    for target_dir in targets.values():
        if not target_dir.is_dir():
            continue
        for link in target_dir.iterdir():
            if _is_managed(link, installed_dir) and not (link / "SKILL.md").is_file():
                link.unlink()
                pruned += 1

    return {"targets": {k: str(v) for k, v in targets.items()}, "linked": linked,
            "pruned": pruned, "conflicts": conflicts}


def status(registry: "SkillRegistry", targets: Optional[dict[str, Path]] = None) -> dict[str, dict[str, str]]:
    """skill -> framework -> ok | missing | conflict, for installs with a SKILL.md."""
    targets = detect_targets() if targets is None else targets
    installed_dir = registry.installed_dir
    report: dict[str, dict[str, str]] = {}
    if not installed_dir.is_dir():
        return report
    for skill_dir in sorted(p for p in installed_dir.iterdir() if (p / "SKILL.md").is_file()):
        row = {}
        for framework, target_dir in targets.items():
            link = target_dir / skill_dir.name
            if link.is_symlink() and link.resolve() == skill_dir.resolve():
                row[framework] = "ok"
            elif link.exists() or link.is_symlink():
                row[framework] = "conflict"
            else:
                row[framework] = "missing"
        report[skill_dir.name] = row
    return report
