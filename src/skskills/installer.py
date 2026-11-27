"""SKSkills installer API — the stable, high-level install surface.

This module is the public install API that orchestrators (e.g. skcapstone's
install wizard via ``_install_default_skills``) import:

    from skskills.installer import install_from_catalog, install_from_local

It unifies the three install sources behind two functions:

  * ``install_from_local(path)``   — a local skill directory containing skill.yaml.
  * ``install_from_catalog(name)`` — resolve a catalog entry by name and install
    it from whichever coordinate it declares: ``local`` (first-party dir),
    ``git_path`` (bundled inside the skskills repo), or ``pip`` (PyPI package).

Both return the :class:`~skskills.models.InstalledSkill` (or raise on failure),
and both default to ``force=True`` so re-running an install is idempotent —
the registry overwrites the snapshot under ``~/.skskills/installed/<name>``.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .catalog import SkillCatalog
from .models import InstalledSkill
from .pip_bridge import install_from_pip
from .registry import SkillRegistry

logger = logging.getLogger(__name__)

# Repo root that bundles example/first-party skills referenced by git_path.
# installer.py lives at <repo>/src/skskills/installer.py → parents[2] == <repo>.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Fallback source tree for first-party Claude Code skills resolved by bare name.
_LOCAL_SKILLS_DIR = Path("~/clawd/skills").expanduser()


def _install_git_entry(entry, name: str, agent: str, force: bool) -> InstalledSkill:
    """Clone a catalog Git source and adapt SKILL.md-only trees to SKSkills."""
    if not entry.git:
        raise ValueError(f"catalog entry '{name}' has no Git source")
    with tempfile.TemporaryDirectory(prefix="skskills-") as tmp:
        checkout = Path(tmp) / "checkout"
        subprocess.run(["git", "clone", "--depth=1", entry.git, str(checkout)], check=True,
                       capture_output=True, text=True)
        source = checkout / entry.git_path if entry.git_path else checkout
        if not source.is_dir():
            raise FileNotFoundError(f"Git skill path not found: {entry.git_path}")
        if not (source / "skill.yaml").exists():
            skill_md = source / "SKILL.md"
            if not skill_md.exists():
                raise FileNotFoundError(f"Git skill '{name}' has neither skill.yaml nor SKILL.md")
            staged = Path(tmp) / "staged"
            shutil.copytree(source, staged)
            text = skill_md.read_text(encoding="utf-8")
            match = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
            desc = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
            (staged / "skill.yaml").write_text(
                "name: %s\nversion: 0.0.0+git\ndescription: %s\nauthor:\n  name: catalog-source\nknowledge:\n  - path: SKILL.md\n    description: Skill instructions\n    auto_load: true\n" % (
                    (match.group(1).strip() if match else name),
                    (desc.group(1).strip() if desc else "Git-backed catalog skill"),
                ), encoding="utf-8")
            source = staged
        return install_from_local(source, agent=agent, force=force)


def install_from_local(
    path: str | Path,
    agent: str = "global",
    force: bool = True,
) -> InstalledSkill:
    """Install a skill from a local directory containing ``skill.yaml``.

    Args:
        path: Directory holding the skill (must contain ``skill.yaml``).
        agent: Agent namespace ("global" or a specific agent).
        force: Overwrite an existing install (default True — idempotent).

    Returns:
        InstalledSkill metadata.

    Raises:
        FileNotFoundError: If the directory or its skill.yaml is missing.
    """
    source = Path(path).expanduser().resolve()
    registry = SkillRegistry()
    installed = registry.install(source, agent=agent, force=force)
    logger.info("installer: installed %s from %s", installed.manifest.name, source)
    return installed


def install_from_catalog(
    name: str,
    agent: str = "global",
    force: bool = True,
    catalog: Optional[SkillCatalog] = None,
) -> InstalledSkill:
    """Resolve a catalog entry by name and install it from its declared source.

    Resolution order for the entry's coordinates:
      1. ``local``    → install_from_local(<dir>)
      2. ``git_path`` → install_from_local(<repo>/<git_path>)  (bundled skill)
      3. ``pip``      → install_from_pip(<package>)

    If the name is not in the catalog, fall back to ``~/clawd/skills/<name>``
    when that directory carries a skill.yaml (first-party Claude Code skill).

    Raises:
        ValueError: If the skill cannot be resolved to any installable source.
    """
    catalog = catalog or SkillCatalog()
    entry = catalog.get(name)

    if entry is None:
        candidate = _LOCAL_SKILLS_DIR / name
        if (candidate / "skill.yaml").exists():
            return install_from_local(candidate, agent=agent, force=force)
        raise ValueError(
            f"skill '{name}' is not in the catalog and no local "
            f"{candidate}/skill.yaml exists"
        )

    if entry.local:
        return install_from_local(Path(entry.local).expanduser(), agent=agent, force=force)

    if entry.git_path:
        bundled = _REPO_ROOT / entry.git_path
        if bundled.exists():
            return install_from_local(bundled, agent=agent, force=force)
        return _install_git_entry(entry, name, agent, force)

    if entry.git:
        return _install_git_entry(entry, name, agent, force)

    if entry.pip:
        registry = SkillRegistry()
        installed = install_from_pip(entry.pip, registry, agent=agent, force=force)
        logger.info("installer: installed %s from pip %s", name, entry.pip)
        return installed

    # Last resort: a first-party dir matching the catalog name.
    candidate = _LOCAL_SKILLS_DIR / name
    if (candidate / "skill.yaml").exists():
        return install_from_local(candidate, agent=agent, force=force)

    raise ValueError(
        f"catalog entry '{name}' declares no installable source "
        f"(local / git_path / pip)"
    )


__all__ = ["install_from_local", "install_from_catalog"]
