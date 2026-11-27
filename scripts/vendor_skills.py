#!/usr/bin/env python3
"""Vendor third-party Claude Code skills into skills/<name>/ with a skill.yaml.

Upstream skill packs ship bare SKILL.md directories; the skskills registry needs
a skill.yaml beside each one. This copies every skill directory under
<checkout>/<skills_subdir>, writes the manifest, and carries the upstream
LICENSE (and NOTICE, if any) into each directory so it survives installation.

    git clone --depth 1 https://github.com/heygen-com/hyperframes /tmp/hf
    python3 -I scripts/vendor_skills.py /tmp/hf skills \\
        --source heygen-com/hyperframes --version 0.8.149 \\
        --license Apache-2.0 --category media --tags video,hyperframes

Re-running replaces the vendored directories in place (refresh = re-run).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORE = shutil.ignore_patterns("*.test.mjs", "*.test.ts", "__pycache__", ".DS_Store")


def frontmatter(skill_md: Path) -> dict:
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    return yaml.safe_load(text.split("---", 2)[1]) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkout", type=Path, help="upstream git checkout")
    ap.add_argument("skills_subdir", help="path inside the checkout holding skill dirs (or one skill dir)")
    ap.add_argument("--source", required=True, help="upstream owner/repo")
    ap.add_argument("--version", required=True, help="upstream release version")
    ap.add_argument("--license", required=True, help="SPDX license id")
    ap.add_argument("--category", required=True)
    ap.add_argument("--tags", default="", help="comma-separated tags added to every skill")
    ap.add_argument("--requires", default="", help="runtime requirement note for skill.yaml")
    ap.add_argument("--name", help="vendor a single skill dir under this name")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT / "skills")
    args = ap.parse_args()

    commit = subprocess.run(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    root = args.checkout / args.skills_subdir
    if args.name:
        sources = [(args.name, root)]
    else:
        sources = [(d.name, d) for d in sorted(root.iterdir()) if (d / "SKILL.md").is_file()]

    legal = [args.checkout / f for f in ("LICENSE", "NOTICE") if (args.checkout / f).is_file()]
    tags = [t for t in args.tags.split(",") if t]

    for name, src in sources:
        meta = frontmatter(src / "SKILL.md")
        target = args.dest / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target, ignore=IGNORE)
        for f in legal:
            shutil.copy2(f, target / f.name)

        manifest = {
            "name": name,
            "version": args.version,
            "description": " ".join(str(meta.get("description", "")).split()),
            "category": args.category,
            "tags": tags,
            "author": {"name": args.source.split("/")[0]},
            "license": args.license,
            "upstream": {
                "repo": f"https://github.com/{args.source}",
                "path": str(src.relative_to(args.checkout)),
                "commit": commit,
            },
            "invocation": {
                "type": "claude-code-skill",
                "trigger": f"/{meta.get('name', name)}",
                "entrypoint": "SKILL.md",
            },
        }
        if args.requires:
            manifest["requires"] = {"runtime": args.requires}
        header = f"# Vendored from {args.source}@{commit[:12]} by scripts/vendor_skills.py. Do not edit; re-run to refresh.\n"
        (target / "skill.yaml").write_text(
            header + yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        print(f"vendored {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
