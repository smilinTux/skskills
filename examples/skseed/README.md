# skseed — SKSkills Reference

**skseed** is the canonical example of a *pip-installable sovereign skill*.
It ships its own `skill.yaml` bundled inside the Python package, so no git
submodule or manual copy is needed.

## Install

```bash
# One command — pip installs the package and registers the skill
skskills catalog install skseed

# Or explicitly
skskills pip-install skseed

# Browse catalog first
skskills catalog list
skskills catalog info skseed
```

## What it provides

14 MCP tools across 4 subsystems:

| Subsystem | Tools |
|-----------|-------|
| **Collider** | `collide`, `batch_collide`, `cross_reference`, `verify_soul`, `truth_score_memory`, `audit_beliefs` |
| **Audit** | `audit` |
| **Philosopher** | `philosopher`, `continue_session`, `collide_insight`, `session_summary` |
| **Alignment** | `truth_check`, `alignment_report`, `coherence_trend` |

2 lifecycle hooks: `on_memory_stored`, `on_boot`

## Why this pattern (not git submodules)

| Approach | Verdict |
|----------|---------|
| Git submodules | ❌ Two-step clone, often stale, confusing for contributors |
| Copy into examples/ | ❌ Gets out of sync immediately |
| npm/pip workspace | ⚠️ Adds monorepo complexity |
| **Independent pip package + catalog** | ✅ Each skill owns its repo + release cycle |

Skills are Python packages. Their `skill.yaml` lives at
`<package>/data/skill.yaml`. `skskills pip-install` finds it automatically
via `importlib.util.find_spec`.

## Layout inside skseed

```
skseed/                       ← GitHub repo root
├── skill.yaml                ← developer-facing copy (kept in sync)
├── pyproject.toml            ← ships as PyPI package `skseed`
├── package.json              ← ships as npm package @smilintux/skseed
├── skseed/                   ← Python package
│   ├── data/
│   │   ├── seed.json         ← Neuresthetics framework AST
│   │   └── skill.yaml        ← bundled manifest (pip discoverable)
│   ├── skill.py              ← all 14 entrypoints (dict-in / dict-out)
│   ├── collider.py           ← 6-stage steel man engine
│   ├── alignment.py          ← three-way belief store
│   ├── audit.py              ← memory logic auditor
│   ├── philosopher.py        ← interactive brainstorm sessions
│   └── hooks.py              ← on_memory_stored + on_boot
└── src/index.ts              ← TypeScript types for @smilintux/skseed
```

## Writing your own pip-installable skill

1. Create a repo: `gh repo create smilinTux/my-skill --public`
2. Add a Python package with `data/skill.yaml` inside it
3. Include `data/*.yaml` in `pyproject.toml` package-data
4. Publish to PyPI
5. Add to `skskills/catalog.yaml`

```toml
# pyproject.toml
[tool.setuptools.package-data]
my_skill = ["data/*.yaml"]
```

```yaml
# catalog.yaml entry
- name: my-skill
  pip: my-skill
  git: https://github.com/you/my-skill
  description: "What it does"
  tags: [your, tags]
  category: community
```

Then anyone can install it:

```bash
skskills catalog install my-skill
```
