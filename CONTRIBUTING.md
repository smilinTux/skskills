# Contributing to skskills

## Contents

- [Before you start](#before-you-start)
- [Development setup](#development-setup)
- [Branch model](#branch-model)
- [The test gate](#the-test-gate)
  - [CI is not a gate here](#ci-is-not-a-gate-here)
- [Changing docs](#changing-docs)
- [Commit convention](#commit-convention)
- [Writing style (hard rule)](#writing-style-hard-rule)
- [Review path](#review-path)
- [Things that need extra care](#things-that-need-extra-care)
- [Contributing a skill](#contributing-a-skill)
- [Code of conduct](#code-of-conduct)

skskills is the capability-delivery layer for sovereign agents. A change here
propagates to every agent that installs a skill, so the bar is: **prove it, then ship
it.** CI will not prove it for you, for reasons documented below.

---

## Before you start

Read [SOP.md](SOP.md). Two things in particular will save you an afternoon:

1. **[§6.3 Skill discovery roots](SOP.md#63-skill-discovery-roots-the-real-topology).**
   There are ten skill homes, not one. `~/clawd/skskills/skills` is `role: canonical`
   but holds four skills; `~/clawd/skills` is `role: incubator` and holds 83. Any
   change to discovery, grouping, or precedence has to be reasoned about across all
   ten.
2. **[§4 Test](SOP.md#4-test).** The CI test job runs with `|| true` and cannot fail.
   Your local run is the only real gate.

## Development setup

```bash
git clone https://github.com/smilinTux/skskills
cd skskills
python -m pip install -e ".[dev]"
python -m pytest tests/ -q
ruff check src/
```

Python 3.10 or later. The fleet installs SK packages into the `~/.skenv` venv.

## Branch model

- `main` is the integration branch and the branch releases are cut from.
- Never commit to `main` directly. Branch first: `feat/...`, `fix/...`, `docs/...`,
  `ci/...`, `chore/...`.
- **Sessions share checkouts on this fleet.** Do not run `git checkout -b` inside
  `~/clawd/skskills`; use a worktree under `~/skworld-worktrees/<purpose>-skskills`.
- One logical change per PR. A refactor and a behaviour change in the same diff is two
  PRs.

## The test gate

```bash
python -m pytest tests/ -q      # this is the gate
ruff check src/                 # this one CI does enforce, pinned to ruff 0.15.4
```

**Read the counts, not the exit code.** As of 2026-08-14 the suite is
**133 passed, 1 failed**: `tests/plugin/test_pilot_skcomms.py::test_skcomms_builds_and_validates`
fails because it invokes `skskills plugin build` with no `--roots` override, so
discovery walks the developer's real skill homes and dies on a dangling `SKILL.md`
symlink. Your PR must not make that number worse. If you fix it, say so in the PR and
in `CHANGELOG.md`.

New tests must be **hermetic**: use `tmp_path`, and pass an explicit `--roots` or
`--catalog` override to anything under `skskills.plugin`. A test that reads the
developer's home directory is the bug the pilot test just demonstrated.

### CI is not a gate here

`.github/workflows/ci.yml` runs `pytest ... || true`, and `publish.yml` publishes to
PyPI and npm with `needs: test` plus `if: always()`. A green check on this repo means
`ruff` and `gitleaks` passed and nothing more. Tracked as card **62a5256d**. Do not
remove `|| true` in a PR unless the suite genuinely passes first, and say in the PR
body that you confirmed it.

## Changing docs

`SOP.md` ends with a `docs-evidence` block that `sk-standards/scripts/docs_check.py`
executes. Every check must be repo-local, network-free, and cheap, and must exit
non-zero the moment the documented fact stops being true.

If your change moves an entry point, a config path, a default, or a root count,
**update the SOP and its evidence block in the same PR**. A doc that is confident and
wrong is worse than no doc, because it gets trusted.

Any PR touching `src/**` or `pyproject.toml` must also touch `CHANGELOG.md`.

## Commit convention

Conventional-commit prefixes: `feat`, `fix`, `docs`, `test`, `ci`, `chore`, `refactor`.
Scope with the module where it helps: `fix(loader): ...`, `docs(sop): ...`.

Every commit ends with the co-author trailer:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

## Writing style (hard rule)

**No em dashes and no en dashes anywhere.** Not in docs, not in code comments, not in
commit messages, not in PR bodies. Use a comma, parentheses, a colon, or start a new
sentence. Regular hyphens are fine, including as a connector and in ranges (5-10,
Mon to Fri).

## Review path

1. Open a PR against `main`. Never push to `main`.
2. **Never push a tag.** A `v*` tag publishes to PyPI and npm immediately, and neither
   registry lets you re-upload a version. Tagging is a maintainer action taken after
   merge, following [SOP.md §5.2](SOP.md#52-version-bump-read-this-before-tagging).
3. The PR body states: what changed, how you verified it (paste the pytest counts),
   and anything you could not verify.
4. A maintainer reviews. Security-sensitive changes (the loader, `remote.py`,
   `plugin/scrub.py`, anything touching the signature fields) get a second pass.

## Things that need extra care

| Area | Why |
|---|---|
| `loader.py` entrypoint resolution | This is where third-party code becomes a running callable. Any loosening is a security change. |
| `plugin/scrub.py` | The outbound gate that keeps tokens and private endpoints out of published artifacts. Add patterns freely; **never weaken one to make a publish succeed.** Fix the artifact. |
| `plugin/discovery.py` | Walks ten roots across a live filesystem. Guard every read; one unreadable file currently aborts the whole walk. |
| The `signature` / `signed_by` fields | They are stored and displayed but **never verified** ([SECURITY.md](SECURITY.md)). Do not add a UI or a policy decision that treats them as trust until real verification exists. |
| Version numbers | Five files declare a version and they disagree. Do not add a sixth. |

## Contributing a skill

A skill does not have to live in this repo. `skskills install ./my-skill` takes a
local path, `skskills clone <git-url>` takes a repo, and a skill can ship inside a pip
package. Only add a skill to `skills/` here if it is genuinely part of the framework's
own surface.

`skskills init <name>` scaffolds the correct layout. Schema reference:
[SOP.md §7.3](SOP.md#73-manifest-schema).

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
