# Changelog

## Contents

- [[Unreleased]](#unreleased)
  - [Fixed](#fixed)
  - [Changed](#changed)
  - [Added](#added)
  - [Documented (no code change)](#documented-no-code-change)
  - [Not changed](#not-changed)
- [[0.2.0] - 2026-07-03](#020---2026-07-03)
  - [Added](#added-1)
  - [Fixed](#fixed-1)
- [[Untagged] - 2026-07-24](#untagged---2026-07-24)
  - [Added](#added-2)
- [[Untagged] - 2026-06-13](#untagged---2026-06-13)
  - [Changed](#changed-1)
- [[v1.1.1] - 2026-06-10](#v111---2026-06-10)
  - [Added](#added-3)
  - [Fixed](#fixed-2)
- [[0.1.1] - 2026-04-12](#011---2026-04-12)
  - [Added](#added-4)
  - [Fixed](#fixed-3)
- [[v1.1.0] - 2026-03-21](#v110---2026-03-21)
  - [Added](#added-5)
- [[Untagged] - 2026-03-04](#untagged---2026-03-04)
  - [Added](#added-6)
- [[Untagged] - 2026-02-27](#untagged---2026-02-27)
  - [Added](#added-7)
- [[Untagged] - 2026-02-26](#untagged---2026-02-26)
  - [Added](#added-8)

All notable changes to skskills are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning intent: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Read this before trusting a version number here.** This file was reconstructed
> from git history on 2026-08-14; it did not exist before then, so entries dated
> earlier than that are a best-effort reconstruction from commits and tags, not a
> record written at release time. Two independent drifts make the numbering
> unreliable, and both are open:
>
> 1. **Five files declare a version and they disagree.** `pyproject.toml` and
>    `src/skskills/__init__.py` say 0.2.0; `package.json`, `skill.yaml`, and
>    `SKILL.md` say 0.1.0. The npm artifact and the PyPI artifact are built from the
>    same commit and announce different versions.
> 2. **The git tags do not follow either of them.** Three tags exist on `origin`:
>    `v0.1.1`, `v1.1.0`, and `v1.1.1`, and they were not created in version order.
>    `v1.1.0` points at a 2026-03-21 commit, `v0.1.1` at 2026-04-12, `v1.1.1` at
>    2026-06-10. No `v0.2.0` tag exists despite `pyproject.toml` declaring 0.2.0.
>
> The durable fix is deriving the version from the git tag (setuptools-scm) instead
> of hardcoding it in five places. See
> [SOP.md §5.2](SOP.md#52-version-bump-read-this-before-tagging).

---

## [Unreleased]

### Fixed
- **The discovery walk no longer aborts on one unreadable `SKILL.md`.**
  `plugin/discovery.py` called `read_text()` unguarded on every `SKILL.md` it
  walked, so a single dangling symlink raised `FileNotFoundError` and killed
  discovery across all ten configured roots. Reads now go through
  `_read_skill_md()`, which logs `skipping unreadable SKILL.md <path>: <reason>` at
  WARNING and skips that one skill. `skill.yaml` reads are guarded the same way.
  Card **748c82f9**.
- **`tests/plugin/test_pilot_skcomms.py::test_skcomms_builds_and_validates` is now
  hermetic.** It passes its own `--roots` pinned to this repo's `skills/` dir
  instead of walking the developer's real `~/clawd` and `~/.claude` trees, so it no
  longer depends on host filesystem state and can pass on a bare CI runner. Suite
  goes from 133 passed / 1 failed to 138 passed / 0 failed, verified on Python 3.11
  and 3.12. Card **748c82f9**.
- **CI can fail again.** Removed `|| true` from the pytest step in `ci.yml` and in
  `publish.yml`, and removed `if: always()` from `publish-pypi` and `publish-npm`,
  which together let a fully red suite publish to PyPI and npm. Card **62a5256d**.
  The very first run of the un-guarded gate did its job immediately: it caught a
  version-dependent test that was green on 3.12 and red on 3.10/3.11. `Path.glob`
  yields a dangling symlink on Python 3.12+ but silently drops it on 3.11 and
  earlier, so an assertion that the skip was logged at WARNING could never hold on
  the older legs. The behavioural assertion (the walk continues) is version
  independent and stays in the discovery test; the WARNING is now asserted by
  calling `_read_skill_md` directly. Documented in SOP.md section 4.

### Changed
- `SOP.md` §3, §4, §5.2, §8 and its `docs-evidence` block rewritten to match the
  above. Three checks that pinned the broken state (the `|| true` in `ci.yml`) are
  replaced by four that pin the fixed state and fail if either regression returns.

### Added
- `SOP.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and this
  `CHANGELOG.md`, bringing the repo to SK_REPO_DOC_STANDARD.
- `.github/workflows/docs-check.yml`, running the shared sk-standards docs gate.
  Now at `tiers: "1,2,3"`, so the `docs-evidence` block is executed on every push.
- A `docs-evidence` block at the end of `SOP.md`: 11 repo-local checks that fail
  when a documented entry point, default, root count, CI gate, or known defect
  drifts.

### Documented (no code change)
- **The multi-root skill topology.** `src/skskills/plugin/skill-roots.yaml` declares
  ten skill homes across four runtimes, with `role` breaking name collisions.
  `~/clawd/skskills/skills` is `role: canonical` and holds four skills; the
  `role: incubator` root `~/clawd/skills` holds 83. The convention "author skills in
  the canonical root only" is a migration target, not the present state.
- **The signature gap.** `SkillManifest.signature` / `signed_by` are stored and
  displayed, and `is_signed()` is a non-empty-string check. No PGP verification code
  exists anywhere in `src/`. Recorded in `SECURITY.md` as a gap, not a mitigation.
- **The root `skill.yaml` points at a module that does not exist.** It declares
  `skskills.skill:list_tools`, `:install`, `:catalog`, and `:run_tool`, but
  `src/skskills/skill.py` is absent from `origin/main`. Any consumer resolving those
  dotpaths fails. The working tool surface is the aggregator's.
- ~~**CI cannot fail on a test failure.**~~ Superseded within this same Unreleased
  block: fixed above under card **62a5256d**. The original finding was that `ci.yml`
  ran pytest with `|| true`, `publish.yml` repeated that job, and both publish jobs
  carried `needs: test` plus `if: always()`, so a total test failure still shipped to
  PyPI and npm. It was left in place by the docs pass only because the suite was red.
- ~~**The failing test is not hermetic.**~~ Superseded within this same Unreleased
  block: fixed above under card **748c82f9**. The original finding was that
  `tests/plugin/test_pilot_skcomms.py::test_skcomms_builds_and_validates` invoked
  `skskills plugin build` with no `--roots` override, so discovery walked the
  developer's real skill homes, and `plugin/discovery.py` read each `SKILL.md`
  unguarded so a dangling symlink aborted the whole walk.

### Not changed
- The license stays **GPL-3.0-or-later**. `LICENSE` is verbatim GPLv3 and both
  `pyproject.toml` and `package.json` already declare it.

---

## [0.2.0] - 2026-07-03

No `v0.2.0` tag was ever pushed. This is the version declared in `pyproject.toml` and
`src/skskills/__init__.py`.

### Added
- **The `skskills plugin` compiler**, a new command group that compiles discovered
  skills into the Anthropic plugin / marketplace envelope:
  - `plugin/discovery.py`: discovery across every configured skill home, with dedup
    and role-ranked precedence.
  - `plugin/grouping.py`: plugin views over skills, driven by the top-level
    `plugins:` list in `catalog.yaml`.
  - `plugin/emit.py`: emits `.claude-plugin/plugin.json`, `.mcp.json`,
    `CONNECTORS.md`, and `marketplace.json`.
  - `plugin/scrub.py`: an outbound safety gate scanning emitted artifacts for
    secrets and private endpoints, with `publish` additionally requiring
    `publish: true` in the catalog and an explicit `--i-am-chef` flag.
  - `plugin/cli.py`: `discover`, `build`, `validate`, `publish`, `add`.
- `plugin add` wires through to `claude plugin marketplace add` and
  `skcapstone register`, best-effort and never fatal.
- The skcomms pilot plugin definition and an end-to-end golden test.
- `docs/DEPENDENTS.md`, mapping who consumes skskills and how tightly.

### Fixed
- Scrub coverage extended to the `172.16.0.0/12` private range.
- Restored the `src/skskills/catalog.yaml` symlink after a merge dropped it.
- Removed a dead import flagged in final review.

---

## [Untagged] - 2026-07-24

### Added
- `skills/bot-roundtrip-canary`: a bot round-trip canary skill (coordination card
  08a1e578).

## [Untagged] - 2026-06-13

### Changed
- Renamed all `skcomm` references to `skcomms`.

## [v1.1.1] - 2026-06-10

Tagged out of version order; see the note at the top of this file.

### Added
- `skills/chat` and `skills/who`: Claude Code skills that front the skchat MCP
  server.
- `README.md` and `docs/ARCHITECTURE.md` for the skills platform.

### Fixed
- Corrected the default install path and added a local-skill source.

## [0.1.1] - 2026-04-12

Tagged `v0.1.1`.

### Added
- `skills/unhinged-mode` and its catalog entry.
- The `itil-ops` skill example.

### Fixed
- Corrected the `unhinged-mode` `skill.yaml` author format.
- Tracked the `l1b3rt4s` and `obliteratus` library directories as regular files.

## [v1.1.0] - 2026-03-21

Tagged out of version order; see the note at the top of this file.

### Added
- The `cognitive-gear` skill: SPRINT / CRUISE / DEEP DIVE operational modes.

## [Untagged] - 2026-03-04

### Added
- `package.json` and the npm publish workflow, making `@smilintux/skskills` a second
  published artifact alongside the PyPI package.
- The root `skill.yaml` manifest and the publish workflows.
- The pip-install plus catalog architecture, replacing git submodules as the
  delivery mechanism.

## [Untagged] - 2026-02-27

### Added
- The remote registry client: `publish`, `pull`, and `clone` against a community hub
  at `skills.smilintux.org/api`, with SHA-256 tarball integrity checking.
- `SKILL.md` with the CLI reference and the skill-primitives documentation.

## [Untagged] - 2026-02-26

### Added
- Initial skskills framework: the `skill.yaml` manifest model, the local registry
  with global and per-agent namespaces, the entrypoint loader, and the stdio MCP
  aggregator. Positioned as the OpenClaw replacement.
