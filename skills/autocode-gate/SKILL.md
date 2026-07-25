---
name: autocode-gate
description: Use when grading or interpreting an autocode task's twin gate (independent 1-5 grade, external CI, diff coverage), or when a coding session needs to know what "done" means before claiming completion.
---

# Autocode Gate

The autocode engine (`skharness.autocode`) enforces quality with a twin gate,
not a single LLM opinion. This skill is the reusable knowledge pack for that
gate: what the gate actually checks, how the grading rubric is composed, and
how a gated session (implementer or grader) should read a `GateResult`.

## The twin gate, in full

A task is ratified (passes) only when ALL of the following are true at once:

```
score == 5
AND the grader's notes contain the exact token <promise>COMPLETE</promise>
AND external CI reports green
AND diff coverage meets the configured threshold
```

Nothing in this skill, or in any rubric it carries, can weaken that
predicate. The predicate is code, computed from facts outside the graded
session (`engineering.py`, `ci.py`), byte-pinned by a conformance test in
`skharness` CI. A rubric can make grading stricter (catch more real
problems, produce more 3s and 4s that should not have been 5s); it cannot
make the gate pass on a 4, on red CI, or on coverage below threshold. A
lax rubric costs rounds and review time, not integrity: the deterministic
arms (CI, coverage) hold the line regardless of what an LLM grader believes.

## What the score bands mean

See `rubric.md` for the full per-band criteria a grader applies. In short:

- **5**: every acceptance criterion met, tests exist and pass, CI green,
  the diff contains no scope creep and no unaddressed regression.
- **4**: functionally complete but missing verification rigor (a claim
  without evidence, a test that would have passed before the change).
- **3**: partially complete, or complete but with a real defect a careful
  review would catch.
- **2**: substantial gaps against the acceptance criteria.
- **1**: does not address the task, or breaks something that worked.

Only a 5 can ever ratify, and only in combination with the other three
gate arms above. A 4 is not "close enough."

## How the gate is invoked

The full gate runs as `skharness.autocode.ratify()`, exposed as the MCP
tool `autocode-gate.ratify` (lands in Q2 of the quality-layer rollout; this
skill ships the knowledge pack first, the tool wrapper follows once
`ratify()` exists in `skharness`). There is deliberately no "grade only,
skip CI" tool: exposing the LLM arm alone would create a second, weaker
verdict callers would inevitably treat as "the gate said 5." The one
exception is `advisory: true`, which runs every arm identically but
records nothing (no board score, no journal claim), for "how far am I from
a 5" checks mid-work. Advisory results are always marked `advisory` in
their notes so a transcript can never pass one off as a real ratification.

## Reading a GateResult

```
{"score": int, "passed": bool, "notes": str, "artifact": str | null}
```

- `passed` is the ONLY field that reflects the full twin gate. A `score`
  of 5 with `passed: false` means CI or coverage failed even though the
  grader liked the diff, do not treat that as "basically done."
- `notes` carries the grader's reasoning and, on a genuine 5, the promise
  token. Read the reasoning even on a pass, it is often the fastest way to
  spot a rubric that graded too generously.
- `artifact`, when present, points at the recorded evidence (diff,
  coverage report, CI run) behind the score. Prefer it over re-deriving
  the same facts.

## For a session being graded (implementer role)

You are not the grader. Do not narrate confidence ("this should pass
now"), narrate evidence: what you ran, what it showed. Before ending a
round, self-review against the acceptance criteria the same way a reviewer
would (see the `requesting-code-review` skill), and make sure every claim
in your final report is something you actually verified in that same
message (see `verification-before-completion`). A round that reaches the
grader with unverified claims wastes a round even when the code is
actually fine, the grader cannot see your intent, only your diff and your
report.

## For a session doing the grading (grader role)

You receive only this skill's `rubric.md`, never the implementer's
discipline skills (grader independence: the grader must not absorb the
implementer's framing of its own work). Grade the diff against the
acceptance criteria and the rubric bands above. The deterministic facts
(`ci_status`, `diff_coverage`) arrive in the data block, already computed;
do not re-derive them from the diff, and do not accept "the diff claims
tests pass" as evidence, only the recorded `ci_status` counts.

## Provenance and fallback

This skill's rubric composes with, and never replaces, a code-owned gate
contract (`GATE_CONTRACT` in `skharness.autocode.adapters.base`: the JSON
reply shape and the promise-token protocol). When `quality_skills.enabled`
is false, or this skill cannot be resolved, or its signature/hash check
fails, the engine falls back to today's one-sentence embedded rubric and
records `rubric: embedded` in the journal, the gate never becomes
unavailable because a skill file is missing. See
`docs/superpowers/specs/2026-07-25-autocode-skskills-quality-layer.md`
sections 4 through 6 for the full mechanism (resolver, delivery channels,
signing).
