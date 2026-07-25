# Autocode Grading Rubric

The knowledge pack an independent grader composes with the code-owned gate
contract (JSON shape, promise-token protocol; see `SKILL.md`). This file is
prose judgment criteria, not protocol, it can be revised, reviewed, and
versioned independently of the engine that consumes it. Version this
rubric's substantive changes; the compiled skill carries its own
`skill.yaml` version and provenance hash.

## Inputs available to the grader

The data block (never the instruction block) carries: `task_id`, the
diff, the acceptance criteria list, `ci_status`, and `diff_coverage`. Grade
from these. Do not ask for, infer, or trust anything the diff itself
claims about its own test results, `ci_status` is the fact, the diff's
prose is not.

## Score bands

### 5, ratify-eligible

All of the following, with no exceptions:

- Every acceptance criterion is met, verifiably, by the diff.
- The diff includes a test that would have failed before the change (a
  TDD tell: look for a test file touched in the same commit range as the
  behavior it covers, and check whether the assertion would plausibly
  have passed against the prior code).
- No scope creep: nothing implemented that was not asked for, unless the
  acceptance criteria explicitly allow discretion.
- No regression: existing behavior the acceptance criteria depend on
  still works after the change.
- Conventions match the surrounding codebase (naming, error handling
  style, existing test patterns).
- `ci_status` is green and `diff_coverage` meets the configured threshold.
  A grader NEVER assigns 5 when either deterministic fact is failing,
  even if the code otherwise reads as excellent; note the mismatch and
  score honestly (a 3 or 4), the twin gate's CI/coverage arms will block
  ratification regardless of the score, but an honest score keeps the
  round-count and journal trail meaningful.

### 4, functionally complete, not verification-clean

The behavior is correct and every acceptance criterion is addressed, but
one of these holds:

- A completion claim in the diff's commit message or the task report is
  not backed by evidence in `ci_status`/`diff_coverage` (see
  `verification-before-completion`'s evidence table: "should pass" is not
  evidence, a fresh run's output is).
- Tests exist but do not demonstrably fail before the fix (no visible
  red-green cycle, so the test's power to catch a regression is
  unverified).
- A minor deviation from repo convention that does not affect behavior.

### 3, partial or defective

- Some acceptance criteria are met, others are not, or are met only
  incidentally.
- A real defect a careful review would catch: an unhandled edge case
  named in the acceptance criteria, an error path that silently swallows
  a failure, a test that asserts nothing meaningful (asserts a mock was
  called rather than asserting real behavior).
- Verbatim duplication of a logic block that should have been shared.

### 2, substantial gaps

- Multiple acceptance criteria unmet.
- The diff addresses a different problem than the one described, or
  solves a narrower version of it without flagging the gap.
- Tests are missing for the changed behavior entirely.

### 1, does not address the task, or actively breaks something

- The diff does not implement the requested behavior.
- The diff breaks previously-working behavior the acceptance criteria (or
  the existing test suite) depend on.
- No tests, and the change is non-trivial.

## The TDD tells a grader checks

Adapted from `test-driven-development`'s discipline, recast for someone
reading a diff rather than watching a live session:

- Is there a test touched in the same change whose assertion targets the
  new/changed behavior specifically (not a broad smoke test)?
- Would that assertion plausibly have failed against the code as it stood
  before this diff? If the test could have passed against the old code
  too, it is not proof the change did anything.
- Is the test testing real behavior, or a mock's call count? A test that
  only asserts a mock was invoked N times proves the mock was invoked, not
  that the feature works.

## The verification evidence table a grader checks

Adapted from `verification-before-completion`, recast as what counts as
evidence in the data block rather than in the grader's own message:

| Claim in the diff/report | Evidence that supports it | Not evidence |
|---|---|---|
| Tests pass | `ci_status: green` with the run's failure count at 0 | The diff's commit message saying "tests pass" |
| Bug fixed | A test reproducing the original symptom, now passing | "Code changed, should be fixed" |
| Build succeeds | A CI stage that runs the build, green | Linter passing alone |
| Requirements met | Each acceptance criterion traceable to a specific diff hunk | "All good" with no line-by-line mapping |

## The review checklist a grader applies

Adapted from `requesting-code-review`, applied by the grader instead of a
human reviewer:

- Does the implementation match the acceptance criteria, line by line?
- Are any deviations from the criteria justified improvements, or
  problematic departures the task did not ask for?
- Is the diff scoped to the task, or does it touch unrelated code?
- Are edge cases named in the acceptance criteria actually handled?
- Would a maintainer reading this diff cold understand why each hunk
  exists?

## What a grader must never do

- Never invent an acceptance criterion that was not in the task's list.
- Never treat "the implementer says it is done" as evidence for anything.
- Never emit the promise token (`<promise>COMPLETE</promise>`) except on a
  genuine, fully-justified 5. The token is parsed deterministically by the
  engine and ANDed with CI and coverage, an accidental or generous token
  on a lesser score does not ratify anything by itself, but it pollutes
  the journal's record of what the grader actually believed.
- Never grade based on the diff's own narration of its test results;
  grade based on `ci_status` and `diff_coverage` in the data block.
