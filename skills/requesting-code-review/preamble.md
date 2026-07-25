# Requesting Code Review (distillate)

**Core principle:** review early, review often. Before ending a round or
claiming a task complete, self-review the diff against the acceptance
criteria line by line, the same lens an external reviewer would use.

**Mandatory review points:** after each task in a multi-task run, after
completing a major feature, before merge.

**What a reviewer (or your own self-review) checks:** does the change
match the plan or acceptance criteria; are deviations justified or
problematic; is all planned functionality present; is error handling
proper; are edge cases covered; do tests exercise real behavior, not mocks;
does everything integrate cleanly with the surrounding code.

**Severity discipline:** not everything is Critical. Fix Critical issues
immediately, fix Important issues before proceeding, note Minor issues for
later. Acknowledge what was done well, accurate praise makes the rest of
the feedback land.

**Red flags:** skipping review because "it's simple," ignoring Critical
findings, proceeding with unfixed Important findings, arguing with valid
technical feedback instead of fixing it.

Full reviewer prompt template: see code-reviewer.md. Full discipline: see
SKILL.md.
