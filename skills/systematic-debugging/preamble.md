# Systematic Debugging (distillate)

**Core principle:** understand the actual root cause before proposing a
fix. A fix for a cause you have not confirmed is a guess, not a fix.

**When a prior round came back with CI red or a failed grade:** do not
immediately patch the reported symptom. Reproduce the failure first, trace
it back through the call chain to where the wrong state or wrong value
actually originates, then write a failing test that reproduces the root
cause (not just the symptom) before changing implementation code.

**Watch for:** flaky failures rooted in timing assumptions rather than
real races (condition-based waiting, not sleep-based); a fix that only
patches one call site when the same wrong assumption exists elsewhere
(defense in depth, fix the class of bug, not the instance); a test suite
failure that only reproduces in a certain run order (a polluting test
leaking state, not the test you are looking at).

**Red flag:** proposing a fix before you can state, in one sentence, why
the bug happens. If you cannot state the cause, you are not ready to fix
it yet, keep tracing.

Full discipline and worked traces: see SKILL.md, root-cause-tracing.md,
defense-in-depth.md, condition-based-waiting.md.
