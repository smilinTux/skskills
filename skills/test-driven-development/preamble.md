# Test-Driven Development (distillate)

**Iron law:** no production code without a failing test first. Write code
before the test? Delete it, start over. Don't keep it "as reference."

**Cycle:** RED (write one minimal failing test) -> verify it actually fails,
for the right reason -> GREEN (simplest code to pass) -> verify it passes,
all tests pass, output pristine -> REFACTOR (clean up, stay green) -> repeat.

**Why order matters:** tests written after code pass immediately, which
proves nothing. Test-first forces you to watch it fail, proving it tests
something real.

**Top red flags, stop and restart with TDD if you catch yourself:**
- Code before test, or "I'll test after"
- Test passes immediately on first run
- "Already manually tested it" (ad hoc, not systematic, no record)
- "Deleting X hours of work is wasteful" (sunk cost; unverified code is debt)
- "This is different because..." (it isn't)

**Verification checklist before marking work complete:** every new
function has a test; you watched each test fail first, for the expected
reason; minimal code made it pass; all tests pass; output is pristine (no
errors, no warnings); edge cases are covered.

Full discipline, rationalization table, and worked example: see SKILL.md.
