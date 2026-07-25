# Subagent-Driven Development (distillate)

**Core principle:** fresh subagent per task, task review (spec compliance
+ code quality) after each, broad whole-branch review at the end. Never
inherit your own session's context or history into a dispatch, construct
exactly what the subagent needs.

**Per-task loop:** dispatch implementer with its task brief -> implementer
implements, tests, commits, self-reviews -> generate a diff review
package -> dispatch task reviewer -> if findings, dispatch a fix, then
re-review -> mark complete in the progress ledger only once clean.

**Model selection:** use the least powerful model that can do the job;
mechanical, well-specified tasks take the cheap tier; integration and
judgment take a standard tier; architecture, design, and the final
whole-branch review take the most capable tier. Always specify the model
explicitly, an omitted model silently inherits the expensive default.

**Durable progress:** conversation memory does not survive compaction.
Track completed tasks in a ledger file, not only in todos; after
compaction, trust the ledger and git log over your own recollection.

**Never:** skip task review or accept a report missing either verdict;
proceed with unfixed issues; dispatch multiple implementers in parallel
(conflicts); pre-judge a reviewer's findings in the dispatch prompt.

Full loop, templates, and file-handoff conventions: see SKILL.md.
