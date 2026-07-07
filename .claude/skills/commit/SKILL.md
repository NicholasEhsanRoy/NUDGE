---
name: commit
description: Use whenever creating a git commit in this repository. Ensures every commit message explicitly credits Claude as a co-author via a Co-Authored-By trailer — this project is an experiment in Claude-assisted development, so Claude's involvement must be visible and auditable in the git history.
---

# commit

NUDGE is deliberately built with heavy Claude involvement, and the git history
should make that involvement **obvious and auditable**. Follow this whenever you
create a commit in this repo.

## Required

1. **Always append a `Co-Authored-By` trailer for Claude** as the final line of
   the commit message:

   ```
   Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
   ```

   Name the exact model that did the work (e.g. `Claude Opus 4.8`). If a
   different Claude model produced the change, credit that one instead.

2. **Write a real body for anything non-trivial**, and in it say briefly *what
   Claude actually did* (analysis, drafting, research, code) so the credit is
   specific and honest rather than boilerplate.

## Conventions

- Subject line: imperative mood, ≤ ~72 chars, no trailing period.
- Blank line between subject and body; wrap the body at ~72 chars.
- **Never fabricate co-authorship.** Only credit Claude for work Claude did. If a
  commit was written entirely by the human with no Claude involvement, the
  trailer is not required — honesty over ritual.
- Only commit (and only push) when the user has asked you to.

## Example

```
Add synthetic Perturb-seq generator with ground-truth labels

Implements generate_synthetic_perturbseq(...) -> AnnData with a 0-3 realism
dial and ground truth stashed in .uns, plus the first three decoy-battery
cases. Claude designed the generator API and wrote the decoy specs; the human
reviewed the noise model against the brief.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
