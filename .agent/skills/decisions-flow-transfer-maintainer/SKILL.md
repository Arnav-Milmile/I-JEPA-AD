---
name: decisions-flow-transfer-maintainer
description: >
  Maintains DECISIONS.md, FLOW.md, and TRANSFER.md for the I-JEPA-AD project after any
  coding task and at every session boundary. Use this whenever a task adds/changes a
  module, fixes a bug, changes an algorithm or library choice, alters results-affecting
  code (metrics, masking, fusion, training), or when a session is starting or ending
  (including a switch to a different tool/IDE). Always consult this before declaring a
  coding task complete in this repo, and always consult it first thing in a new session.
---

# Decisions, Flow & Transfer Maintainer

After completing a coding task in this repository:

1. Check whether the task involved a decision with a reasonable alternative
   (architecture, library, algorithm, schema, test strategy, bugfix approach).
   If yes, append an entry to DECISIONS.md using the template in CLAUDE.md.
   If no, state explicitly that no entry was needed and why.

2. Open FLOW.md and check whether the change affects:
   - the HLD system diagram or module descriptions (§1)
   - any class/function signature or data schema documented in the LLD (§2)
   - the training-time or inference-time numbered flow (§3)
   - a divergence from the original PRD worth noting (§4)
   Update whichever sections are affected. Do not skip this for "small" changes.

3. If this is the ~10th task since the last full audit, do a full top-to-bottom
   re-read of FLOW.md against the current src/ tree and correct any drift.

At the end of a session (or when the user signals they're switching tools/IDEs):

4. Fully rewrite TRANSFER.md from scratch (never append) using the template in
   CLAUDE.md. §3 ("verified working") must only include items backed by a passing
   test or reproduced result — do not upgrade "written this session" to "verified."
   Fill §8 with the concrete next step. Confirm the recorded commit SHA matches HEAD.

At the start of a session:

5. Read TRANSFER.md in full before touching code. If its recorded commit doesn't
   match current HEAD, flag it as stale to the user before proceeding.

Never report a task as complete without having done steps 1-2. Never start new work
in a fresh session without having done step 5.
