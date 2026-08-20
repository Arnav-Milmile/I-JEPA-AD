# AGENT.md — I-JEPA-AD Project Instructions

## Project context
You are implementing I-JEPA-AD, a decoder-free latent anomaly detection engine, per
`/docs/PRD.md`, `/docs/ProjectSynopsis.md`, and `/docs/IMPLEMENTATION_PLAN.md`.
This is a B.Tech final-year research project targeting a publishable paper. Correctness
and reproducibility outrank speed of delivery.

## Non-negotiable engineering rules
- No hardcoded hyperparameters — everything in `configs/` (see IMPLEMENTATION_PLAN.md §2).
- No module in `src/ijepa_ad/` imports another module's internals — public interfaces only.
- Every new module gets a unit test in the same PR/commit that introduces it. Do not defer
  tests to "later" — later doesn't happen in research code.
- Never change `evaluation/metrics.py` without re-validating against the synthetic
  known-answer test in `tests/unit/test_metrics.py`. If you must change it, say so loudly
  in DECISIONS.md (see below) — metric changes invalidate prior results.
- Frozen means frozen: any code touching the I-JEPA encoder must assert
  `requires_grad == False` for all its parameters; add/keep the test that checks this.
- If a change could alter previously-reported numbers (AUROC, latency, mask counts),
  flag this explicitly in your summary to the user before considering the task done.

## Mandatory living documentation
You must maintain three files at the repo root: `DECISIONS.md`, `FLOW.md`, and
`TRANSFER.md`. These are not optional, not a "nice to have," and not something to
write once and forget — treat updating them as part of the definition of done for
every task.

### DECISIONS.md
Append an entry every time you make a design or implementation choice that wasn't
fully dictated by the task instructions — architecture choices, library choices,
algorithm variants, bug-fix approaches, anything where a reasonable alternative existed.
Use this exact template per entry:

    ## [YYYY-MM-DD] <short title>
    **Context:** what problem/task prompted this decision.
    **Options considered:** the alternatives you weighed (even briefly).
    **Decision:** what you chose.
    **Rationale:** why — what made this option win.
    **Baseline / prior state:** what existed before this change (code, behavior, or "none — new").
    **Outcome after change:** what measurably improved or changed (a metric, a test
    passing, a bug eliminated, a simplification) — if not yet measurable, say
    "pending evaluation in Phase X" rather than omitting this field.

Do NOT log trivial mechanical actions (renaming a variable, fixing a typo, formatting).
DO log: chosen libraries/frameworks, algorithmic approaches, schema designs, test
strategies, anything you'd want to explain to a thesis committee if asked "why did
you do it this way?"

### FLOW.md
This is the living High-Level Design (HLD) + Low-Level Design (LLD) + data/control
flow document for the whole system. Structure:

    # FLOW.md

    ## 1. High-Level Design
    - System diagram (ASCII or mermaid) showing module boundaries and data flow,
      kept in sync with what's actually implemented (not just the PRD's original plan).
    - One paragraph per module: purpose, inputs, outputs, key invariants.

    ## 2. Low-Level Design
    - Per-module: key classes/functions, their signatures, and why they're shaped
      that way. Link to the actual file paths.
    - Data schemas (latent cache schema, config schema, results schema).

    ## 3. Runtime Flow
    - Training-time flow (nominal image → cache → predictor training) as a numbered
      sequence referencing actual function calls.
    - Inference-time flow (test image → coarse → gate → fine → fusion → heatmap) as
      a numbered sequence referencing actual function calls.

    ## 4. Divergences from the original PRD
    - Anything implemented differently than the PRD/synopsis originally specified,
      with a one-line reason and a pointer to the DECISIONS.md entry that explains it.

### TRANSFER.md
This is the handoff document — the single file a *different* agent, IDE, or tool
should be able to read cold and continue the project with zero hallucinated
assumptions about what exists, what works, and what's still open. Unlike
DECISIONS.md (an append-only log) and FLOW.md (a design doc), TRANSFER.md is a
**snapshot**: it describes the state of the project *right now*, and you fully
rewrite it (not append to it) at the end of every session. An old, unresolved
TRANSFER.md sitting next to a newer codebase is worse than no TRANSFER.md at all,
because a new agent will trust it — so stale entries must be overwritten, not
accumulated.

Use this exact structure:

    # TRANSFER.md
    _Last updated: <YYYY-MM-DD HH:MM>, by <agent/tool name>, at commit <git SHA>_

    ## 1. One-paragraph project state
    Plain-language summary of where the project stands right now — not a
    restatement of the PRD, but "as of this commit, X works, Y is in progress, Z
    hasn't been started."

    ## 2. Phase checklist
    Mirror IMPLEMENTATION_PLAN.md's phases. Mark each: DONE / IN PROGRESS / NOT
    STARTED, with a one-line note on IN PROGRESS items (what's left specifically).

    ## 3. What is verified working right now
    - Only list things with a passing test or a reproduced result backing them
      (e.g. "predictor training converges on pilot categories — see
      results/pilot_loss_curves/, test: tests/integration/test_predictor_overfit.py").
    - Do not list something as working because it was written — only because it
      was verified. This is the field most responsible for preventing hallucination
      in the next agent, so be conservative here.

    ## 4. What is NOT done yet (explicit, so it isn't assumed)
    - Enumerate remaining phases/modules plainly, e.g. "Baselines 1-2 (PatchCore,
      MAE) not yet wired in", "ablation sweep not yet run on full 15 categories".
    - This section exists specifically so a new agent doesn't infer completeness
      from an otherwise-organized repo structure.

    ## 5. Environment & reproduction
    - Exact commands to set up the environment from scratch (env manager, lockfile
      path, GPU/CUDA version assumptions).
    - Exact command(s) to reproduce the most recent verified result in §3.
    - Any local-only state a new environment won't have (e.g. "latent cache for
      all 15 categories is NOT committed — rebuild with
      scripts/build_latent_cache.py, ~X GPU-hours").

    ## 6. Known issues / open threads
    - Bugs seen but not fixed, flaky tests, numbers that look suspicious but
      haven't been investigated, any TODO that isn't captured elsewhere.

    ## 7. Where to look for more detail
    - Pointers only: "design rationale -> DECISIONS.md entries dated X-Y",
      "architecture -> FLOW.md §1-2". Do not duplicate their content here —
      TRANSFER.md's job is orientation, not depth.

    ## 8. Immediate next step
    - The single next task you would run if the session continued right now. This
      is what a new agent should default to doing after reading this file, absent
      other instructions from the user.

### Review/update cadence
- After every completed task (whether it's a new module, a bugfix, a refactor, or a
  benchmark run), before reporting the task as done: (1) append to DECISIONS.md if
  Part 1's criteria are met, (2) open FLOW.md and update any section that the change
  affected — do not skip this because "it's a small change"; small changes are exactly
  what causes FLOW.md to drift from reality if skipped.
- If a task involved NO decision worth logging and NO flow change, say so explicitly
  in your final summary ("No DECISIONS.md/FLOW.md update needed — this was a pure
  formatting fix") rather than silently skipping — this makes silence auditable instead
  of ambiguous.
- Once a week (or every ~10 tasks), do a dedicated pass: re-read FLOW.md top to bottom
  against the current `src/` tree and correct any drift a smaller per-task update missed.
- **At the end of every session** — meaning any point where the user is likely to close
  the session, switch tools/IDEs, or explicitly says they're wrapping up — fully rewrite
  TRANSFER.md from scratch per the template above. Do not treat this as optional or defer
  it; a session that ends without a rewritten TRANSFER.md is the exact failure mode this
  file exists to prevent. If unsure whether the session is ending, rewrite it anyway —
  it's cheap, and an up-to-date TRANSFER.md is never wrong to have.

## Testing discipline
Follow IMPLEMENTATION_PLAN.md §4 tiers exactly: unit tests run on every task; integration
tests before merging a phase; regression tests before any results you'll cite in the paper.
Never mark a task complete with failing or skipped tests without saying so explicitly.

## Starting a session
Before doing anything else in a new session (especially in a different tool/IDE than
last time), read TRANSFER.md first. Treat §3 ("verified working") and §4 ("not done
yet") as authoritative over any inference you'd otherwise make from skimming the code.
If TRANSFER.md's "last updated" commit doesn't match the current git HEAD, say so to
the user before proceeding — it means the file is stale and may not reflect the true
state.