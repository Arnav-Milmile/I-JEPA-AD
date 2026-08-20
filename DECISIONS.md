# DECISIONS.md
_Append-only log of design/implementation decisions for I-JEPA-AD._
_Entry format and rules for what belongs here: see CLAUDE.md § Mandatory living documentation._
_Do not edit or remove past entries — if a decision is later reversed, add a new entry that supersedes it and links back._

---

## [2026-08-20] Establish initial repository toolchain
**Context:** §1 and §4.4 of the implementation plan require a reproducible Python
package layout, automated quality checks, and CPU-only tests before modeling work.
**Options considered:** Keep dependencies implicit; use a requirements-only layout;
or define package and development tooling in `pyproject.toml` with a pinned tool
lockfile.
**Decision:** Use a `src/` package layout, `pyproject.toml` for setuptools plus
Ruff, Black, mypy, and pytest configuration, `requirements-lock.txt` for exact
development-tool versions, and GitHub Actions/pre-commit for the same checks.
**Rationale:** This keeps local and CI commands aligned while leaving runtime and
modeling dependencies empty until their implementation phases.
**Baseline / prior state:** Documentation-only repository with no Python package or
automated checks.
**Outcome after change:** Empty package scaffold and smoke test are present; local
tooling verification is recorded in `TRANSFER.md` after the checks run.

## [2026-08-20] Centralize deterministic seeding and observability
**Context:** Weeks 1–2 require reproducibility controls and structured resource
logging before data and model modules are added.
**Options considered:** Let each future script seed its own libraries; configure
logging ad hoc per script; or expose shared utility entrypoints for seeding,
structured logs, and CUDA memory measurement.
**Decision:** Use one `set_seed(seed)` entrypoint for Python, NumPy, PyTorch, and
CUDA determinism; `setup_logging()` for idempotent JSONL logging; and
`profile_peak_vram()` around measurable code blocks using
`torch.cuda.max_memory_allocated()`.
**Rationale:** Centralized controls prevent inconsistent experiment state and make
peak VRAM and run events observable without coupling future modules together. Lazy
PyTorch loading keeps CPU-only configuration and test environments importable.
**Baseline / prior state:** Utility package contained only an empty initializer.
**Outcome after change:** All three utilities have unit tests; CPU test environments
report zero CUDA memory safely, while CUDA environments use PyTorch's peak allocator
counter. Further integration with training and inference is pending Phase 1+.
