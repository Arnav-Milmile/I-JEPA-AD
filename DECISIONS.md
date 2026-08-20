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

## [2026-08-20] Keep official MVTec test data evaluation-only
**Context:** MVTec AD ingestion needs strict train/test boundaries and a future
nominal validation strategy without leaking defect labels into model selection.
**Options considered:** Randomly split all images including test defects; use the
official test split for validation; or hold out validation samples from nominal
training images and reserve the official test split for final evaluation.
**Decision:** Discover training only from `train/good`, discover mixed labels from
the official `test` tree, require each defect image's matching
`ground_truth/<label>/<stem>_mask.*`, and reserve future validation as a nominal-only
holdout from training. Detect cross-split leakage using SHA-256 content IDs.
**Rationale:** This follows the MVTec AD protocol, keeps defect labels out of
hyperparameter selection, and catches copied images even when filenames are reused
between official directories.
**Baseline / prior state:** No data discovery or split validation existed.
**Outcome after change:** Synthetic fixtures verify nominal-only training, 2:2
nominal/defect test counts, mask pairing, non-nominal directory rejection, and
cross-split duplicate-content rejection. Future validation split implementation is
pending the predictor phase.

## [2026-08-20] Adopt Meta official I-JEPA ViT-g/16 checkpoint
**Context:** The PRD requested an official ViT-B/16 checkpoint, but Meta's official
I-JEPA release provides no pretrained ViT-B/16 weights. The user authorized the
official ViT-g/16 alternative.
**Options considered:** Train or source a non-official ViT-B/16 checkpoint; use
Meta's official ViT-g/16 checkpoint; or stop until an official ViT-B/16 release
exists.
**Decision:** Use Meta's official ImageNet-22K ViT-g/16 44-epoch checkpoint at
`https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar`, with the
official `in22k_vitg16_ep44.yaml` architecture settings: 224px input, 16px patch,
196 tokens, and embedding dimension 1408.
**Rationale:** This is the only available official Meta checkpoint that matches the
project's 224px/16px/196-token geometry. The tradeoff is substantially higher VRAM
usage and a changed latent dimension; the first run must profile memory on the
RTX 4060 before pipeline commitment.
**Baseline / prior state:** Encoder implementation was blocked on a nonexistent
official ViT-B/16 checkpoint.
**Outcome after change:** The project configuration and wrapper contract now target
ViT-g/16; production construction remains pending a local checkout of Meta's
archived source repository and a memory smoke test.
