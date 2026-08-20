# TRANSFER.md
_Last updated: 2026-08-20 21:41 IST, by Codex, at commit 1afa2f066046c966b1ca9882810f43bf76a3804b (working tree changes not committed)_

## 1. One-paragraph project state
The repository now has the §1 directory skeleton, an importable empty `ijepa_ad`
package, placeholder experiment configuration files, development-tool configuration,
pre-commit hooks, and push/PR GitHub Actions CI. No modeling, dataset, or runtime
pipeline code has been written.

## 2. Phase checklist
- Phase 0 (Scaffolding): DONE — package layout, tooling, CI, and smoke test added.
- Phase 1 (Encoder & Caching): NOT STARTED
- Phase 2 (Predictor): NOT STARTED
- Phase 3 (Baselines): NOT STARTED
- Phase 4 (Adaptive Masking Engine): NOT STARTED
- Phase 5 (Benchmarking & Regression Safety): NOT STARTED
- Phase 6 (Full Dataset & Ablations): NOT STARTED
- Phase 7 (Paper Assets & Documentation): NOT STARTED

## 3. What is verified working right now
- The scaffold smoke test imports `ijepa_ad`: `tests/unit/test_scaffold.py`.
- Local equivalents of every CI quality check pass: Ruff, Black (single-worker),
  mypy, and pytest (1 test passed).
- The configured push/PR workflow is defined in `.github/workflows/ci.yaml`.

## 4. What is NOT done yet (explicit, so it isn't assumed)
- All modeling, data loading, caching, evaluation, and experiment scripts.
- Hydra/OmegaConf configuration composition and concrete experiment configs.
- Unit, integration, and regression tests beyond the scaffold smoke test.
- A GitHub-hosted CI run has not yet occurred.

## 5. Environment & reproduction
- From the repository root: `python -m pip install -e ".[dev]"`.
- Or install exact development tools with `python -m pip install -r requirements-lock.txt`.
- Run checks with `ruff check .`, `black --check .`, `mypy src tests`, and
  `pytest tests/unit tests/integration`.
- No model checkpoints, datasets, caches, or GPU/CUDA state are present.

## 6. Known issues / open threads
- The base environment initially lacked the development tools; they were installed
  successfully for local verification.
- CI dependency installation relies on the package index available to GitHub Actions.

## 7. Where to look for more detail
- Requirements and phase plan: `docs/IMPLEMENTATION_PLAN.md`.
- Architecture state: `FLOW.md`.
- Design choices: `DECISIONS.md`.
- Agent conventions: `AGENT.md`.

## 8. Immediate next step
Install the development extras and run the CI command set locally, then begin the
data-foundation phase with the required tests.
