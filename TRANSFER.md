# TRANSFER.md
_Last updated: 2026-08-20 22:12 IST, by Codex, at commit 1afa2f066046c966b1ca9882810f43bf76a3804b (working tree changes not committed)_

## 1. One-paragraph project state
The repository now contains the §1 scaffold, foundational reproducibility utilities,
and strict MVTec AD split discovery/validation. Synthetic unit tests verify nominal
training data, official test mask pairing, split ratios, and cross-split leakage.
No encoder, cache, predictor, masking, or evaluation modeling code exists yet.

## 2. Phase checklist
- Phase 0 (Scaffolding): DONE — package layout, tooling, CI, utilities, and MVTec data foundation added.
- Phase 1 (Encoder & Caching): NOT STARTED
- Phase 2 (Predictor): NOT STARTED
- Phase 3 (Baselines): NOT STARTED
- Phase 4 (Adaptive Masking Engine): NOT STARTED
- Phase 5 (Benchmarking & Regression Safety): NOT STARTED
- Phase 6 (Full Dataset & Ablations): NOT STARTED
- Phase 7 (Paper Assets & Documentation): NOT STARTED

## 3. What is verified working right now
- `MVTecADDataset` discovers synthetic train/test records and enforces labels and
  official masks: `tests/unit/test_mvtec_dataset.py`.
- Duplicate content across train and test is rejected as leakage, and non-nominal
  training directories are rejected.
- Synthetic split fixture verifies 4 nominal train samples and a 2 nominal/2 defect
  test split.
- Ruff, Black, mypy, and pytest pass locally: 12 passed, 1 real-PyTorch test skipped
  because PyTorch is not installed in this environment.

## 4. What is NOT done yet (explicit, so it isn't assumed)
- A real MVTec download has not been ingested or benchmarked.
- Nominal validation holdout implementation for predictor model selection.
- All encoder, caching, predictor, baseline, masking, evaluation, and experiment code.
- A GitHub-hosted CI run has not yet occurred.

## 5. Environment & reproduction
- From the repository root: `python -m pip install -e ".[dev]"`.
- Run checks with `ruff check .`, `black --check --workers 1 .`, `mypy src tests`,
  and `pytest tests/unit tests/integration`.
- Synthetic data tests require no download or GPU. PyTorch remains lazy-loaded and
  is not installed in the current lightweight environment.

## 6. Known issues / open threads
- Content-hash image IDs intentionally detect copied files, while allowing repeated
  MVTec filenames with different content; verify this against the downloaded data.
- Future predictor work must carve validation only from nominal training records.

## 7. Where to look for more detail
- MVTec split contract: `FLOW.md` §2.
- Split-vs-validation rationale: `DECISIONS.md` entry dated 2026-08-20.
- Requirements and phase plan: `docs/IMPLEMENTATION_PLAN.md`.
- Agent conventions: `AGENT.md`.

## 8. Immediate next step
Add a nominal-only validation splitter and its tests when implementing the predictor
training foundation, preserving the official MVTec test set for final evaluation.
