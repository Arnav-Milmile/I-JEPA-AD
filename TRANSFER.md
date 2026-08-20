# TRANSFER.md
_Last updated: 2026-08-20 22:06 IST, by Codex, at commit 1afa2f066046c966b1ca9882810f43bf76a3804b (working tree changes not committed)_

## 1. One-paragraph project state
The repository contains the §1 scaffold plus foundational utilities for deterministic
seeding, structured logging, and peak CUDA-memory profiling. The utilities are
implemented and unit-tested; no data, encoder, predictor, masking, or evaluation
modeling code exists yet.

## 2. Phase checklist
- Phase 0 (Scaffolding): DONE — package layout, tooling, CI, and foundational utilities added.
- Phase 1 (Encoder & Caching): NOT STARTED
- Phase 2 (Predictor): NOT STARTED
- Phase 3 (Baselines): NOT STARTED
- Phase 4 (Adaptive Masking Engine): NOT STARTED
- Phase 5 (Benchmarking & Regression Safety): NOT STARTED
- Phase 6 (Full Dataset & Ablations): NOT STARTED
- Phase 7 (Paper Assets & Documentation): NOT STARTED

## 3. What is verified working right now
- `set_seed()` repeats Python/NumPy sequences and exercises torch/CUDA calls through
  a fake backend: `tests/unit/test_seeding.py`.
- `setup_logging()` emits parseable JSON records and is idempotent:
  `tests/unit/test_logging_setup.py`.
- `profile_peak_vram()` safely handles CPU-only hosts and reads/reset/logs the CUDA
  peak counter through a fake backend: `tests/unit/test_profiling.py`.
- Ruff, Black, mypy, and pytest all pass locally: 8 passed, 1 real-PyTorch test
  skipped because PyTorch is not installed in this environment.

## 4. What is NOT done yet (explicit, so it isn't assumed)
- Real PyTorch execution of the optional torch determinism test on a PyTorch host.
- All data loading, encoder, caching, predictor, baseline, masking, evaluation, and
  experiment scripts.
- Hydra/OmegaConf configuration composition and concrete experiment configs.
- A GitHub-hosted CI run has not yet occurred.

## 5. Environment & reproduction
- From the repository root: `python -m pip install -e ".[dev]"`.
- Or install exact tools and NumPy with `python -m pip install -r requirements-lock.txt`.
- Run checks with `ruff check .`, `black --check --workers 1 .`, `mypy src tests`,
  and `pytest tests/unit tests/integration`.
- PyTorch is intentionally lazy-loaded by these utilities and is not installed in
  the current lightweight environment; a full modeling environment must provide it.

## 6. Known issues / open threads
- PyTorch/CUDA behavior still needs one verification run on the target training
  environment.
- Mypy is configured to skip imported third-party implementations so local tooling
  remains compatible with NumPy stubs across Python versions.

## 7. Where to look for more detail
- Requirements and phase plan: `docs/IMPLEMENTATION_PLAN.md`.
- Utility interfaces and runtime flow: `FLOW.md`.
- Seeding/toolchain rationale: `DECISIONS.md`.
- Agent conventions: `AGENT.md`.

## 8. Immediate next step
Begin the data-foundation phase by implementing `mvtec_dataset.py`, validators, and
their unit tests; call `set_seed()` from the first executable data/training entrypoint.
