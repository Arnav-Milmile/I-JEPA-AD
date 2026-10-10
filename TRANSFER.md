# TRANSFER.md
_Last updated: 2026-10-10, by Claude Code (claude-opus-5-5), at commit 1e09fdc on branch `DataPipelineAndLatentCaching` (Phase 1 data-pipeline changes uncommitted in the working tree)_

## 1. One-paragraph project state
The data path is implemented and verified on CPU with synthetic data, up to the frozen encoder's input: MVTec discovery and validation, a deterministic nominal-only train/validation split, and deterministic I-JEPA preprocessing to `N×3×224×224`. The ViT-g/16 wrapper exists and is tested only with a tiny stand-in model. The real checkpoint has never been loaded, and the CUDA VRAM spike has not been run. Latent caching, the predictor, masking, baselines, and evaluation are not started.

## 2. Phase checklist
Mirrors `docs/IMPLEMENTATION_PLAN.md` §3.
- **Weeks 1–2, Foundations:** DONE for the data pipeline. Tooling, CI, utilities, MVTec ingestion, `data/transforms.py`, and `data/splits.py` all exist with tests. Still open: Hydra is not wired in (`configs/*.yaml` are not loaded by code).
- **Weeks 3–4, Encoder & Caching:** IN PROGRESS. The wrapper exists. Left: load the real checkpoint, run the VRAM spike (`docs/ENCODER_VITG16_CUDA_RUNBOOK.md`), build `caching/latent_cache.py` and `cache_schema.py`, and `encoder/checkpoint_registry.py`.
- **Weeks 5–6, Predictor:** NOT STARTED.
- **Weeks 7–8, Baselines:** NOT STARTED.
- **Weeks 9–10, Adaptive Masking Engine:** NOT STARTED.
- **Weeks 11–12, Benchmarking:** NOT STARTED.
- **Weeks 13–14, Full Dataset & Ablations:** NOT STARTED.
- **Weeks 15–16, Paper Assets:** NOT STARTED. `REPRODUCIBILITY.md` exists but only covers the source pin, checkpoint, and preprocessing.

## 3. What is verified working right now
Verified on Windows, Python 3.12.10, torch 2.14.1, Pillow 12.3.0, in `.venv`, by running the commands in §5:
- `ruff check .`, `black --check --workers 1 .`, and `mypy src tests` (26 files) all pass.
- `pytest tests/unit tests/integration`: 51 passed, also with `-W error::UserWarning`.
  - `tests/unit/test_transforms.py` (16): shapes, float32, exact normalized values for black and white, grayscale, RGBA and palette transparency, no-crop borders, non-square input, determinism for path vs. in-memory input, and error paths.
  - `tests/unit/test_splits.py` (12): nominal-only, disjoint, complete, seed-reproducible, input-order independent, global RNG untouched; rejects defect, masked, test, and empty records; insufficient-sample and invalid-argument errors.
  - `tests/unit/test_mvtec_dataset.py` (9): the original 4 split/leakage/mask tests plus metadata, label-by-directory, mask-by-stem, missing category, missing test dir, and empty train dir.
  - `tests/unit/test_seeding.py` (6), `test_ijepa_wrapper.py` (2), `test_logging_setup.py` (2), `test_profiling.py` (2), `test_scaffold.py` (1).
  - `tests/integration/test_data_pipeline.py` (1): synthetic PNG MVTec tree → split → `preprocess_batch` → `IJEPAEncoder.encode` with a stub model → `[N, 196, 1408]`.
- Preprocessing constants were checked against Meta's `src/transforms.py` at the pinned commit `52c1ae95` (see `REPRODUCIBILITY.md`).

Not verified: real MVTec images, the real checkpoint, any GPU run, CI on Linux/Python 3.11, and a numeric cross-check against torchvision (not installed).

## 4. What is NOT done yet (explicit, so it isn't assumed)
- Real-checkpoint load and the ViT-g/16 VRAM spike at batch sizes 1, 4, 8, and 16.
- Latent cache and cache schema (embedding dim, grid, checkpoint hash, preprocessing hash).
- Mask preprocessing for evaluation: no transform resizes ground-truth masks to 224 yet. This must stay geometrically consistent with the full-image resize, using nearest-neighbor interpolation.
- Hydra/OmegaConf config loading.
- Predictor, losses, training loop, masking engine, baselines, metrics (`tests/unit/test_metrics.py` does not exist), efficiency logging, statistical tests, run manifests, and all `scripts/` entrypoints except the VRAM spike.
- Regression tests. Integration tests cover only the data pipeline.

## 5. Environment & reproduction
```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/black.exe --check --workers 1 .
./.venv/Scripts/mypy.exe src tests
./.venv/Scripts/python.exe -m pytest tests/unit tests/integration
```
On Linux, use `.venv/bin/...`. CI uses Python 3.11, and this machine ran 3.12.

Local-only state that is not committed:
- `checkpoints/IN22K-vit.g.16-600e.pth.tar` and `external/ijepa` (pinned commit in `REPRODUCIBILITY.md`). See `docs/ENCODER_VITG16_CUDA_RUNBOOK.md`.
- MVTec AD under `data/raw/mvtec_ad` (the root-level `/data/` is gitignored). Validate it with `python validate_mvtec.py data/raw/mvtec_ad`.

Remotes: `origin` → `unknown7751/I-JEPA-AD` (fork), `upstream` → `Auxilus08/I-JEPA-AD`. Branch `basicfix` (commit `1e09fdc`) is pushed to `origin`.

## 6. Known issues / open threads
- **mypy and numpy on Python 3.12:** numpy 2.5.x stubs use 3.12-only `type` syntax, which mypy rejects with `python_version = "3.11"` once any checked module imports numpy directly. `transforms.py` avoids numpy for this reason. CI (Python 3.11) installs an older numpy and is unaffected.
- **`.gitignore` fix:** `data/` used to ignore `src/ijepa_ad/data/`, so new data modules were silently excluded from `git add`, and ruff and black (which respect `.gitignore`) never checked the data package. It is now `/data/`. Existing data modules had been force-added. The first ruff/black pass over the package found only one trailing-whitespace line in `mvtec_dataset.py`.
- **Hardware:** `docs/PRD.md` targets an RTX 4060 (8 GB), while the runbook targets an RTX 4090. The user does not plan to use their local GPU for training or full ViT-g/16 inference. VRAM results decide whether to keep ViT-g/16 or fall back to ViT-H/14.
- **Stale configs:** `configs/encoder/vitb16.yaml` and `vitl16.yaml` remain from the pre-ViT-g plan.
- **Stray file:** `.validate_mvtec.py.swp`, a vim swap file, is tracked in git and should be untracked.
- **Wrong filename references:** `DECISIONS.md` and `FLOW.md` headers reference `CLAUDE.md`, but the governing file is `AGENT.md`.
- **Loader leniency:** a mask under `ground_truth/good/` and orphan masks with no matching image are ignored, not rejected. Neither is required by the docs.

## 7. Where to look for more detail
- Decisions: `DECISIONS.md`, 2026-08-20 (toolchain, seeding, MVTec split, ViT-g/16) and 2026-10-10 (preprocessing, nominal split).
- Design and flow: `FLOW.md` §1–3.
- Source and checkpoint pin, preprocessing source: `REPRODUCIBILITY.md`.
- CUDA steps: `docs/ENCODER_VITG16_CUDA_RUNBOOK.md`.

## 8. Immediate next step
Run `python validate_mvtec.py data/raw/mvtec_ad` on the real dataset. Then construct `validate_category_splits` and `split_nominal_train_val` for the pilot categories (`bottle`, `capsule`, `hazelnut`), and run `preprocess_batch` on a few real images, all on CPU. After that, Phase 2 starts with the latent cache schema, which must record the preprocessing parameters from `PreprocessConfig`.
