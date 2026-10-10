# TRANSFER.md
_Last updated: 2026-10-10 10:45, at commit 24dca75 on branch `DataPipelineAndLatentCaching` (Phase 2 cache changes uncommitted in the working tree)_

## 1. One-paragraph project state
The data path is implemented and verified on CPU with synthetic data: MVTec discovery and validation, a deterministic nominal-only train/validation split, and deterministic I-JEPA preprocessing to `N×3×224×224`. As of this commit plus the working tree, a versioned latent cache (`caching/cache_schema.py`, `caching/latent_cache.py`) can save and load encoder-shaped tensors with validated metadata, integrity checksum, invalidation checks, and atomic writes; it is tested only with small synthetic tensors. The ViT-g/16 wrapper is tested only with a tiny stand-in model. The real checkpoint has never been loaded, the CUDA VRAM spike has not been run, and no real latent cache exists. The predictor, masking, baselines, and evaluation are not started.

## 2. Phase checklist
Mirrors `docs/IMPLEMENTATION_PLAN.md` §3.
- **Weeks 1–2, Foundations:** DONE for the data pipeline. Still open: Hydra is not wired in (`configs/*.yaml` are not loaded by code).
- **Weeks 3–4, Encoder & Caching:** IN PROGRESS. Done: wrapper, cache schema and storage (synthetic-tested). Left: load the real checkpoint, run the VRAM spike (`docs/ENCODER_VITG16_CUDA_RUNBOOK.md`), `encoder/checkpoint_registry.py`, a script that builds real caches on a GPU host, and decide the production latent dtype.
- **Weeks 5–6, Predictor:** NOT STARTED.
- **Weeks 7–8, Baselines:** NOT STARTED.
- **Weeks 9–10, Adaptive Masking Engine:** NOT STARTED.
- **Weeks 11–12, Benchmarking:** NOT STARTED.
- **Weeks 13–14, Full Dataset & Ablations:** NOT STARTED.
- **Weeks 15–16, Paper Assets:** NOT STARTED. `REPRODUCIBILITY.md` covers only the source pin, checkpoint, and preprocessing.

## 3. What is verified working right now
Verified on Windows, Python 3.12.10, torch 2.14.1+cpu, Pillow 12.3.0, in `.venv`, by running the commands in §5:
- `ruff check .`, `black --check --workers 1 .`, and `mypy src tests` (29 files) all pass.
- `pytest tests/unit tests/integration`: 88 passed.
  - `tests/unit/test_latent_cache.py` (37): canonical metadata JSON round trip; every missing field, unknown field, unsupported schema version, and a table of invalid values rejected; float32/float16/bfloat16 round trip with equal values and dtype; inference-mode tensors and the `2×196×1408` contract shape; shape, dtype, value, and NaN mismatches rejected on save; missing, truncated, garbage, wrong-key, non-tensor, tampered, and pickled-object files rejected on load; invalidation on checkpoint hash, preprocessing, encoder name, patch size, embedding dim, category, split, sample ids, and schema version; `try_load_latent_cache` returns `None` for missing/stale but raises on corruption; a failed write or failed `os.replace` leaves the previous entry intact and no temp file.
  - Phase 1 tests: `test_transforms.py` (16), `test_splits.py` (12), `test_mvtec_dataset.py` (9), `test_seeding.py` (6), `test_ijepa_wrapper.py` (2), `test_logging_setup.py` (2), `test_profiling.py` (2), `test_scaffold.py` (1), and `tests/integration/test_data_pipeline.py` (1: synthetic MVTec tree → split → preprocess → stub encoder → `[N, 196, 1408]`).
- Preprocessing constants were checked against Meta's `src/transforms.py` at the pinned commit `52c1ae95` (see `REPRODUCIBILITY.md`).

Not verified: real MVTec images, the real checkpoint (including hashing it), any GPU run, a real cache for any category, CI on Linux/Python 3.11, atomic-write behavior under a real power loss, and a numeric cross-check against torchvision (not installed).

## 4. What is NOT done yet (explicit, so it isn't assumed)
- Real-checkpoint load and the ViT-g/16 VRAM spike at batch sizes 1, 4, 8, and 16.
- Any real latent cache. No script yet builds one: `save_latent_cache` exists but nothing calls it outside tests. Cache dtype for production (float16 vs float32) is undecided.
- No shard/append support: an entry is one whole (category, split) tensor held in memory. Revisit if a category is too large for host RAM.
- Mask preprocessing for evaluation: no transform resizes ground-truth masks to 224 yet. It must stay geometrically consistent with the full-image resize, using nearest-neighbor interpolation.
- Hydra/OmegaConf config loading (`latent_cache.root` in `configs/data/mvtec.yaml` and `encoder_name` in `configs/encoder/vitg16.yaml` are read by nothing).
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
./.venv/Scripts/python.exe -m pytest tests/unit/test_latent_cache.py   # cache only
```
On Linux, use `.venv/bin/...`. CI uses Python 3.11, and this machine ran 3.12.

Cache usage (see `FLOW.md` §2 for the schema and invalidation rules):
```python
from ijepa_ad.caching.cache_schema import CacheExpectations, PreprocessingSpec
from ijepa_ad.caching.latent_cache import (
    build_metadata, compute_file_sha256, latent_cache_path,
    save_latent_cache, try_load_latent_cache,
)
spec = PreprocessingSpec.from_config(PreprocessConfig())
path = latent_cache_path("data/cache/latents", "bottle", "train")
meta = build_metadata(latents, category="bottle", split="train",
    sample_ids=[s.image_id for s in split.train], encoder_name="ijepa_vit_g16_in22k_ep44",
    checkpoint_sha256=compute_file_sha256(ckpt), patch_size=16, preprocessing=spec)
save_latent_cache(path, latents, meta)
cached = try_load_latent_cache(path, CacheExpectations(category="bottle", split="train",
    encoder_name="ijepa_vit_g16_in22k_ep44", checkpoint_sha256=meta.checkpoint_sha256,
    patch_size=16, embedding_dim=1408, preprocessing=spec))  # None => rebuild
```

Local-only state that is not committed:
- `checkpoints/IN22K-vit.g.16-600e.pth.tar` and `external/ijepa` (pinned commit in `REPRODUCIBILITY.md`). See `docs/ENCODER_VITG16_CUDA_RUNBOOK.md`.
- MVTec AD under `data/raw/mvtec_ad` (the root-level `/data/` is gitignored). Validate it with `python validate_mvtec.py data/raw/mvtec_ad`.
- Latent caches under `data/cache/latents/` (also covered by `/data/`). None exist yet; they are to be generated on a GPU host.

Remotes: `origin` → `unknown7751/I-JEPA-AD` (fork), `upstream` → `Auxilus08/I-JEPA-AD`. Branch `DataPipelineAndLatentCaching` is pushed to `origin` at `24dca75`.

## 6. Known issues / open threads
- **mypy and numpy on Python 3.12:** numpy 2.5.x stubs use 3.12-only `type` syntax, which mypy rejects with `python_version = "3.11"` once any checked module imports numpy directly. `transforms.py` and the cache modules avoid importing numpy directly (the cache calls `Tensor.numpy()` only).
- **mypy and pytest:** with `follow_imports = "skip"`, `@pytest.mark.parametrize` is reported as an untyped decorator under `strict`, so tests use plain loops instead.
- **Hardware:** `docs/PRD.md` targets an RTX 4060 (8 GB), while the runbook targets an RTX 4090. The user does not plan to use their local GPU for training or full ViT-g/16 inference. VRAM results decide whether to keep ViT-g/16 or fall back to ViT-H/14.
- **Stale configs:** `configs/encoder/vitb16.yaml` and `vitl16.yaml` remain from the pre-ViT-g plan.
- **Stray file:** `.validate_mvtec.py.swp`, a vim swap file, is tracked in git and should be untracked.
- **Wrong filename references:** `DECISIONS.md` and `FLOW.md` headers reference `CLAUDE.md`, but the governing file is `AGENT.md`.
- **Loader leniency:** a mask under `ground_truth/good/` and orphan masks with no matching image are ignored, not rejected. Neither is required by the docs.
- **Checkpoint hash cost:** `compute_file_sha256` reads the whole multi-GB checkpoint; compute it once per run, not per cache entry.

## 7. Where to look for more detail
- Decisions: `DECISIONS.md`, 2026-08-20 (toolchain, seeding, MVTec split, ViT-g/16), 2026-10-10 (preprocessing, nominal split, latent cache format and schema).
- Design, cache schema, and flow: `FLOW.md` §1–3.
- Source and checkpoint pin, preprocessing source: `REPRODUCIBILITY.md`.
- CUDA steps: `docs/ENCODER_VITG16_CUDA_RUNBOOK.md`.

## 8. Immediate next step
On a GPU host: load the real checkpoint, run the VRAM spike, then add a script that, for each pilot category (`bottle`, `capsule`, `hazelnut`), runs `validate_category_splits` → `split_nominal_train_val` → `preprocess_batch` → `IJEPAEncoder.encode` → `build_metadata` → `save_latent_cache`, and confirms the saved files reload through `try_load_latent_cache`. Before that, on CPU: run `python validate_mvtec.py data/raw/mvtec_ad` on the real dataset and `preprocess_batch` on a few real images. The predictor (Phase 3) reads only from the cache.
