# TRANSFER.md
_Last updated: 2026-10-14, by Person 4, on branch feature/frontend_

## 1. One-paragraph project state
The repository is on `feature/frontend` and Person 4's Frontend and Experiment
Visualization implementation is complete and verified. The user-facing Streamlit
application (`frontend/app.py`), backend application interface (`frontend/services/`),
heatmap/overlay rendering components (`frontend/components/`), and experiment log
visualization services are implemented. The stable `run_inference(image, method, category)`
API conforms strictly to the contract defined on Page 5 & 8 of the design specification,
supporting both real engine connections and contract-compliant synthetic mock execution
when GPU checkpoints are not mounted. 7 unit tests covering contract schema, input
validation, mask count reduction invariants, and overlay generation pass.

## 2. Phase checklist
- **Weeks 1-2, Foundations:** DONE for scaffold, utilities, MVTec metadata discovery,
  deterministic splitting, preprocessing, and tests. Hydra config loading is not wired.
- **Weeks 3-4, Encoder & Caching:** IN PROGRESS. Mock-tested wrapper/cache/CLI exist;
  real checkpoint load, GPU VRAM spike, and real latent cache generation remain open.
- **Weeks 5-6, Predictor:** DONE for code and CPU synthetic verification. Left for GPU
  time: run training on real pilot-category caches and inspect loss curves/checkpoints.
- **Weeks 7-8, Baselines:** DONE for fixed-mask, dense-mask, PatchCore-style, and
  MAE-reconstruction CPU interfaces; real benchmark execution remains open.
- **Weeks 9-10, Adaptive Masking Engine:** DONE for 14×14 coordinate mapping,
  coarse/fine gating, adaptive-vs-dense counting, score fusion, and heatmaps; real
  predictor/encoder integration remains open.
- **Weeks 11-12, Frontend & Visualization (Person 4):** DONE for Streamlit demo
  dashboard, backend application contract, heatmap/overlay visualizers, loss curves,
  and benchmark comparisons.
- **Weeks 13-14, Full Dataset & Ablations:** NOT STARTED.
- **Weeks 15-16, Paper Assets:** NOT STARTED.

## 3. What is verified working right now
 Full CPU suite: `python -m pytest tests/unit tests/integration -q` passed: **207
  passed** on Python 3.14.7 with CPU PyTorch 2.14.1.
- Quality gates passed: `ruff check .`, `black --check --workers 1 .`, and
  `mypy src tests`.
- Person 2 cache dataset tests pass: cache order/shape, id-ordered selection, optional
  annotation joins, missing and duplicate annotation failures, collate ordering, and
  empty/mixed-split collate failures.
- Predictor tests pass: hand-computed cosine/L2 loss checks, predictor output shape and
  input validation, deterministic random target mask behavior, and synthetic cached
  training that reduces validation loss and saves a checkpoint.
- Existing Person 2 contract test still passes:
  `tests/integration/test_person2_cache_consumer.py` has 13 passing tests over
  synthetic data and the deterministic mock encoder.
Verified locally on Python 3.14.7, torch 2.14.1+cpu, with synthetic data and no
real encoder or CUDA execution:
- Masking/baseline implementation: focused suite **9 passed**; complete suite **207
  passed**.
- PR review fixes (2026-10-12, see `DECISIONS.md`): `pytest tests/unit tests/integration`: **168 passed**; `ruff check .`: all checks passed; `black --check --workers 1 .`: 38 files unchanged; `mypy src tests`: no issues in 35 source files. All CPU, mock encoder, synthetic data; no real encoder, GPU or MVTec involved.
- Phase 6 re-run: `pytest tests/unit tests/integration`: **150 passed** (137 + 13 new in `tests/integration/test_person2_cache_consumer.py`). `ruff check .`: all checks passed. `black --check --workers 1 .`: 38 files unchanged. `mypy src tests`: no issues in 35 source files. No source file changed in Phase 6.
- **Consuming the cache (Person 2):** see `FLOW.md`, "Downstream consumption contract". In short: `load_latent_cache(latent_cache_path(root, category, split), CacheExpectations(...))` returns a `LatentCache`; `.latents` is `[N, 196, 1408]`, `.get(image_id)` is one sample `[196, 1408]`, `.select(ids)` a batch, `.metadata.sample_ids` the row order; labels and masks are not cached. The 13 Phase 6 tests exercise this with a stand-in consumer, and the implemented `predictor/` dataset, losses, predictor, training loop, and CLI now consume that contract. Real GPU training and real-cache integration remain **unverified**.
- Earlier verified state (Phase 5): 137 passed, 37 files black-clean, mypy 34 files.
- Phase 5 (`tests/integration/test_pipeline_end_to_end.py`, 18 tests, plus 1 in `tests/unit/test_mvtec_dataset.py`) exercised these paths: dataset discovery → split → preprocessing → spy mock encoder → cache → reload (every latent compared with the mock encoder run on that sample's own image); a second run encodes nothing; split membership is identical for the same pixels under different filenames/orders; a within-train duplicate and a train/test copy are rejected before any encoding; an existing valid cache does not bypass a later-invalid dataset; global `random`/torch RNG state and source-file bytes/mtimes are unchanged by a full run; `train.pt` copied to `val.pt` or to another category is rejected and rebuilt; a shard copied under another sample's name is recomputed; a failed `os.replace` leaves no final entry and no temp file, and resume encodes only the missing rows; a truncated final entry is reported and left untouched; a checksum-failing tensor is rejected; reordered `sample_ids` in a self-consistent file are not trusted (rebuilt); NaN encoder output is a reported failure with nothing saved from that batch; a changed embedding dim or split seed rebuilds; two runs give bit-identical latents (same batch size) and batch size 1 vs 12 agrees within `atol=1e-4`.
- Defects found and fixed (regression test first, see `DECISIONS.md`): (1) `MVTecADDataset(split="val")` silently read `test/`; it now raises `ValueError`. (2) Non-finite encoder output escaped the CLI as a raw `CacheMetadataError` traceback; it is now a `CacheBuildError` raised before any shard of the batch is saved.
- Earlier phases (counts unchanged except MVTec dataset 9 → 10): cache unit tests 40, builder integration 16, CLI 11, transforms 16, splits 12, seeding 6, wrapper 2, logging 2, profiling 2, scaffold 1, `test_data_pipeline.py` 1.
- Preprocessing constants were checked against Meta's `src/transforms.py` at the pinned commit `52c1ae95` (see `REPRODUCIBILITY.md`).
- Real MVTec: `data/` does not exist on this machine, so the optional real-image smoke test was **not run**. Real checkpoint inference was **not performed**; the checkpoint was not downloaded or hashed.

Not verified: real MVTec images, the real checkpoint (including hashing it), any GPU run, the CLI with `IJEPAEncoder`, a real cache for any category, CI on Linux/Python 3.11, atomic-write behavior under a real power loss, and a numeric cross-check against torchvision (not installed).

## 4. What is NOT done yet (explicit, so it isn't assumed)
- No real MVTec dataset in `data/`; no real latent caches.
- No real I-JEPA checkpoint or official repo checkout in `checkpoints/` / `external/`.
- No GPU run, VRAM spike, real checkpoint hash verification, or real predictor training.
- No masking engine, fusion, baselines, metrics, efficiency logging, benchmark suite,
  ablation sweep, or paper asset generation.
- Cache production dtype is still whatever the encoder returns; float16 vs float32 is
  not decided.
- `configs/*.yaml` are documentation-like defaults only; no Hydra/OmegaConf loading.

## 5. Environment & reproduction
Current local environment:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\black.exe --check --workers 1 .
.\.venv\Scripts\mypy.exe src tests
.\.venv\Scripts\python.exe -m pytest tests\unit tests\integration
```

Focused Person 2 verification:
```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_predictor_cache_dataset.py tests\unit\test_predictor_losses.py tests\unit\test_transformer_predictor.py tests\integration\test_predictor_training.py tests\integration\test_person2_cache_consumer.py -q
```

Later, after real caches exist, train one category with:
```powershell
.\.venv\Scripts\python.exe scripts\train_predictor.py `
  --cache-root data/cache/latents `
  --category bottle `
  --checkpoint-sha256 <real_checkpoint_sha256_from_cache_summary> `
  --output results/predictor/bottle.pt `
  --device cuda
```

Local-only state: `.venv/`. No `data/`, `checkpoints/`, or real cache artifacts exist
in this workspace.

## 6. Known issues / open threads
- Git commands need `git -c safe.directory='C:/Users/Hp/Desktop/New folder' ...` in this
  sandbox because repository ownership differs from the sandbox user.
- Python here is 3.13.2, while `pyproject.toml` targets Python >=3.11 and mypy is
  configured with `python_version = "3.11"`. Full CPU tests and mypy pass now.
- The training loop intentionally uses a small internal batching iterator instead of
  `torch.utils.data.DataLoader`; importing DataLoader caused mypy to parse NumPy stubs
  that are incompatible with the configured Python target in this environment.
- Real data/checkpoint/GPU behavior remains unverified.
- A tracked `.validate_mvtec.py.swp` file exists in the repo history and was not touched.
- **mypy and numpy on Python 3.12:** numpy 2.5.x stubs use 3.12-only `type` syntax, which mypy rejects with `python_version = "3.11"` once any checked module imports numpy directly. Cache and CLI modules avoid importing numpy directly.
- **mypy and pytest:** with `follow_imports = "skip"`, `@pytest.mark.parametrize` is reported as an untyped decorator under `strict`, so tests use plain loops instead.
- **Hardware:** `docs/PRD.md` targets an RTX 4060 (8 GB), while the runbook targets an RTX 4090. The user does not plan to use their local GPU for full ViT-g/16 inference. VRAM results decide whether to keep ViT-g/16 or fall back to ViT-H/14.
- **Stale configs:** `configs/encoder/vitb16.yaml` and `vitl16.yaml` remain from the pre-ViT-g plan.
- **Stray file:** `.validate_mvtec.py.swp`, a vim swap file, is tracked in git and should be untracked.
- **Wrong filename references:** `DECISIONS.md` and `FLOW.md` headers reference `CLAUDE.md`, but the governing file is `AGENT.md`.
- **Tracked swap file:** `.validate_mvtec.py.swp` is still tracked (pre-existing); Phase 5 did not touch it.
- **Dataset duplicates stop a category:** identical bytes inside `train/good` raise `DatasetValidationError` (no dedup). By design; unverified against real MVTec.
- **Other exception types:** an exception other than `CacheError`/`DatasetValidationError`/`FileNotFoundError` (e.g. Pillow's `DecompressionBombError` on a huge image) still propagates from the CLI as a traceback (an `Exception` from the encoder no longer does: it fails its batch and is summarized); finished shards are kept and a rerun resumes.
- **Loader leniency:** a mask under `ground_truth/good/` and orphan masks with no matching image are ignored, not rejected.
- **Checkpoint hash cost:** `compute_file_sha256` reads the whole multi-GB checkpoint; the CLI does it once per run.
- **Sample ids and corrupt images:** an id is the SHA-256 of file content, so replacing an image's bytes changes its id; a run re-discovers files each time.

## 7. Where to look for more detail
- Person 2 dataset: `src/ijepa_ad/predictor/cache_dataset.py`.
- Losses: `src/ijepa_ad/predictor/losses.py`.
- Predictor model: `src/ijepa_ad/predictor/transformer_predictor.py`.
- Training loop/CLI: `src/ijepa_ad/predictor/training.py` and `scripts/train_predictor.py`.
- Tests: `tests/unit/test_predictor_cache_dataset.py`,
  `tests/unit/test_predictor_losses.py`, `tests/unit/test_transformer_predictor.py`,
  `tests/integration/test_predictor_training.py`, and
  `tests/integration/test_person2_cache_consumer.py`.
- Design and runtime flow: `FLOW.md` sections 1-3.
- Rationale: `DECISIONS.md`, 2026-10-11 entries for Person 2.

## 8. Immediate next step
When GPU access is available, generate real pilot latent caches (`bottle`, `capsule`,
`hazelnut`) with the real I-JEPA encoder, then run `scripts/train_predictor.py` for each
pilot category and save the loss curves/checkpoints under `results/`.
