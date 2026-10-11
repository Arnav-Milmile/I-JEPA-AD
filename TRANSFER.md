# TRANSFER.md
_Last updated: 2026-10-11 14:40, by Codex, at commit c9b7bd9_

## 1. One-paragraph project state
The repo is on `main` and Person 2's implementation is now complete up to the point
that real GPU cache/training execution is required. The project can discover/validate
synthetic MVTec-style data, build/load mock latent caches, expose cached latents through
a predictor dataset, train a lightweight masked-token Transformer predictor on cached
latents, compute cosine or L2 prediction losses, and save a predictor checkpoint. All of
that is verified on CPU with synthetic cached latents. What remains for later GPU access
is generating real I-JEPA latent caches from MVTec and running `scripts/train_predictor.py`
on those caches. Masking/inference, baselines, evaluation metrics, real checkpoint load,
real MVTec, and real GPU results are still not done.

## 2. Phase checklist
- **Weeks 1-2, Foundations:** DONE for scaffold, utilities, MVTec metadata discovery,
  deterministic splitting, preprocessing, and tests. Hydra config loading is not wired.
- **Weeks 3-4, Encoder & Caching:** IN PROGRESS. Mock-tested wrapper/cache/CLI exist;
  real checkpoint load, GPU VRAM spike, and real latent cache generation remain open.
- **Weeks 5-6, Predictor:** DONE for code and CPU synthetic verification. Left for GPU
  time: run training on real pilot-category caches and inspect loss curves/checkpoints.
- **Weeks 7-8, Baselines:** NOT STARTED.
- **Weeks 9-10, Adaptive Masking Engine:** NOT STARTED.
- **Weeks 11-12, Benchmarking:** NOT STARTED.
- **Weeks 13-14, Full Dataset & Ablations:** NOT STARTED.
- **Weeks 15-16, Paper Assets:** NOT STARTED.

## 3. What is verified working right now
- Full CPU suite: `.\.venv\Scripts\python.exe -m pytest tests\unit tests\integration`
  passed: **165 passed** on Windows, Python 3.13.2.
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
