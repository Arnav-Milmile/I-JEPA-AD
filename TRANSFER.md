# TRANSFER.md
_Last updated: 2026-10-10, at commit 71d33be (branch `DataPipelineAndLatentCaching`) plus Phase 4 changes_

## 1. One-paragraph project state
The data path is implemented and verified on CPU with synthetic data: MVTec discovery and validation, a deterministic nominal-only train/validation split, and deterministic I-JEPA preprocessing to `N×3×224×224`. A versioned latent cache (`caching/cache_schema.py`, `caching/latent_cache.py`, schema v2) saves and loads encoder-shaped tensors with validated metadata, checksum, invalidation checks, atomic writes, and lookup by sample id. `caching/builder.py` generates it resumably, and (Phase 4) `caching/cli.py` exposes that as a command with a structured run summary. All of it is tested only with synthetic tensors and a mock encoder. The ViT-g/16 wrapper is tested only with a tiny stand-in model. The real checkpoint has never been loaded, the CUDA VRAM spike has not been run, and no real latent cache exists. The predictor, masking, baselines, and evaluation are not started.

## 2. Phase checklist
Mirrors `docs/IMPLEMENTATION_PLAN.md` §3.
- **Weeks 1–2, Foundations:** DONE for the data pipeline. Still open: Hydra is not wired in (`configs/*.yaml` are not loaded by code; the CLI uses argparse defaults that mirror them).
- **Weeks 3–4, Encoder & Caching:** IN PROGRESS. Done: wrapper, cache schema/storage, resumable builder, CLI + run summary (all synthetic/mock-tested). Left: load the real checkpoint, run the VRAM spike (`docs/ENCODER_VITG16_CUDA_RUNBOOK.md`), `encoder/checkpoint_registry.py`, run the CLI with the real encoder on a GPU host, decide the production latent dtype.
- **Weeks 5–6, Predictor:** NOT STARTED.
- **Weeks 7–8, Baselines:** NOT STARTED.
- **Weeks 9–10, Adaptive Masking Engine:** NOT STARTED.
- **Weeks 11–12, Benchmarking:** NOT STARTED.
- **Weeks 13–14, Full Dataset & Ablations:** NOT STARTED.
- **Weeks 15–16, Paper Assets:** NOT STARTED. `REPRODUCIBILITY.md` covers source pin, checkpoint, preprocessing, and the cache-generation command.

## 3. What is verified working right now
Verified on Windows, Python 3.12.10, torch 2.14.1+cpu, Pillow 12.3.0, in `.venv`, by running the commands in §5:
- `pytest tests/unit tests/integration`: 118 passed. `ruff check .`, `black --check --workers 1 .` and `mypy src tests` (33 files) pass.
- Phase 4 tests: 5 added to `tests/integration/test_latent_cache_pipeline.py` (recomputed-shard counting; failed-sample report attached to `CacheBuildError`; wrong token count / embedding dim rejected before anything is saved; metadata consistent with the run; `test` split built only from official test records and rejecting train records) and 11 in `tests/integration/test_cache_cli.py` (summary totals and mock labelling; nominal caches exclude all test ids and write no `test.pt`; a second run skips everything; resume after a crash encodes only missing samples; a corrupted shard is counted as recomputed; an unreadable image is reported while another category still builds; `--purpose test`; `--rebuild`; configuration errors raise before any encoding or directory creation; CLI JSON output, `--summary-json`, exit codes 0/1/2, required `--category`).
- A mock CLI run on a synthetic tree (6 train images, 224px, batch 4) wrote a `[5, 196, 1408]` train entry under `<cache-root>/_mock`.
- Earlier phases: cache unit tests (`test_latent_cache.py`, 40), builder integration tests (`test_latent_cache_pipeline.py`, 16 now), transforms (16), splits (12), MVTec dataset (9), seeding (6), wrapper (2), logging (2), profiling (2), scaffold (1), `test_data_pipeline.py` (1).
- Preprocessing constants were checked against Meta's `src/transforms.py` at the pinned commit `52c1ae95` (see `REPRODUCIBILITY.md`).

Not verified: real MVTec images, the real checkpoint (including hashing it), any GPU run, the CLI with `IJEPAEncoder`, a real cache for any category, CI on Linux/Python 3.11, atomic-write behavior under a real power loss, and a numeric cross-check against torchvision (not installed).

## 4. What is NOT done yet (explicit, so it isn't assumed)
- Real-checkpoint load and the ViT-g/16 VRAM spike at batch sizes 1, 4, 8, and 16.
- Any real latent cache. The CLI's real-encoder path (`_load_real_encoder`, device adapter, checkpoint hashing) has never executed, only its configuration checks are tested. Cache dtype for production (float16 vs float32) is undecided; latents are stored in the dtype the encoder returns.
- Encoding memory is bounded by batch size (per-sample shards), but the final assembly still holds one whole (category, split) tensor in host RAM (about 0.55 MB per sample in float16, 1.1 MB in float32).
- Encoder exceptions abort the run with no per-sample retry (finished shards are kept; rerun resumes). Only unreadable images and unusable final entries are recorded and skipped past.
- Latents are not bit-identical across batch sizes (float summation order); tests use `atol=1e-4`.
- Schema v1 files are rejected as incompatible (none existed); stale `.<split>.parts/` directories are only reused if every shard matches the current configuration.
- Mask preprocessing for evaluation: no transform resizes ground-truth masks to 224 yet (nearest-neighbor, consistent with the full-image resize).
- Hydra/OmegaConf config loading (`latent_cache.root` and `encoder_name` are mirrored as CLI defaults only).
- Predictor, losses, training loop, masking engine, baselines, metrics, efficiency logging, statistical tests, run manifests, and `scripts/` entrypoints other than the VRAM spike and `build_latent_cache.py`.
- Regression tests. Integration tests cover only the data and cache pipelines.

## 5. Environment & reproduction
```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/black.exe --check --workers 1 .
./.venv/Scripts/mypy.exe src tests
./.venv/Scripts/python.exe -m pytest tests/unit tests/integration
./.venv/Scripts/python.exe -m pytest tests/integration/test_cache_cli.py   # Phase 4 only
```
On Linux, use `.venv/bin/...`. CI uses Python 3.11, and this machine ran 3.12.

Phase 4 files: new `src/ijepa_ad/caching/cli.py`, `scripts/build_latent_cache.py`, `tests/integration/test_cache_cli.py`; modified `src/ijepa_ad/caching/builder.py` (`BuildReport.recomputed/failed`, `CacheBuildError.report`, `num_tokens` check, `_shard_is_valid`), `tests/integration/test_latent_cache_pipeline.py`, and the four docs.

Cache generation (see `REPRODUCIBILITY.md` for the full procedure):
```bash
python -m ijepa_ad.caching.cli --help
python -m ijepa_ad.caching.cli --category bottle --mock-encoder          # CPU smoke test
python -m ijepa_ad.caching.cli --category bottle capsule hazelnut --device cuda   # real, GPU host
python -m ijepa_ad.caching.cli --category bottle capsule hazelnut --device cuda --purpose test
```
Python API:
```python
from ijepa_ad.caching.cli import CacheRunConfig, run_cache_generation
summary = run_cache_generation(CacheRunConfig(dataset_root=..., cache_root=..., categories=("bottle",)))
summary.to_dict()["totals"]   # samples_discovered, skipped_compatible, encoded_new, recomputed, failed
```
Lower-level API (`build_latent_cache`, `build_nominal_latent_caches`, `load_latent_cache`, `CacheExpectations`) is described in `FLOW.md` §2.

Local-only state that is not committed:
- `checkpoints/IN22K-vit.g.16-600e.pth.tar` and `external/ijepa` (pinned commit in `REPRODUCIBILITY.md`). See `docs/ENCODER_VITG16_CUDA_RUNBOOK.md`.
- MVTec AD under `data/raw/mvtec_ad` (the root-level `/data/` is gitignored). Validate it with `python validate_mvtec.py data/raw/mvtec_ad`.
- Latent caches under `data/cache/latents/` (also covered by `/data/`). None exist yet; they are to be generated on a GPU host.

Remotes: `origin` → `unknown7751/I-JEPA-AD` (fork), `upstream` → `Auxilus08/I-JEPA-AD`. Branch `DataPipelineAndLatentCaching` is pushed to `origin` at `71d33be`.

## 6. Known issues / open threads
- **mypy and numpy on Python 3.12:** numpy 2.5.x stubs use 3.12-only `type` syntax, which mypy rejects with `python_version = "3.11"` once any checked module imports numpy directly. Cache and CLI modules avoid importing numpy directly.
- **mypy and pytest:** with `follow_imports = "skip"`, `@pytest.mark.parametrize` is reported as an untyped decorator under `strict`, so tests use plain loops instead.
- **Hardware:** `docs/PRD.md` targets an RTX 4060 (8 GB), while the runbook targets an RTX 4090. The user does not plan to use their local GPU for full ViT-g/16 inference. VRAM results decide whether to keep ViT-g/16 or fall back to ViT-H/14.
- **Stale configs:** `configs/encoder/vitb16.yaml` and `vitl16.yaml` remain from the pre-ViT-g plan.
- **Stray file:** `.validate_mvtec.py.swp`, a vim swap file, is tracked in git and should be untracked.
- **Wrong filename references:** `DECISIONS.md` and `FLOW.md` headers reference `CLAUDE.md`, but the governing file is `AGENT.md`.
- **Loader leniency:** a mask under `ground_truth/good/` and orphan masks with no matching image are ignored, not rejected.
- **Checkpoint hash cost:** `compute_file_sha256` reads the whole multi-GB checkpoint; the CLI does it once per run.
- **Sample ids and corrupt images:** an id is the SHA-256 of file content, so replacing an image's bytes changes its id; a run re-discovers files each time.

## 7. Where to look for more detail
- Decisions: `DECISIONS.md`, 2026-08-20 (toolchain, seeding, MVTec split, ViT-g/16), 2026-10-10 (preprocessing, nominal split, cache format, resumable generation, CLI and failure policy).
- Design, cache schema, and flow: `FLOW.md` §1–3.
- Source and checkpoint pin, preprocessing, cache-generation command: `REPRODUCIBILITY.md`.
- CUDA steps: `docs/ENCODER_VITG16_CUDA_RUNBOOK.md`.

## 8. Immediate next step
On the GPU host: follow the runbook to load the real checkpoint and run the VRAM spike, then run `python -m ijepa_ad.caching.cli --category bottle capsule hazelnut --device cuda` (and `--purpose test`), confirm the summary reports `encoder_kind: "real"` and `[N, 196, 1408]` entries, and reload one entry with `load_latent_cache`. Before that, on CPU, run `python validate_mvtec.py data/raw/mvtec_ad` on the real dataset. The predictor (Phase 5) reads only from the cache.
