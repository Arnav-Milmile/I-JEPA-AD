# REPRODUCIBILITY.md

## Latest repository audit

Audit date: **2026-10-13**, after pulling `origin/main` to repository commit
`621aa9c`. The working tree contains the documentation reconciliation and the
coarse-to-fine masking/baseline implementation. With Python 3.14.7 and CPU PyTorch
2.14.1, the full suite passes: **207 passed** across `tests/unit` and
`tests/integration`. Ruff, Black, and mypy also pass. CUDA validation remains
pending; the prior CUDA PyTorch installation was replaced locally with the CPU wheel
for this verification.

The pinned Meta source checkout and the 704,655,588-byte checkpoint are present
locally, but neither real checkpoint inference nor CUDA validation has been run.

The real MVTec tree is present and passed `python3 validate_mvtec.py
data/raw/mvtec_ad` (`15/15` categories). The strict repository validator also
passed `validate_category_splits` for every category; discovered counts are recorded
below. Real image decoding and cache generation were not run in this audit.

| Category | Train | Test | Nominal test | Defect test |
| --- | ---: | ---: | ---: | ---: |
| bottle | 209 | 83 | 20 | 63 |
| cable | 224 | 150 | 58 | 92 |
| capsule | 219 | 132 | 23 | 109 |
| carpet | 280 | 117 | 28 | 89 |
| grid | 264 | 78 | 21 | 57 |
| hazelnut | 391 | 110 | 40 | 70 |
| leather | 245 | 124 | 32 | 92 |
| metal_nut | 220 | 115 | 22 | 93 |
| pill | 267 | 167 | 26 | 141 |
| screw | 320 | 160 | 41 | 119 |
| tile | 230 | 117 | 33 | 84 |
| toothbrush | 60 | 42 | 12 | 30 |
| transistor | 213 | 100 | 60 | 40 |
| wood | 247 | 79 | 19 | 60 |
| zipper | 240 | 151 | 32 | 119 |

## Official I-JEPA source

- Repository: https://github.com/facebookresearch/ijepa
- Local checkout: `external/ijepa`
- Exact commit: `52c1ae95d05f743e000e8f10a1f3a79b10cff048`
- Commit date: 2023-06-13
- Working tree: clean, detached at the recorded commit

The repository is archived/read-only. The commit SHA above pins the exact source
used by `scripts/spikes/check_encoder_vram.py` and the production encoder wrapper.

## Image preprocessing

`src/ijepa_ad/data/transforms.py` matches `src/transforms.py` at the commit above:
`ToTensor()` scaling to `[0, 1]`, then `Normalize` with mean
`(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`, at 224px with bilinear
interpolation. The official random crop is replaced by a deterministic full-image
resize (see `DECISIONS.md`, 2026-10-10). Requires Pillow (`pillow==12.3.0` in
`requirements-lock.txt`).

## Official checkpoint

- URL: https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar
- Local path: `checkpoints/IN22K-vit.g.16-600e.pth.tar`
- Required state-dict key: `target_encoder`
- Architecture: ViT-g/16, 224px input, 196 patch tokens, 1408-dimensional output

## VRAM spike

Run from the repository root on a CUDA host:

```bash
python scripts/spikes/check_encoder_vram.py
```

The script is intentionally a throwaway diagnostic and is not part of the tested
package or CI. It measures forward-pass `torch.cuda.max_memory_allocated()` at batch
sizes 1, 4, 8, and 16 using FP16, `torch.no_grad()`, and CUDA autocast.

For the complete RTX 4090 setup, installation, checkpoint verification, non-skipped
unit-test check, VRAM table, and real-wrapper smoke test, follow
[`docs/ENCODER_VITG16_CUDA_RUNBOOK.md`](docs/ENCODER_VITG16_CUDA_RUNBOOK.md).

## Latent cache generation

A cache entry is reusable only when schema version (2), category, split, ordered
sample ids (SHA-256 of image content), encoder name, patch size, embedding dim,
checkpoint SHA-256, and the full preprocessing spec (image size, mean, std, resize,
crop, `PREPROCESSING_VERSION`) all match. Change any of them and the entry is rebuilt;
nothing is reused silently. Absolute paths are never part of the identity.

Procedure (GPU host, after the runbook's checkpoint verification):

1. `ckpt_sha = compute_file_sha256("checkpoints/IN22K-vit.g.16-600e.pth.tar")`, once
   per run, and record it with the run. Also record the `external/ijepa` commit above.
2. `train, test = validate_category_splits(root, category)`.
3. `build_nominal_latent_caches(train.samples, IJEPAEncoder(...), val_fraction=0.1,
   seed=0, cache_root="data/cache/latents", category=category,
   encoder_name="ijepa_vit_g16_in22k_ep44", checkpoint_sha256=ckpt_sha, patch_size=16,
   embedding_dim=1408)`.
4. Interrupted? Rerun the same call; compatible per-sample shards in
   `<category>/.<split>.parts/` are skipped. Delete that directory to force a rebuild.
5. Consumers call `load_latent_cache(path, CacheExpectations(...))` and `.get(image_id)`.

Latents may differ in the last float bits when the batch size changes, so record the
batch size and dtype with a run. Preprocessing compatibility: change
`preprocess_image` output, and `PREPROCESSING_VERSION` must be bumped.

Validation status: the generation path (dataset, split, preprocessing, resume, failure
handling, invalidation) is tested on CPU with synthetic images and a mock encoder. A
real run with the real checkpoint on a GPU, real MVTec images, and real
`[N, 196, 1408]` latents has not been done.

### Cache-generation command (Phase 4)

Inputs: MVTec AD under `--dataset-root` (default `data/raw/mvtec_ad`), the checkpoint
(`--checkpoint`, default `checkpoints/IN22K-vit.g.16-600e.pth.tar`) and the pinned
`external/ijepa` checkout (`--official-repo`). Nothing machine-specific is hardcoded.

```bash
# CPU smoke test, no checkpoint (writes to data/cache/latents/_mock):
python -m ijepa_ad.caching.cli --category bottle --mock-encoder

# Real run on the GPU host: nominal train/val caches, then the evaluation cache.
python -m ijepa_ad.caching.cli --category bottle capsule hazelnut --device cuda \
    --batch-size 8 --summary-json results/cache_nominal.json
python -m ijepa_ad.caching.cli --category bottle capsule hazelnut --device cuda \
    --purpose test --summary-json results/cache_test.json
```

Record with the run: the summary JSON (it contains the checkpoint SHA-256, device,
batch size, and `encoder_kind`), the `external/ijepa` commit, `--val-fraction` (0.1) and
`--seed` (0), and the torch version. Resume: rerun the identical command; compatible
entries and shards are skipped, missing/corrupt/incompatible shards are recomputed
(`recomputed` in the summary). A corrupted final entry is reported as failed and left
untouched; `--rebuild` discards the selected entries first and is the way to replace it.
An encoder exception (e.g. out of memory) fails its batch only: those samples appear
under `failed_samples` and a rerun encodes just them. Exit
code 0 means every entry is complete, 1 that some failed (see `failed_samples`), 2 a
configuration error. Mock caches live under `_mock` and carry encoder name
`mock_encoder` and a checkpoint hash derived from `--seed`, so they never satisfy a real
run's compatibility check and a different seed never reuses another projection.

Validation status: the CLI and summary are tested on CPU with synthetic images and the
mock encoder only. No real-checkpoint, GPU, or real-MVTec cache generation has been run.

## Historical CPU verification records

The following Phase 5 and Phase 6 sections preserve results from earlier
environments. They are evidence for the synthetic test coverage, not a claim that
the current Python environment can reproduce them; the latest audit is at the top of
this file.

### Phase 5 validation: reproducing the CPU test run

Environment used (Windows, CPU only, no network, no CUDA, no dataset, no checkpoint):
Python 3.12.10 (CI uses 3.11), torch 2.14.1+cpu, Pillow 12.3.0, ruff 0.8.6, black
25.1.0, mypy 1.14.1.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"      # Linux: .venv/bin/python
./.venv/Scripts/python.exe -m pytest tests/unit tests/integration        # 137 passed
./.venv/Scripts/python.exe -m pytest tests/integration/test_pipeline_end_to_end.py  # 18 passed
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/black.exe --check --workers 1 .
./.venv/Scripts/mypy.exe src tests
```

Obtained with synthetic data and the mock encoder only: every test result above, the
end-to-end pipeline, the cache failure/recovery tests, and the identical-latents check.

Deterministic guarantees (tested): split membership depends only on image content, the
seed, and `val_fraction` (not on filenames, enumeration or input order, and global RNG
state is left untouched); preprocessing is a pure function of the file bytes and
`PreprocessConfig`; cache metadata JSON is canonical (sorted, compact) and, apart from
`created_at_utc`, two runs give equal metadata; the mock encoder on CPU gives
bit-identical latents for the same inputs and batch size.

Not guaranteed or not measured: latents are only equal within float tolerance across
different batch sizes (tested at `atol=1e-4` with the mock); real-model inference on
GPU or CPU, other dtypes, and other torch/CUDA/cuDNN versions are unmeasured, as are
the `created_at_utc` and `torch_version` provenance fields, which vary by environment.
No bitwise-reproducibility claim is made for the real ViT-g/16.

Required for future real execution: MVTec AD under `data/raw/mvtec_ad`
(`python validate_mvtec.py data/raw/mvtec_ad` first), the checkpoint at
`checkpoints/IN22K-vit.g.16-600e.pth.tar`, the pinned `external/ijepa` checkout above,
and a CUDA host per `docs/ENCODER_VITG16_CUDA_RUNBOOK.md`.

Unverified on external GPU hardware: checkpoint load and hashing, the VRAM spike,
`IJEPAEncoder` inside the CLI (`_load_real_encoder`, device adapter), real
`[N, 196, 1408]` caches, float16 vs float32 storage, real-MVTec behaviour (including
whether any category has duplicate training images), and CI on Linux/Python 3.11.

### Phase 6 handoff: CPU commands and GPU validation

CPU commands (dev environment from `TRANSFER.md` §5):

```bash
./.venv/Scripts/python.exe -m pytest tests/integration/test_person2_cache_consumer.py -q  # 13 passed
./.venv/Scripts/python.exe -m pytest tests/unit tests/integration -q                      # 168 passed
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/black.exe --check --workers 1 .
./.venv/Scripts/mypy.exe src tests
```

What a downstream reader provides and receives: `FLOW.md`, "Downstream consumption
contract". It is tested with a stand-in consumer, not with Person 2's code.

External GPU procedure: [`docs/GPU_VALIDATION_CHECKLIST.md`](docs/GPU_VALIDATION_CHECKLIST.md)
covers the environment, checkpoint hashing (`sha256sum` and `compute_file_sha256` must
agree), the encoder smoke test with peak memory, real-dataset checks, a pilot cache
(`python -m ijepa_ad.caching.cli --category bottle --device cuda --batch-size 4
--summary-json results/pilot_nominal.json`), interrupt/resume, batch-size and
float16/float32 comparisons, and a results table. **It has not been executed.** The
checkpoint hash, GPU memory, real-image behaviour and real cache numbers stay unknown
until it is filled in. Suspected but unobserved: `IJEPAEncoder` loads the checkpoint with
torch's default `weights_only`, which recent torch may reject for this file.

## Person 2 predictor path: CPU verification and later GPU command

Verified locally on Windows/Python 3.13.2 with synthetic cached latents only:

```bash
./.venv/Scripts/python.exe -m pytest tests/unit/test_predictor_losses.py tests/unit/test_transformer_predictor.py tests/integration/test_predictor_training.py -q  # 8 passed
./.venv/Scripts/python.exe -m pytest tests/unit tests/integration  # 165 passed
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/black.exe --check --workers 1 .
./.venv/Scripts/mypy.exe src tests
```

Once real `train.pt` and `val.pt` latent caches exist for a category, train with:

```bash
./.venv/Scripts/python.exe scripts/train_predictor.py \
  --cache-root data/cache/latents \
  --category bottle \
  --checkpoint-sha256 <real_checkpoint_sha256_from_cache_summary> \
  --output results/predictor/bottle.pt \
  --device cuda
```

Adjust `--batch-size`, `--hidden-dim`, `--depth`, `--num-heads`, `--mask-ratio`, and
`--loss-type` to run ablations. This command has not been executed on real caches.
