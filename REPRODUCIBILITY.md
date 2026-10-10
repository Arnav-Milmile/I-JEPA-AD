# REPRODUCIBILITY.md

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
