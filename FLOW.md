# FLOW.md
_Living HLD + LLD + runtime-flow reference for I-JEPA-AD._
_This file is rewritten in place as the system changes — it is NOT append-only. Update rules: see CLAUDE.md._
_Last reviewed against src/: 2026-10-10 — Phase 3 resumable cache generation and lookup added._

## 1. High-Level Design
The data path is implemented through latent caching (generation, resume, lookup),
tested with a mock encoder. Predictor, masking, baselines, and evaluation
packages are still empty.

```text
<root>/<category>/{train,test,ground_truth}
        │
        ▼
data.mvtec_dataset.MVTecADDataset ──> data.validators (labels, masks, leakage)
        │  tuple[MVTecSample]   (path metadata only, no pixels)
        ▼
data.splits.split_nominal_train_val ──> NominalSplit(train, val)   [train/good only]
        │  image paths
        ▼
data.transforms.preprocess_batch ──> Tensor[N, 3, 224, 224]  (I-JEPA normalized)
        │
        ▼
encoder.ijepa_wrapper.IJEPAEncoder.encode ──> Tensor[N, 196, 1408]  (frozen)
        │
        ▼
caching.builder.build_latent_cache  (batches; per-sample shards
        │   <split>.parts/<id>.pt; resumable; assembles only when complete)
        ▼                                 ──> <root>/<category>/<split>.pt
        │  load_latent_cache(path, CacheExpectations) ──> LatentCache(latents, metadata)
        │  .get(image_id) / .select(ids)   — no encoder re-run
        ▼
predictor / masking / baselines / evaluation   — not implemented
```

The `utils` package now provides cross-cutting controls used by every future
training and evaluation entrypoint: `set_seed()` establishes reproducible random
state, `setup_logging()` emits structured JSONL records, and `profile_peak_vram()`
records peak CUDA allocation around a code block.

The `encoder` package wraps Meta's official I-JEPA ViT-g/16 ImageNet-22K
44-epoch checkpoint. It is configured for 224×224 RGB inputs, 16×16 patches,
196 patch tokens, and 1408-dimensional latents. The wrapper requires Meta's
official source checkout to construct the architecture and loads the checkpoint's
`target_encoder` state dict.

The `data` package owns MVTec AD ingestion and split invariants. It discovers only
`<root>/<category>/train/good/*` for training and all label directories under
`<root>/<category>/test/*` for testing. Training records must be nominal and
unmasked. Test `good` records are nominal and unmasked; every defect record must
have the corresponding official mask at
`<root>/<category>/ground_truth/<defect>/<image_stem>_mask.*`. Image identity is a
SHA-256 digest of file content, which catches copied images even when MVTec
filenames repeat across its official train and test directories. The split validator
rejects duplicate IDs within a split and any ID shared across train and test.

The predictor's validation set is a nominal-only holdout carved from the official
training records by `data.splits`; the official test split is never read or
modified by splitting. `data.transforms` turns any split's image paths into
encoder-ready tensors with one deterministic transform for train, validation, and
test: RGB conversion, full-image bilinear resize to 224×224 (no crop), `[0, 1]`
scaling, and the official I-JEPA ImageNet normalization.

## 2. Low-Level Design
Foundational utility interfaces:

- [`seeding.py`](src/ijepa_ad/utils/seeding.py): `set_seed(seed: int) -> None`
  seeds Python, NumPy, PyTorch, and CUDA and enables deterministic PyTorch
  algorithms. `seeded_rng(seed: int) -> random.Random` returns an isolated RNG
  with the same seed validation, for library code that must be deterministic
  without touching global state.
- [`logging_setup.py`](src/ijepa_ad/utils/logging_setup.py):
  `setup_logging(level, log_file, logger_name) -> logging.Logger` configures
  idempotent stream/file JSON logging.
- [`profiling.py`](src/ijepa_ad/utils/profiling.py):
  `profile_peak_vram(label, logger, device) -> Iterator[PeakMemory]` resets and
  reads `torch.cuda.max_memory_allocated()` on CUDA hosts and safely reports zero
  on CPU-only hosts.

Predictor, masking, baselines, and evaluation packages are still empty.
Hydra/OmegaConf is not wired in: `configs/data/mvtec.yaml` records `root`,
`pilot_categories`, `preprocessing.{image_size, resize, mean, std}`, and
`nominal_split.{val_fraction, seed, min_train, min_val}`, `latent_cache.root`, but no code loads it yet.
Its values mirror `PreprocessConfig` defaults and the arguments callers must pass to
`split_nominal_train_val`.

Data interfaces:

- [`mvtec_dataset.py`](src/ijepa_ad/data/mvtec_dataset.py):
  `MVTecADDataset(root, category, split)` discovers immutable `MVTecSample` records
  and validates the selected official split. `validate_category_splits(root,
  category)` validates both boundaries together.
- [`validators.py`](src/ijepa_ad/data/validators.py):
  `validate_training_records()`, `validate_test_records()`, and
  `validate_no_split_leakage()` enforce nominal-only training, official mask/label
  pairing, and no cross-split image IDs.
- [`splits.py`](src/ijepa_ad/data/splits.py):
  `split_nominal_train_val(samples, *, val_fraction, seed, min_train=1, min_val=1)
  -> NominalSplit` runs `validate_training_records()`, orders records by SHA-256
  `image_id`, shuffles with `seeded_rng(seed)`, and holds out
  `round(n * val_fraction)` records as validation. It raises
  `InsufficientNominalSamplesError` (a `DatasetValidationError`) when either side
  would fall below its minimum. `NominalSplit(train, val, seed, val_fraction)` is
  frozen, with both tuples sorted by image path.
- [`transforms.py`](src/ijepa_ad/data/transforms.py):
  `PreprocessConfig(image_size=224, mean=IJEPA_MEAN, std=IJEPA_STD)`,
  `load_image(path)`, `to_rgb(image)`, `preprocess_image(path | PIL.Image, config)
  -> Tensor[3, S, S]`, and `preprocess_batch(images, config) -> Tensor[N, 3, S, S]`.
  Grayscale is replicated to three channels, images with transparency are
  alpha-composited onto black, and 16-bit or float modes raise
  `ImagePreprocessingError` instead of being clipped.

Encoder interface:

- [`ijepa_wrapper.py`](src/ijepa_ad/encoder/ijepa_wrapper.py):
  `IJEPAEncoder(checkpoint_path, official_repo_path, device)` loads the official
  ViT-g/16 target encoder, switches it to `eval()`, freezes every parameter, and
  exposes `encode(image_batch) -> Tensor[N, 196, 1408]`. Loading fails if the
  checkpoint is missing, lacks `target_encoder`, or does not exactly match the
  architecture.

Caching interfaces (`src/ijepa_ad/caching/`):

- [`builder.py`](src/ijepa_ad/caching/builder.py): `build_latent_cache(samples,
  encoder, *, cache_root, category, split, encoder_name, checkpoint_sha256,
  patch_size, embedding_dim, config=None, batch_size=8) -> BuildReport`;
  `build_nominal_latent_caches(train_samples, encoder, *, val_fraction, seed,
  ...)` (split, then cache `train` and `val`); `parts_dir(...)`. `encoder` is
  anything with `encode(Tensor[N,3,S,S]) -> Tensor[N,tokens,dim]` (`IJEPAEncoder`
  or a mock). Error: `CacheBuildError` (has `.failed`).

- [`cache_schema.py`](src/ijepa_ad/caching/cache_schema.py): `SCHEMA_VERSION = 1`;
  frozen `CacheMetadata` (`to_json()`, `from_json()`, `from_dict()`),
  `PreprocessingSpec.from_config(PreprocessConfig)`, and `CacheExpectations`
  (`mismatches(metadata)`). Errors: `CacheMetadataError`,
  `UnsupportedSchemaVersionError`.
- [`latent_cache.py`](src/ijepa_ad/caching/latent_cache.py):
  `latent_cache_path(root, category, split)`, `build_metadata(latents, *, category,
  split, sample_ids, encoder_name, checkpoint_sha256, patch_size, preprocessing)`,
  `save_latent_cache(path, latents, metadata)`, `load_latent_cache(path, expected)`,
  `try_load_latent_cache(path, expected)`, `compute_file_sha256(path)`, `cache_exists(path, expected)`. `LatentCache` offers
  `get(id)`, `select(ids)`, `index_of(id)`, and `in`. Errors:
  `CacheNotFoundError`, `CacheCorruptedError`, `CacheIncompatibleError` (all
  `CacheError`).

Latent cache schema (version 2; v1 lacked `preprocessing.version`). Layout: `<cache_root>/<category>/<split>.pt`, where
split is `train`, `val` or `test`; default root `data/cache/latents` (gitignored via
`/data/`). The file holds exactly `latents` (`N×tokens×dim` tensor) and
`metadata_json` (canonical JSON of):

| field | meaning |
| --- | --- |
| `schema_version` | integer, must equal 2 |
| `category`, `split` | MVTec category and split name |
| `sample_ids` | ordered, unique SHA-256 `image_id`s, one per latent row |
| `encoder_name`, `patch_size` | encoder identifier (`configs/encoder/vitg16.yaml`) and patch size |
| `checkpoint_sha256` | SHA-256 of the checkpoint file |
| `input_shape` | `[3, S, S]` fed to the encoder |
| `preprocessing` | `image_size`, `mean`, `std`, `resize`, `crop`, `version` (`PREPROCESSING_VERSION`) |
| `latent_shape`, `latent_dtype`, `num_tokens`, `embedding_dim` | tensor description (`float16`/`bfloat16`/`float32`) |
| `latents_sha256` | SHA-256 of the latent bytes (integrity) |
| `created_at_utc`, `torch_version` | provenance |

Invalidation: an entry is reusable only if schema version, category, split, encoder
name, checkpoint hash, patch size, embedding dim, and the full preprocessing spec
(plus `sample_ids`, when the caller passes them) equal the caller's
`CacheExpectations`. Otherwise `CacheIncompatibleError`; `try_load_latent_cache`
returns `None` so the caller rebuilds. A checksum, shape, dtype, or metadata failure
is `CacheCorruptedError` and is never swallowed.

## 3. Runtime Flow
Before training or inference, an entrypoint calls `set_seed(seed)` once. Runtime
blocks can be wrapped as `with profile_peak_vram("stage"):` and all run output is
sent through the logger returned by `setup_logging()`.

Implemented data flow for one category (verified on synthetic data in
`tests/integration/test_data_pipeline.py` and `test_latent_cache_pipeline.py`
with mock encoders):

1. `train, test = validate_category_splits(root, category)`
2. `split = split_nominal_train_val(train, val_fraction=0.1, seed=0)`
3. `batch = preprocess_batch([sample.image_path for sample in split.train])`
4. `latents = IJEPAEncoder(...).encode(batch)` → `[N, 196, 1408]`
5. `meta = build_metadata(latents, category=..., split=..., sample_ids=[s.image_id
   ...], encoder_name=..., checkpoint_sha256=compute_file_sha256(ckpt),
   patch_size=16, preprocessing=PreprocessingSpec.from_config(config))`, then
   `save_latent_cache(latent_cache_path(root, category, split), latents, meta)`
6. Later runs: `try_load_latent_cache(path, CacheExpectations(...))`; `None` means
   rebuild from step 3.

Steps 3–6 are wrapped by `build_nominal_latent_caches(train.samples, encoder,
val_fraction=0.1, seed=0, cache_root=..., category=..., encoder_name=...,
checkpoint_sha256=..., patch_size=16, embedding_dim=1408)`. Per batch it
preprocesses each sample, encodes, and writes one shard per sample; a rerun skips
compatible shards; the final `<split>.pt` is written only when every sample has a
shard. An unreadable image raises `CacheBuildError` (good shards kept, no final
entry). Only mock encoders have been run; no real-encoder cache exists.

Test defects remain evaluation-only and never participate in model selection.

For training-time caching, nominal images flow through `IJEPAEncoder.encode()` once
and produce `[N, 196, 1408]` latent tensors. The first ViT-g/16 run must use batch
size 1, inference mode, and `profile_peak_vram()` to establish whether the RTX 4060
8 GB target is viable.

## 4. Divergences from the original PRD
The initial scaffold uses `requirements-lock.txt` as the permitted lockfile variant
from §1 and keeps modeling dependencies empty. Foundational utilities use lazy
PyTorch loading so CPU-only tooling remains importable. See `DECISIONS.md`,
2026-08-20.

The official I-JEPA repository has only a training transform (`RandomResizedCrop`).
Preprocessing here uses a deterministic full-image resize instead, so no border
region is cropped away. See `DECISIONS.md`, 2026-10-10 "Deterministic I-JEPA
preprocessing".
