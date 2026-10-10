# DECISIONS.md
_Append-only log of design/implementation decisions for I-JEPA-AD._
_Entry format and rules for what belongs here: see CLAUDE.md § Mandatory living documentation._
_Do not edit or remove past entries — if a decision is later reversed, add a new entry that supersedes it and links back._

---

## [2026-08-20] Establish initial repository toolchain
**Context:** §1 and §4.4 of the implementation plan require a reproducible Python
package layout, automated quality checks, and CPU-only tests before modeling work.
**Options considered:** Keep dependencies implicit; use a requirements-only layout;
or define package and development tooling in `pyproject.toml` with a pinned tool
lockfile.
**Decision:** Use a `src/` package layout, `pyproject.toml` for setuptools plus
Ruff, Black, mypy, and pytest configuration, `requirements-lock.txt` for exact
development-tool versions, and GitHub Actions/pre-commit for the same checks.
**Rationale:** This keeps local and CI commands aligned while leaving runtime and
modeling dependencies empty until their implementation phases.
**Baseline / prior state:** Documentation-only repository with no Python package or
automated checks.
**Outcome after change:** Empty package scaffold and smoke test are present; local
tooling verification is recorded in `TRANSFER.md` after the checks run.

## [2026-08-20] Centralize deterministic seeding and observability
**Context:** Weeks 1–2 require reproducibility controls and structured resource
logging before data and model modules are added.
**Options considered:** Let each future script seed its own libraries; configure
logging ad hoc per script; or expose shared utility entrypoints for seeding,
structured logs, and CUDA memory measurement.
**Decision:** Use one `set_seed(seed)` entrypoint for Python, NumPy, PyTorch, and
CUDA determinism; `setup_logging()` for idempotent JSONL logging; and
`profile_peak_vram()` around measurable code blocks using
`torch.cuda.max_memory_allocated()`.
**Rationale:** Centralized controls prevent inconsistent experiment state and make
peak VRAM and run events observable without coupling future modules together. Lazy
PyTorch loading keeps CPU-only configuration and test environments importable.
**Baseline / prior state:** Utility package contained only an empty initializer.
**Outcome after change:** All three utilities have unit tests; CPU test environments
report zero CUDA memory safely, while CUDA environments use PyTorch's peak allocator
counter. Further integration with training and inference is pending Phase 1+.

## [2026-08-20] Keep official MVTec test data evaluation-only
**Context:** MVTec AD ingestion needs strict train/test boundaries and a future
nominal validation strategy without leaking defect labels into model selection.
**Options considered:** Randomly split all images including test defects; use the
official test split for validation; or hold out validation samples from nominal
training images and reserve the official test split for final evaluation.
**Decision:** Discover training only from `train/good`, discover mixed labels from
the official `test` tree, require each defect image's matching
`ground_truth/<label>/<stem>_mask.*`, and reserve future validation as a nominal-only
holdout from training. Detect cross-split leakage using SHA-256 content IDs.
**Rationale:** This follows the MVTec AD protocol, keeps defect labels out of
hyperparameter selection, and catches copied images even when filenames are reused
between official directories.
**Baseline / prior state:** No data discovery or split validation existed.
**Outcome after change:** Synthetic fixtures verify nominal-only training, 2:2
nominal/defect test counts, mask pairing, non-nominal directory rejection, and
cross-split duplicate-content rejection. Future validation split implementation is
pending the predictor phase.

## [2026-08-20] Adopt Meta official I-JEPA ViT-g/16 checkpoint
**Context:** The PRD requested an official ViT-B/16 checkpoint, but Meta's official
I-JEPA release provides no pretrained ViT-B/16 weights. The user authorized the
official ViT-g/16 alternative.
**Options considered:** Train or source a non-official ViT-B/16 checkpoint; use
Meta's official ViT-g/16 checkpoint; or stop until an official ViT-B/16 release
exists.
**Decision:** Use Meta's official ImageNet-22K ViT-g/16 44-epoch checkpoint at
`https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar`, with the
official `in22k_vitg16_ep44.yaml` architecture settings: 224px input, 16px patch,
196 tokens, and embedding dimension 1408.
**Rationale:** This is the only available official Meta checkpoint that matches the
project's 224px/16px/196-token geometry. The tradeoff is substantially higher VRAM
usage and a changed latent dimension; the first run must profile memory on the
RTX 4060 before pipeline commitment.
**Baseline / prior state:** Encoder implementation was blocked on a nonexistent
official ViT-B/16 checkpoint.
**Outcome after change:** The project configuration and wrapper contract now target
ViT-g/16; production construction remains pending a local checkout of Meta's
archived source repository and a memory smoke test.

## [2026-10-10] Deterministic I-JEPA preprocessing
**Context:** Phase 1 needs one preprocessing path that turns MVTec images into
`N×3×224×224` tensors for the frozen ViT-g/16 encoder.
**Options considered:** Reuse Meta's training transform (`RandomResizedCrop`,
scale 0.3–1.0); `Resize(256)` + `CenterCrop(224)` as in many ImageNet eval and
PatchCore setups; or a direct full-image resize to 224×224.
**Decision:** Decode with Pillow, convert to RGB, bilinear-resize the full image to
224×224 with no crop, scale to `[0, 1]`, then normalize with mean
`(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`. Use the same transform for
train, validation, and test. Grayscale is replicated to 3 channels; images with
transparency are alpha-composited onto black; 16-bit and float modes are rejected.
Implemented in `src/ijepa_ad/data/transforms.py`.
**Rationale:** Normalization, `[0, 1]` scaling, 224px size, and bilinear interpolation
are verified against Meta's official `facebookresearch/ijepa` `src/transforms.py`
(`make_transforms` defaults) and `configs/in22k_vitg16_ep44.yaml` (flip, color
distortion, and blur all disabled). The official repo ships no deterministic eval
transform, so the resize policy is a project choice: random crops would make
anomaly scores non-deterministic, and center-cropping would discard border regions
where MVTec defects can occur.
**Baseline / prior state:** No preprocessing existed; `MVTecADDataset` returns path
metadata only.
**Outcome after change:** 16 unit tests cover shapes, dtype, exact normalized values,
grayscale, RGBA, palette transparency, no-crop borders, determinism, and error paths.
Not numerically cross-checked against torchvision (not installed). Pillow added as a
runtime dependency because torch alone cannot decode PNG/JPEG.

## [2026-10-10] Nominal-only train/validation split
**Context:** Predictor training needs a validation set for early stopping that never
contains defect images (IMPLEMENTATION_PLAN §3 Weeks 5–6, §5).
**Options considered:** Use `set_seed()` plus global `random.shuffle`; shuffle in
filesystem order with a local RNG; or sort by content hash and shuffle with an
isolated seeded RNG.
**Decision:** `split_nominal_train_val(samples, *, val_fraction, seed, min_train=1,
min_val=1)` in `src/ijepa_ad/data/splits.py`. It accepts only records that pass
`validate_training_records`, holds out `round(n * val_fraction)` samples (ties to
even), orders by SHA-256 `image_id` before shuffling with
`ijepa_ad.utils.seeding.seeded_rng(seed)`, and raises
`InsufficientNominalSamplesError` when either side would fall below its minimum.
`seeded_rng` is a new public helper sharing `set_seed`'s seed validation.
**Rationale:** Calling `set_seed()` inside a data function would reset the caller's
global RNG and torch determinism flags as a side effect. Ordering by content hash
makes the split depend only on image content and seed, not on paths or directory
listing order. Reusing the training validator guarantees no defect, masked, or
test-split record can enter train or validation.
**Baseline / prior state:** No validation split existed; the 2026-08-20 entry
reserved it as future nominal-only holdout work.
**Outcome after change:** 12 unit tests cover nominal-only output, disjointness,
completeness, seed reproducibility, input-order independence, global-state
isolation, rejection of defect/masked/test records, insufficient-sample errors, and
invalid arguments. Default `val_fraction: 0.1`, `seed: 0` recorded in
`configs/data/mvtec.yaml`.

## [2026-10-10] Latent cache format and schema
**Context:** Phase 2 needs a way to store frozen-encoder outputs (`N×196×1408`) so the
predictor never re-runs ViT-g/16, and to know when a stored cache is stale. Real
cache generation happens later on an external GPU machine; this phase is
CPU/synthetic only.
**Options considered:** `torch.save` of a plain dict with a pickled metadata object;
`safetensors` (new dependency); NumPy `.npz`; or `torch.save` of tensors and strings
only, with metadata as a canonical JSON string.
**Decision:** One `.pt` file per (category, split) at `<root>/<category>/<split>.pt`,
containing exactly `{"latents": Tensor, "metadata_json": str}`, loaded with
`torch.load(weights_only=True)`. Metadata is the frozen dataclass `CacheMetadata`
(`schema_version=1`) in `src/ijepa_ad/caching/cache_schema.py`, serialized as sorted,
compact JSON and fully re-validated on load (unknown or missing keys rejected, bool
is not an int, shape/dtype/token-count cross-checked, unsupported version raises
`UnsupportedSchemaVersionError`). It records category, split, ordered SHA-256 sample
ids, encoder name, checkpoint SHA-256, patch size, input shape, a `PreprocessingSpec`
(image size, mean, std, resize, crop), latent shape/dtype, token count, embedding
dim, a SHA-256 of the latent bytes, UTC creation time, and torch version. Reuse goes
through `CacheExpectations`; any difference raises `CacheIncompatibleError`.
`try_load_latent_cache` maps missing/stale to `None` but lets corruption raise.
Writes use a same-directory temp file, `fsync`, then `os.replace`. Category and split
must be plain names so they cannot escape the cache root.
**Rationale:** Tensors plus strings is the whole payload, so `weights_only=True`
works and no arbitrary object is ever unpickled; no dependency is added. The content
checksum catches truncation and bit rot that `torch.load` might not. Keeping the
metadata in the same file as the tensor means they cannot drift apart, and atomic
replace means a crash leaves either the old entry or none. One file per
category/split matches how the predictor consumes data (MVTec categories are a few
hundred images). Hostname, user, and absolute paths are deliberately not recorded.
**Baseline / prior state:** none — new (`caching/` held only an empty `__init__.py`).
**Outcome after change:** 37 unit tests in `tests/unit/test_latent_cache.py` pass on
CPU with small synthetic tensors (round trip, dtypes, bad/missing fields, corruption,
tampering, each invalidation rule, atomic-write failure). Not verified: real
checkpoint hashing, real ViT-g/16 outputs, real MVTec images, GPU. Latents are
stored in the dtype the caller passes (float32/float16/bfloat16); choosing the
production dtype is pending the VRAM spike.

## [2026-10-10] Resumable cache generation and schema version 2
**Context:** Phase 3 connects the data pipeline to the cache. A split-level entry held
whole in memory cannot be resumed after a crash, a failed sample must never shift
latents onto the wrong sample, and the schema did not record a preprocessing version.
Supersedes the schema-version-1 details of the 2026-10-10 "Latent cache format and
schema" entry; the file layout, payload keys, safe loading, and atomic write are
unchanged.
**Options considered:** Shard files per batch (fragile id association); one per-sample
file as the only layout (changes the decided layout); or per-sample *shards* that are
themselves valid one-sample cache entries, assembled into the unchanged
`<category>/<split>.pt`.
**Decision:** (1) `caching/builder.py::build_latent_cache` preprocesses each sample
individually, encodes in batches, and writes every sample's latents to
`<root>/<category>/.<split>.parts/<image_id>.pt` (a one-sample entry saved with
`save_latent_cache`, so it carries and is checked against full metadata). Compatible
shards are skipped on resume; missing, incompatible, or corrupted shards are
recomputed (shards are derived data). When every sample has a shard they are
assembled in sample order into the final entry, then the parts directory is removed.
A compatible final entry short-circuits everything; a corrupted final entry raises
and is never overwritten silently. (2) Any sample that cannot be preprocessed is
recorded and raises `CacheBuildError` after the remaining samples are processed; good
shards are kept and no final entry is written from partial results. An encoder that
returns the wrong number of rows raises before anything is saved. (3) `train`/`val`
entries accept only records passing `validate_training_records` (`train/good`,
unmasked, `split == "train"`); `build_nominal_latent_caches` goes through
`split_nominal_train_val`. (4) `PreprocessingSpec` gains `version`
(`transforms.PREPROCESSING_VERSION = 1`); `SCHEMA_VERSION` is now 2. Bump the
preprocessing version whenever `preprocess_image` changes output for the same config.
(5) `LatentCache` gains `get/select/index_of/in` for lookup by `image_id`, and
`cache_exists(path, expected)` fully validates before answering. Cache identity is
category + split + ordered sample ids (content hashes) + encoder name + checkpoint
SHA-256 + patch size + embedding dim + preprocessing spec + schema version; no
absolute path is part of it.
**Rationale:** Reusing `save_latent_cache` and `CacheExpectations` for shards avoids a
second format and gives resume the same checks as reuse. Writing shards per batch
bounds memory during encoding. Failing loudly on a bad sample keeps split-level
entries aligned with the dataset instead of silently shorter.
**Baseline / prior state:** Schema v1 (no preprocessing version); nothing called
`save_latent_cache` outside tests; no lookup by id; no resume.
**Outcome after change:** New unit tests (lookup, `cache_exists`, preprocessing version)
and 11 integration tests (mock encoder, synthetic MVTec tree, CPU) cover association,
resume, corrupt shards, invalidation, failed samples, and nominal-only enforcement. No
v1 caches existed, so nothing is migrated: v1 files are rejected as incompatible. Known
limits: assembly still builds the whole split tensor in host RAM (about 0.55 MB per
sample in float16); latents from different batch compositions can differ by float
rounding (not bit-identical); real checkpoint and GPU runs are unverified.

## [2026-10-10] Cache-generation CLI, run summary, and failure policy
**Context:** Phase 4 asks for a configurable, resumable generation pipeline with a
structured run summary, usable on an external GPU host. Phase 3's `caching/builder.py`
already is the orchestrator (batching, per-sample shards, resume, invalidation), so
only the gaps below were filled; no second orchestrator was created.
**Options considered:** A new `cache_builder.py` duplicating the builder; Hydra/
OmegaConf for configuration; per-sample retry when the encoder raises; or a thin CLI
and summary layer over the existing builder with argparse.
**Decision:** (1) `BuildReport` gains `recomputed` (shards that existed but were
corrupt or incompatible) and `failed`; `CacheBuildError` carries the partial `report`.
`build_latent_cache` gains `num_tokens`, checked with `embedding_dim` on every encoder
output before anything is written; a wrong shape raises `CacheBuildError`. (2) New
`caching/cli.py`: `CacheRunConfig`, `run_cache_generation(config, encoder=None) ->
RunSummary`, `main(argv)` (`python -m ijepa_ad.caching.cli`, thin wrapper
`scripts/build_latent_cache.py`). Plain argparse; Hydra stays unwired (no new
dependency). Defaults mirror `configs/data/mvtec.yaml` and `configs/encoder/vitg16.yaml`.
(3) Purposes: `nominal` caches `train`+`val` from `train/good` via
`split_nominal_train_val`; `test` caches only the official test split, as a separate
entry. Both go through `validate_category_splits`. (4) Failure policy: unreadable
samples and unusable final entries are recorded in the summary and the run continues
with the next entry; exit code 1 if any failed, 2 for bad configuration. An exception
raised by the encoder itself aborts the run and is not retried per sample (OOM or a
wrong model fails every sample; finished shards stay, so rerunning resumes). (5) The
mock encoder (`--mock-encoder`) is CPU-only, writes under `<cache-root>/_mock`, uses
the fixed identity `mock_encoder` / checkpoint hash `0*64`, and the summary reports
`encoder_kind: "mock"`, `real_checkpoint_inference: false`. (6) `--rebuild` deletes the
selected entries and shards before building; resume is otherwise automatic. (7) The
real encoder's input batches are moved to `--device` by a small adapter; the checkpoint
is hashed once per run.
**Rationale:** The mock cache directory cannot shadow or overwrite an expensive real
cache. Retrying per sample would hide systematic failures and double GPU time, while
resume already gives safe recovery. Continuing past a bad sample or entry gives one
complete report per run instead of one error at a time.
**Baseline / prior state:** Phase 3 builder and tests only; no entry point, no
recomputed/failed accounting, no token-count check, no device handling.
**Outcome after change:** 118 tests pass on CPU (102 before): 5 more builder tests and
11 CLI/summary tests (summary totals, resume counts, recomputed shards, nominal-only
and test-only purposes, `--rebuild`, config errors, exit codes). A mock CLI run on a
synthetic tree produced `[5, 196, 1408]`. Not verified: real checkpoint, GPU, real MVTec.

## [2026-10-10] Phase 5 end-to-end validation and two integration fixes
**Context:** Phase 5 asks whether dataset → validation → nominal split → preprocessing
→ encoder interface → cache → reload works as one system, and whether leakage,
stale-cache reuse, or misassociation is possible. Verified against the code, not the
earlier docs. No existing decision is reversed; the split policy, preprocessing, cache
schema (v2), and failure policy are unchanged.
**Options considered:** Only add tests; fix defects found by the tests with a regression
test written first; or also redesign the failure policy (per-sample retry, catching all
exceptions in the CLI).
**Decision:** (1) `MVTecADDataset` raises `ValueError` for any split other than `train`
or `test`. Before, an unknown split such as `"val"` fell through to the test-tree
branch, and with `validate=False` returned official test images labelled `val`; a
validation set must never be readable from `test/`. (2) `build_latent_cache` checks
that every encoder output batch is finite and raises `CacheBuildError` before any shard
of that batch is written. Before, `save_latent_cache` raised a bare `CacheMetadataError`
partway through the batch, which the CLI did not catch (traceback, no summary). It now
follows the existing wrong-shape policy: the CLI records the entry as failed and moves
on; shards from earlier batches stay and are valid. (3) Confirmed, unchanged: duplicate
image content inside `train/good` is rejected, not deduplicated (`duplicate image ID`),
so one bad category stops that category's run. (4) Confirmed, unchanged: cache rows
follow the order of the records passed in (split output is path-sorted), so the same
images under different filenames give a different row order and are therefore rebuilt,
never misread. (5) Reuse always re-runs record validation first; an existing cache never
bypasses dataset or leakage checks.
**Rationale:** Both defects were found by probing, then pinned by a failing regression
test before the fix. The fixes are minimal and keep existing behaviour for valid input.
**Baseline / prior state:** 118 tests passed; the two behaviours above were untested.
**Outcome after change:** 137 tests pass on CPU: 18 in
`tests/integration/test_pipeline_end_to_end.py` and 1 in `test_mvtec_dataset.py`. Covered:
full pipeline with association checks, order/filename-independent split membership,
duplicate and leaked images, reuse after dataset invalidation, global RNG and source
files untouched, cache files moved across category/split, shards copied under another
sample's name, failed atomic replace, truncated and tampered entries, reordered ids,
changed dim/seed, repeat-run identity, batch-size tolerance. Synthetic data and a mock
encoder only; no real MVTec, checkpoint, or GPU.
