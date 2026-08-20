# FLOW.md
_Living HLD + LLD + runtime-flow reference for I-JEPA-AD._
_This file is rewritten in place as the system changes — it is NOT append-only. Update rules: see CLAUDE.md._
_Last reviewed against src/: 2026-08-20 — foundational utilities added._

## 1. High-Level Design
The current scaffold contains package boundaries only; no runtime data flow or
modeling modules exist yet.

```text
configs/ ──> future scripts/ ──> src/ijepa_ad/
                              ├── data
                              ├── encoder
                              ├── caching
                              ├── predictor
                              ├── masking
                              ├── baselines
                              ├── evaluation
                              └── utils
```

The `utils` package now provides cross-cutting controls used by every future
training and evaluation entrypoint: `set_seed()` establishes reproducible random
state, `setup_logging()` emits structured JSONL records, and `profile_peak_vram()`
records peak CUDA allocation around a code block.

The `data` package owns MVTec AD ingestion and split invariants. It discovers only
`<root>/<category>/train/good/*` for training and all label directories under
`<root>/<category>/test/*` for testing. Training records must be nominal and
unmasked. Test `good` records are nominal and unmasked; every defect record must
have the corresponding official mask at
`<root>/<category>/ground_truth/<defect>/<image_stem>_mask.*`. Image identity is a
SHA-256 digest of file content, which catches copied images even when MVTec
filenames repeat across its official train and test directories. The split validator
rejects duplicate IDs within a split and any ID shared across train and test.

## 2. Low-Level Design
Foundational utility interfaces:

- [`seeding.py`](src/ijepa_ad/utils/seeding.py): `set_seed(seed: int) -> None`
  seeds Python, NumPy, PyTorch, and CUDA and enables deterministic PyTorch
  algorithms.
- [`logging_setup.py`](src/ijepa_ad/utils/logging_setup.py):
  `setup_logging(level, log_file, logger_name) -> logging.Logger` configures
  idempotent stream/file JSON logging.
- [`profiling.py`](src/ijepa_ad/utils/profiling.py):
  `profile_peak_vram(label, logger, device) -> Iterator[PeakMemory]` resets and
  reads `torch.cuda.max_memory_allocated()` on CUDA hosts and safely reports zero
  on CPU-only hosts.

All remaining package modules are still empty. Configuration files are placeholders
for the Hydra/OmegaConf work planned in the next phase.

Data interfaces:

- [`mvtec_dataset.py`](src/ijepa_ad/data/mvtec_dataset.py):
  `MVTecADDataset(root, category, split)` discovers immutable `MVTecSample` records
  and validates the selected official split. `validate_category_splits(root,
  category)` validates both boundaries together.
- [`validators.py`](src/ijepa_ad/data/validators.py):
  `validate_training_records()`, `validate_test_records()`, and
  `validate_no_split_leakage()` enforce nominal-only training, official mask/label
  pairing, and no cross-split image IDs.

## 3. Runtime Flow
Before training or inference, an entrypoint calls `set_seed(seed)` once. Runtime
blocks can be wrapped as `with profile_peak_vram("stage"):` and all run output is
sent through the logger returned by `setup_logging()`.

The future predictor validation split must be carved from nominal training records
only; test defects remain evaluation-only and never participate in model selection.

## 4. Divergences from the original PRD
The initial scaffold uses `requirements-lock.txt` as the permitted lockfile variant
from §1 and keeps modeling dependencies empty. Foundational utilities use lazy
PyTorch loading so CPU-only tooling remains importable. See `DECISIONS.md`,
2026-08-20.
