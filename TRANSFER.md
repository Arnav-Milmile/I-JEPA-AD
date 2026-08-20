# TRANSFER.md
_Last updated: 2026-08-20 23:32 IST, by Codex, at commit 1afa2f066046c966b1ca9882810f43bf76a3804b (working tree changes not committed)_

## 1. One-paragraph project state
The repository targets Meta's official I-JEPA ViT-g/16 ImageNet-22K checkpoint and
now includes a throwaway CUDA VRAM spike script. Meta's archived source is cloned at
`external/ijepa` and pinned to commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`.
The checkpoint exists locally but is Git-ignored. No VRAM numbers were produced here
because the environment has neither PyTorch nor CUDA.

## 2. Phase checklist
- Phase 0 (Scaffolding): DONE — package layout, tooling, CI, utilities, and MVTec data foundation added.
- Phase 1 (Encoder & Caching): IN PROGRESS — wrapper, source pin, checkpoint, and spike added; GPU memory smoke test remains.
- Phase 2 (Predictor): NOT STARTED
- Phase 3 (Baselines): NOT STARTED
- Phase 4 (Adaptive Masking Engine): NOT STARTED
- Phase 5 (Benchmarking & Regression Safety): NOT STARTED
- Phase 6 (Full Dataset & Ablations): NOT STARTED
- Phase 7 (Paper Assets & Documentation): NOT STARTED

## 3. What is verified working right now
- `scripts/spikes/check_encoder_vram.py` passes Ruff, Black, and Python compilation.
- The spike explicitly loads `target_encoder`, reports top-level keys when absent,
  uses Meta's `vit_giant` source, and measures batch sizes 1/4/8/16 with FP16
  autocast and `max_memory_allocated()`.
- The exact Meta source commit is recorded in `REPRODUCIBILITY.md`.

## 4. What is NOT done yet (explicit, so it isn't assumed)
- Actual ViT-g/16 CUDA execution and VRAM measurements.
- RTX 4060 8 GB decision based on batch 4–8 peak memory.
- Nominal validation holdout implementation for predictor model selection.
- All caching, predictor, baseline, masking, evaluation, and experiment code.

## 5. Environment & reproduction
- Run the spike from the repository root with:
  `python scripts/spikes/check_encoder_vram.py`.
- It requires PyTorch, a CUDA-capable GPU, `external/ijepa`, and
  `checkpoints/IN22K-vit.g.16-600e.pth.tar`.
- Source SHA, checkpoint URL, and expected output contract are in
  `REPRODUCIBILITY.md`.
- The current host reports no PyTorch installation, so the last spike attempt
  exited with code 2 before model loading.

## 6. Known issues / open threads
- The checkpoint is 704 MB locally and ignored by Git; acquire it separately in a
  fresh environment.
- VRAM results remain the gate for retaining ViT-g/16 versus falling back to ViT-H/14.
- A separate untracked `validate_mvtec.py` has pre-existing root Ruff formatting
  violations; the spike itself is clean.

## 7. Where to look for more detail
- Spike: `scripts/spikes/check_encoder_vram.py`.
- Source/checkpoint pin: `REPRODUCIBILITY.md`.
- ViT-g/16 contract: `FLOW.md` §2–3.
- Backbone decision: `DECISIONS.md`, 2026-08-20 ViT-g/16 entry.

## 8. Immediate next step
Run the spike on the RTX 4060 or another CUDA host and record the four reported
batch-size peaks before implementing latent caching or finalizing the backbone.
