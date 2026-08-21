# TRANSFER.md
_Last updated: 2026-08-21 18:42 IST, by Codex, at commit 1afa2f066046c966b1ca9882810f43bf76a3804b (working tree changes not committed)_

## 1. One-paragraph project state
The project targets Meta's official I-JEPA ViT-g/16 checkpoint. The wrapper, source
pin, checkpoint configuration, and CUDA VRAM spike exist. This session added a full
RTX 4090 reproduction runbook; no CUDA or PyTorch execution was attempted here by
request. The local source checkout and checkpoint remain external/local artifacts.

## 2. Phase checklist
- Phase 0 (Scaffolding): DONE — package layout, tooling, CI, utilities, and MVTec data foundation added.
- Phase 1 (Encoder & Caching): IN PROGRESS — wrapper and CUDA runbook exist; RTX 4090 smoke test remains.
- Phase 2 (Predictor): NOT STARTED
- Phase 3 (Baselines): NOT STARTED
- Phase 4 (Adaptive Masking Engine): NOT STARTED
- Phase 5 (Benchmarking & Regression Safety): NOT STARTED
- Phase 6 (Full Dataset & Ablations): NOT STARTED
- Phase 7 (Paper Assets & Documentation): NOT STARTED

## 3. What is verified working right now
- The CUDA reproduction instructions are documented in
  `docs/ENCODER_VITG16_CUDA_RUNBOOK.md`.
- The existing spike script and wrapper code remain available; no new CUDA result
  is claimed from this session.

## 4. What is NOT done yet (explicit, so it isn't assumed)
- Install a working CUDA-enabled PyTorch environment on the RTX 4090 machine.
- Run the non-skipped wrapper tests.
- Run the ViT-g/16 VRAM spike at batch sizes 1, 4, 8, and 16.
- Record PyTorch/CUDA versions, checkpoint hash, and VRAM results.
- Latent caching, predictor, baselines, masking, evaluation, and experiment code.

## 5. Environment & reproduction
- Follow `docs/ENCODER_VITG16_CUDA_RUNBOOK.md` from start to finish on the RTX 4090.
- The Meta source SHA and checkpoint URL are in `REPRODUCIBILITY.md`.
- Do not report wrapper tests as passing if pytest skips them.

## 6. Known issues / open threads
- The current development host lacks a usable PyTorch/CUDA runtime; installation was
  intentionally not pursued further.
- A pre-existing local modification to `src/ijepa_ad/data/mvtec_dataset.py` is not
  part of this session's changes.
- VRAM results determine whether ViT-g/16 is retained or the plan falls back to
  ViT-H/14.

## 7. Where to look for more detail
- CUDA setup and commands: `docs/ENCODER_VITG16_CUDA_RUNBOOK.md`.
- Source/checkpoint pin: `REPRODUCIBILITY.md`.
- Encoder contract: `FLOW.md` §2–3.
- Backbone decision: `DECISIONS.md`, 2026-08-20 ViT-g/16 entry.

## 8. Immediate next step
On the RTX 4090 machine, follow the runbook, run the wrapper tests without skips, and
record the four VRAM measurements before proceeding to latent caching.
