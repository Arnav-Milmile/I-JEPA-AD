---
name: vram-budget-guard
description: >
  Confirms any change touching batch size, model size, or numerical precision stays
  within the 8GB VRAM budget for I-JEPA-AD (PRD §6). Use this whenever such a change is
  made, wrapping the relevant code path with profiling.py's VRAM context manager and
  failing loudly if the budget is exceeded.
---

# VRAM Budget Guard

1. Identify the code path affected by the change (training loop, inference loop,
   cache-building loop).
2. Wrap it with the peak-VRAM context manager from src/ijepa_ad/utils/profiling.py
   (torch.cuda.max_memory_allocated()-based).
3. Run the affected path (a short smoke run is sufficient — no need for a full
   training run) and record peak VRAM.
4. If peak VRAM exceeds 8GB, do not silently reduce batch size — report this to the
   user and log the mitigation (AMP, smaller batch, gradient checkpointing, etc.) as a
   DECISIONS.md entry with the before/after VRAM numbers.
5. If VRAM stayed within budget, still log the peak figure in the relevant script's
   run manifest / FLOW.md note so it's visible without re-running.
