---
name: tensor-shape-contract-testing
description: >
  Enforces tensor shape/dtype contracts at module boundaries for I-JEPA-AD. Use this
  whenever writing or editing any module with tensor I/O — the encoder wrapper, latent
  cache, predictor, or masking engine — to add explicit shape/dtype assertions and the
  matching unit test. Always consult this before considering a tensor-handling module
  complete.
---

# Tensor Shape Contract Testing

For any function taking or returning a tensor:

1. Add an explicit assertion (not just a docstring note) checking the expected shape
   and dtype at the function boundary — e.g. `assert z.shape == (B, N, D)`,
   `assert z.dtype == torch.float32` (or float16 under AMP — check the config).
2. Reference expected values against the PRD (e.g. N=196 patches for 224x224 input at
   patch size 16, per PRD §4.1) rather than a magic number with no source.
3. Write or update a unit test in tests/unit/ that feeds a small synthetic tensor of
   the correct AND an incorrect shape, and confirms the function behaves correctly on
   the former and raises a clear error on the latter.
4. If the module wraps a frozen component (the I-JEPA encoder), also assert
   requires_grad == False for all its parameters, and test this explicitly.
5. Log any nonstandard shape decision (e.g. deviating from the PRD's assumed patch
   grid) in DECISIONS.md.
