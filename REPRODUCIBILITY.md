# REPRODUCIBILITY.md

## Official I-JEPA source

- Repository: https://github.com/facebookresearch/ijepa
- Local checkout: `external/ijepa`
- Exact commit: `52c1ae95d05f743e000e8f10a1f3a79b10cff048`
- Commit date: 2023-06-13
- Working tree: clean, detached at the recorded commit

The repository is archived/read-only. The commit SHA above pins the exact source
used by `scripts/spikes/check_encoder_vram.py` and the production encoder wrapper.

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
