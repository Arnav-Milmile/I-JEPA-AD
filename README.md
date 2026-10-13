# I-JEPA-AD

Research codebase for decoder-free latent anomaly detection with I-JEPA.

The repository currently contains the validated data and latent-caching foundation:
strict MVTec AD split validation, deterministic preprocessing and nominal
train/validation splitting, a frozen Meta I-JEPA ViT-g/16 wrapper contract, and a
versioned resumable latent-cache CLI, a CPU-tested masked-token Transformer predictor,
coarse-to-fine masking/fusion, anomaly heatmaps, and four comparison baselines.
Real-checkpoint CUDA validation, real-dataset cache generation, predictor integration,
and benchmark metrics remain pending; see [TRANSFER.md](TRANSFER.md).

## Development

```bash
python -m pip install -e ".[dev]"
pre-commit install
ruff check .
black --check .
mypy src tests
pytest tests/unit tests/integration
```

The test suite requires a PyTorch installation that can import successfully on the
host. The CI workflow uses Python 3.11 and installs the project dependencies on an
Ubuntu runner; CUDA validation is intentionally kept out of CI and is documented in
[`docs/GPU_VALIDATION_CHECKLIST.md`](docs/GPU_VALIDATION_CHECKLIST.md).
