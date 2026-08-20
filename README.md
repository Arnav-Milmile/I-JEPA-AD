# I-JEPA-AD

Research codebase for decoder-free latent anomaly detection with I-JEPA.

The repository is currently scaffolded only. Modeling and data-pipeline code will
be added in later implementation phases.

## Development

```bash
python -m pip install -e ".[dev]"
pre-commit install
ruff check .
black --check .
mypy src tests
pytest tests/unit tests/integration
```
