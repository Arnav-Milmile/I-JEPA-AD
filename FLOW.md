# FLOW.md
_Living HLD + LLD + runtime-flow reference for I-JEPA-AD._
_This file is rewritten in place as the system changes — it is NOT append-only. Update rules: see CLAUDE.md._
_Last reviewed against src/: 2026-08-20 — scaffold only._

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

Each `src/ijepa_ad/` subpackage currently contains only an empty initializer.

## 2. Low-Level Design
No classes, functions, or data schemas have been implemented. Configuration files
are placeholders for the Hydra/OmegaConf work planned in the next phase.

## 3. Runtime Flow
Training and inference flows are not implemented yet.

## 4. Divergences from the original PRD
The initial scaffold uses `requirements-lock.txt` as the permitted lockfile variant
from §1 and keeps runtime dependencies empty. See `DECISIONS.md`, 2026-08-20.
