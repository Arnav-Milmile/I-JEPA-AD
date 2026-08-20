# I-JEPA-AD Implementation Plan
### Engineering plan for a publishable, reproducible research codebase

This plan translates the PRD/Synopsis into a buildable software project. It is organized so that every phase produces (a) working, tested code and (b) a artifact you can cite in the paper (a table, a plot, a logged metric). The guiding principle: **treat this like a small research library, not a notebook.** Reviewers and your evaluators will both judge reproducibility as much as accuracy.

---

## 1. Repository Structure

```
ijepa-ad/
├── configs/                      # Hydra/OmegaConf YAML configs — never hardcode hyperparams
│   ├── encoder/vitb16.yaml
│   ├── encoder/vitl16.yaml
│   ├── predictor/base.yaml
│   ├── masking/coarse_to_fine.yaml
│   ├── masking/dense_baseline.yaml
│   ├── masking/fixed_mask.yaml
│   ├── data/mvtec.yaml
│   └── experiment/*.yaml         # composed configs, one per experiment run
│
├── src/ijepa_ad/
│   ├── data/
│   │   ├── mvtec_dataset.py      # Dataset + Dataloader, train/val/test split logic
│   │   ├── transforms.py
│   │   └── validators.py         # sanity-checks on shapes, label integrity
│   │
│   ├── encoder/
│   │   ├── ijepa_wrapper.py      # loads frozen checkpoint, exposes .encode(img) -> Z
│   │   └── checkpoint_registry.py
│   │
│   ├── caching/
│   │   ├── latent_cache.py       # write/read HDF5 or .pt latent shards
│   │   └── cache_schema.py       # explicit schema + version tag (see §6)
│   │
│   ├── predictor/
│   │   ├── transformer_predictor.py
│   │   └── losses.py             # cosine loss, L2 loss (both unit-tested independently)
│   │
│   ├── masking/
│   │   ├── coarse_masker.py
│   │   ├── fine_masker.py
│   │   ├── adaptive_engine.py    # Stage-1 → gate → Stage-2 orchestration
│   │   └── fusion.py             # S_final = S_coarse + λ S_fine
│   │
│   ├── baselines/
│   │   ├── patchcore.py          # thin wrapper around anomalib if licence permits, else reimplement
│   │   ├── mae_reconstruction.py
│   │   ├── fixed_mask_ijepa.py
│   │   └── dense_masked_ijepa.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py            # image AUROC, pixel AUROC, PRO/AUPRO
│   │   ├── efficiency.py         # mask counts, latency, VRAM logging
│   │   └── statistical_tests.py  # bootstrap CIs, paired significance tests
│   │
│   └── utils/
│       ├── seeding.py            # single source of truth for determinism
│       ├── profiling.py          # torch.cuda.max_memory_allocated wrappers
│       └── logging_setup.py
│
├── scripts/
│   ├── build_latent_cache.py
│   ├── train_predictor.py
│   ├── run_inference.py
│   ├── run_benchmark_suite.py    # runs all baselines + proposed model, writes results/
│   └── generate_paper_assets.py  # regenerates every table/figure in the paper from results/
│
├── tests/
│   ├── unit/                     # one file per src/ module, mirrors structure
│   ├── integration/               # end-to-end on a tiny synthetic dataset (see §4)
│   └── regression/                # frozen-checkpoint numeric regression (see §4.4)
│
├── results/                      # git-ignored; DVC- or checksum-tracked (see §6)
├── notebooks/                    # exploration only — nothing here feeds the paper directly
├── .github/workflows/ci.yaml
├── pyproject.toml                # ruff, black, mypy, pytest config
├── environment.lock (or poetry.lock / requirements-lock.txt)
└── README.md
```

**Why this shape:** every PRD component (§4 of the PRD) maps to exactly one module. This lets you cite "see `masking/adaptive_engine.py`" in your thesis and lets your evaluator diff behavior against the architecture diagram directly. It also means Stage 1, Stage 2, and fusion can each be unit-tested and ablated independently — which you need for RQ3–RQ5.

---

## 2. Engineering Principles to Apply

| Principle | Concrete application here |
| :--- | :--- |
| **Separation of concerns** | Encoder, caching, predictor, masking, and evaluation never import each other's internals — only public interfaces (`encode()`, `predict()`, `score()`). |
| **Config-driven, not code-driven, experiments** | Every hyperparameter in §7.4 of the PRD ($\tau$, $\lambda$, predictor depth, loss type) is a config field, never a magic number in code. This is what makes your ablation matrix a one-line CLI sweep instead of manual edits. |
| **Determinism** | One `seeding.py` sets `torch`, `numpy`, `random`, and CUDA determinism flags. Every run logs its seed. Ablations run ≥3 seeds so your AUROC numbers have variance bars, not single-run noise. |
| **Fail fast / explicit contracts** | `cache_schema.py` and `validators.py` assert tensor shapes/dtypes at every module boundary — a mismatched embedding dim should crash at cache-build time, not silently corrupt a 6-hour training run. |
| **Single Responsibility** | `adaptive_engine.py` only orchestrates; it contains no metric code and no model definitions. This is what lets Baseline 3/4 in your experimental matrix reuse `coarse_masker.py` and `fine_masker.py` directly instead of duplicating logic — which also means a bug fix in masking automatically fixes all baselines, avoiding an entire class of "baseline was actually run with different code" reviewer objections. |
| **Don't Repeat Yourself, but don't over-abstract early** | Baselines 3 and 4 (Fixed-Mask, Dense) are literally special cases of the adaptive engine with $\tau=-\infty$ or $\tau=+\infty$-equivalent configs. Implement them as config variants of the same engine where correct, and only fork into separate files where the algorithm is genuinely different (PatchCore, MAE). |
| **Observability** | Every training/inference run emits structured logs (JSON lines) plus a run manifest (config hash, git commit SHA, package versions). This is your reproducibility appendix for the paper, generated automatically rather than written by hand at the end. |

---

## 3. Revised Week-by-Week Plan (SWE-augmented)

The PRD's 16-week roadmap is sound at the milestone level; the additions below are the engineering scaffolding needed to make each milestone trustworthy and paper-ready.

**Weeks 1–2 — Foundations**
- Set up `pyproject.toml`, `ruff`/`black`/`mypy`, pre-commit hooks, GitHub Actions CI (lint + unit tests on every push).
- Stand up `configs/` with Hydra; write `seeding.py`, `logging_setup.py`, `profiling.py` first — everything downstream depends on these.
- Ingest MVTec AD; write `mvtec_dataset.py` **plus its unit tests** (correct image/mask pairing, correct train=nominal-only / test=mixed split, no leakage between categories).
- Deliverable: CI green on an empty-but-structured repo; dataset unit tests passing.

**Weeks 3–4 — Encoder & Caching**
- `ijepa_wrapper.py`: load frozen checkpoint, assert `requires_grad=False` for all parameters (test this explicitly — it's a correctness requirement in PRD §4.1, not just a performance one).
- `latent_cache.py` with an explicit `cache_schema.py` (embedding dim, patch grid size, checkpoint hash, image preprocessing hash — see §6 for why the hash matters).
- Unit tests: cache round-trip (write → read → tensor-equal), schema-version mismatch raises a clear error rather than silently loading stale/incompatible latents.
- Deliverable: latent cache built for `bottle`, `capsule`, `hazelnut`; VRAM logged and confirmed under budget for this stage.

**Weeks 5–6 — Predictor**
- `transformer_predictor.py`, `losses.py` (cosine + L2, unit-tested against hand-computed values on tiny tensors — this catches sign/direction bugs in the loss that are otherwise invisible until training "just doesn't converge").
- Training loop with early stopping on a held-out nominal validation split (carve this out of the nominal training set — do **not** validate on test-set defects, that would leak label information into model selection).
- Integration test: predictor overfits a 2-image synthetic dataset to near-zero loss (a standard "can it learn at all" sanity check).
- Deliverable: loss curves logged (e.g., to Weights & Biases or a local MLflow instance) for all 3 pilot categories, addressing R-04 in the PRD risk matrix directly.

**Weeks 7–8 — Baselines**
- Implement Fixed-Mask and Dense-Sliding-Mask I-JEPA baselines as config variants of the same masking engine (§2).
- Implement/wire PatchCore and MAE-reconstruction baselines — prefer wrapping `anomalib`'s implementations over reimplementing, since reviewers trust established baseline code more than a from-scratch reimplementation, and it removes an entire bug surface from your responsibility.
- Deliverable: `run_benchmark_suite.py` produces a first (rough) results table across all 4 baselines + proposed model on the pilot categories.

**Weeks 9–10 — Adaptive Masking Engine**
- `coarse_masker.py`, `fine_masker.py`, `adaptive_engine.py`, `fusion.py`.
- Unit tests here are unusually important because this is your core contribution — test with synthetic, hand-constructed discrepancy maps:
  - all-normal image → zero fine masks generated (gate correctly bypasses Stage 2).
  - single suspicious region → fine masks generated only over that region, with correct spatial coordinates (test the coordinate math directly, not just end-to-end AUROC — coordinate bugs silently produce plausible-looking but wrong heatmaps).
  - fusion formula matches closed-form expected output for known $S_{coarse}$, $S_{fine}$, $\lambda$.
- Deliverable: adaptive engine passes unit tests + produces qualitatively sane heatmaps on 5–10 hand-picked pilot images (visual regression check).

**Weeks 11–12 — Comparative Benchmarking**
- `efficiency.py`: mask-count counters and wall-clock profiling, averaged over 100 runs per PRD §6.
- Run dense vs. adaptive on pilot categories; this is your first RQ3/RQ4 evidence.
- Add a `tests/regression/` numeric baseline: pin expected AUROC/latency (within tolerance) on the pilot set so future refactors can't silently regress accuracy — this is the single highest-leverage test for a research codebase that will keep changing through week 16.
- Deliverable: first efficiency-vs-accuracy plot (candidate for the paper).

**Weeks 13–14 — Full Dataset & Ablations**
- Scale to all 15 MVTec categories. Run the full ablation matrix from PRD §7.4 as config sweeps (Hydra multirun or a simple shell/Python sweep script), each with ≥3 seeds.
- `statistical_tests.py`: bootstrap confidence intervals on AUROC, paired comparison (adaptive vs. dense) so your "measurably lower latency" and "$\ge 40\%$ reduction" claims (PRD §6) have error bars, not point estimates.
- Deliverable: full results tables, ablation CSVs, all reproducible via one command (`run_benchmark_suite.py --config full_eval`).

**Weeks 15–16 — Documentation & Paper Assets**
- `generate_paper_assets.py`: regenerates every table and figure from `results/` — no hand-edited numbers in the paper.
- Write a REPRODUCIBILITY.md (checkpoint hashes, exact package versions, hardware, total compute/GPU-hours, seeds used) — this is close to what NeurIPS/CVPR-style reproducibility checklists expect and strengthens the paper's credibility.
- Final code cleanup pass: docstring coverage check, remove dead baseline code paths, tag a `v1.0` release matching the paper submission.

---

## 4. Testing Strategy

For a research codebase headed toward a paper, tests serve two purposes: normal software correctness, and **protecting the validity of your reported numbers**. Structure tests in three tiers.

### 4.1 Unit tests (`tests/unit/`)
One file per module, mirroring `src/ijepa_ad/`. Priorities, in order of "how silently wrong could this go":
1. `losses.py` — hand-computed expected values on tiny fixed tensors.
2. `fusion.py` — closed-form check of $S_{final} = S_{coarse} + \lambda S_{fine}$ including $\lambda \in \{0.0, 2.0\}$ edge cases.
3. `coarse_masker.py` / `fine_masker.py` — correct patch/grid coordinate generation, no off-by-one errors in coarse→fine coordinate mapping.
4. `metrics.py` — validate your AUROC/PRO implementation against `anomalib`'s or `sklearn`'s on a synthetic labeled example with a known closed-form answer. This is critical: a subtly wrong metric implementation invalidates every number in the paper, and it's the easiest thing to get subtly wrong (e.g., PRO/AUPRO integration bounds).
5. `latent_cache.py` — round-trip integrity, schema mismatch detection.
6. `mvtec_dataset.py` — no train/test leakage, correct nominal-only training split.

### 4.2 Integration tests (`tests/integration/`)
Build one small synthetic dataset (e.g., 8×8 or 32×32 checkerboard-style synthetic "textures" with a few injected synthetic "defect" patches, generated once and committed or generated by a fixture) so integration tests run in seconds on CPU, not minutes on GPU with real MVTec data. Cover:
- End-to-end: image in → heatmap out, for both dense and adaptive pipelines, shapes and value ranges correct.
- Adaptive engine reduces mask-evaluation count relative to dense on this synthetic set (a cheap, fast proxy for RQ3, run on every CI push).
- Frozen encoder truly frozen: parameter checksums identical before/after a training step of the predictor.

### 4.3 Regression tests (`tests/regression/`)
Pin known-good numeric outputs (AUROC within ±X%, latency within ±Y%) on the 3 pilot categories using a fixed checkpoint and fixed seed. Run these on a schedule (e.g., nightly or pre-merge to `main`), not on every commit, since they need GPU and real data. This is what catches "I refactored the fusion module and silently changed the result by 3 AUROC points" — a very common failure mode in research code that otherwise only surfaces at paper-writing time, three weeks before deadline.

### 4.4 CI configuration
`.github/workflows/ci.yaml` should run, on every push/PR:
- `ruff` + `black --check` + `mypy`
- `pytest tests/unit tests/integration` (CPU-only, must complete in a few minutes)
- Regression tests as a separate, manually-triggered or nightly workflow if you have GPU CI available; otherwise document the exact command to run them locally before each major merge.

---

## 5. Evaluation Protocol (for paper-grade rigor)

- **Splits:** Standard MVTec AD protocol — train on nominal-only images, test on the official mixed nominal/defect test split per category. Do not use test-set defect images for any hyperparameter selection, including $\tau$ and $\lambda$ — hold out a small nominal validation slice for that, or select $\tau$/$\lambda$ on the pilot categories only and report generalization to the remaining 12 as a held-out check.
- **Metrics:** Image AUROC, Pixel AUROC, PRO/AUPRO — implement or reuse validated implementations (§4.1 point 4), report per-category and macro-averaged, matching the convention used by PatchCore/MVTec AD papers so your numbers are directly comparable in a results table.
- **Efficiency metrics:** mask count, latency (mean ± std over 100 runs per PRD §6), peak VRAM — log all three per-image, not just averaged, so you can also report distributions/tails if reviewers ask.
- **Statistical rigor:** ≥3 seeds per configuration; report mean ± std or bootstrap CI; use a paired test (e.g., Wilcoxon signed-rank across categories) when claiming adaptive beats dense on efficiency or matches it on accuracy — a bare point-estimate claim ("50% fewer masks") is exactly the kind of claim reviewers ask you to substantiate statistically.
- **Ablations:** run each PRD §7.4 sweep independently, holding all other config fields fixed — this is mechanically enforced if ablations are expressed as single-field config overrides (§2).
- **Threats to validity section (for the paper):** worth drafting early — likely candidates are (a) single-hardware VRAM constraint limiting batch size/model scale, (b) MVTec AD's relatively clean, well-aligned images vs. messier real factory-floor conditions, (c) frozen-encoder choice meaning results are bounded by the pretrained I-JEPA checkpoint's representation quality rather than your method alone.

---

## 6. Reproducibility & Experiment Tracking

- **Run manifests:** every training/eval run writes a JSON manifest: git commit SHA, config (fully resolved, not just overrides), package versions (`pip freeze` or lockfile hash), seed, hardware, timestamp, and a hash of the exact cached-latent files used.
- **Cache versioning:** `cache_schema.py` embeds the I-JEPA checkpoint hash and preprocessing version into the cache file itself. If you ever swap ViT-B/16 for ViT-L/16, stale caches fail loudly instead of silently mixing embeddings from two different encoders.
- **Experiment tracking:** Weights & Biases (free tier) or a local MLflow instance — either is fine; the requirement is that every number in the paper traces back to a specific logged run ID, not a number copy-pasted from a terminal that no longer exists.
- **Results storage:** `results/` is git-ignored but content-hashed or DVC-tracked, so `generate_paper_assets.py` can be re-run at submission time to regenerate the exact tables/figures without manual editing — this also protects you if a reviewer asks for a number you didn't originally tabulate.

---

## 7. Risk-to-Engineering-Control Mapping

| PRD Risk | Engineering control implemented |
| :--- | :--- |
| R-01 (missed small defects) | Unit tests on coarse-mask coordinate correctness + overlap; $\tau$ ablation is a config sweep, cheap to re-run. |
| R-02 (adaptive overhead negates savings) | `efficiency.py` logs mask counts and latency on every run, not just at the end — regressions caught early via regression tests (§4.3). |
| R-03 (OOM) | `profiling.py` asserts peak VRAM against an 8 GB budget in integration tests; AMP enabled by config default, not opt-in. |
| R-04 (predictor fails to learn) | Integration test that predictor overfits a tiny synthetic set; loss curves logged for pilot categories before scaling to full data. |

---

## 8. Suggested First Two PRs

1. **Scaffolding PR:** repo structure, `pyproject.toml`, CI config, `seeding.py`/`logging_setup.py`/`profiling.py` with unit tests. No modeling code yet — this establishes the contract everything else builds on.
2. **Data PR:** `mvtec_dataset.py` + `validators.py` + unit tests for split correctness. Small, reviewable, and unblocks Week 3 immediately.

Starting with these two keeps the first two weeks' deliverable concrete and testable, rather than "I read the I-JEPA paper" — which sets the tone for the rest of the project being artifact-driven.