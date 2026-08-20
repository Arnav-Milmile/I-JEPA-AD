---
name: anomaly-metrics-validation
description: >
  Validates anomaly-detection metric implementations (image AUROC, pixel AUROC,
  PRO/AUPRO) for I-JEPA-AD against known-answer synthetic examples and established
  libraries. Always use this before trusting any change to evaluation/metrics.py —
  a silently wrong metric implementation invalidates every reported result.
---

# Anomaly Metrics Validation

Before trusting any change to evaluation/metrics.py:

1. Construct a small synthetic labeled example (a handful of images/pixels with known
   ground truth and known prediction scores) where the correct AUROC/PRO value can be
   computed by hand or via a trusted reference (sklearn.metrics.roc_auc_score for
   image/pixel AUROC; anomalib's PRO implementation for PRO/AUPRO).
2. Assert the implementation under test matches the reference value within a tight
   numerical tolerance (e.g. 1e-4).
3. Pay specific attention to PRO/AUPRO's integration bounds and per-region
   normalization — this is the most common source of subtle bugs in anomaly
   localization metrics.
4. If sklearn/anomalib and the custom implementation disagree, do not silently prefer
   one — investigate and log the resolution and reasoning in DECISIONS.md.
5. Re-run this validation any time metrics.py changes, not just when it's first written.
6. Never report a benchmark number in results/ that was produced by an unvalidated
   metrics.py change.
