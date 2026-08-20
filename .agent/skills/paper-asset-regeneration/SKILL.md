---
name: paper-asset-regeneration
description: >
  Regenerates every table and figure for the I-JEPA-AD paper from results/ and diffs
  the output against previously committed assets. Use this whenever results/ schema or
  content changes, to catch any number that silently shifted.
---

# Paper Asset Regeneration

1. Run scripts/generate_paper_assets.py end to end against the current results/
   directory.
2. Diff the newly generated tables/figures against the versions currently committed
   under docs/paper_assets/ (or wherever they're tracked).
3. If any number changed, do NOT silently accept the new version — identify which
   results/ file or code change caused the shift, and confirm whether it's expected
   (e.g. this ablation was intentionally re-run) or a regression (e.g. an accidental
   metrics.py or caching change).
4. If expected, log the change and its cause in DECISIONS.md, referencing the
   regenerated files. If unexpected, treat it as a bug — investigate before updating
   the committed assets.
5. Only commit the regenerated assets after this check, never as a blind overwrite.
