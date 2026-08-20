#!/usr/bin/env python3
"""Sanity-check a downloaded MVTec AD tree before it's wired into the project.
Usage: python validate_mvtec.py /path/to/data/raw/mvtec_ad
"""
import sys
from pathlib import Path

EXPECTED_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
}

def check_category(cat_dir: Path) -> list[str]:
    problems = []
    train_good = cat_dir / "train" / "good"
    test_dir = cat_dir / "test"
    gt_dir = cat_dir / "ground_truth"

    if not train_good.is_dir():
        problems.append(f"[{cat_dir.name}] missing train/good/")
        return problems
    n_train = len(list(train_good.glob("*.png")))
    if n_train == 0:
        problems.append(f"[{cat_dir.name}] train/good/ has 0 images")

    if not test_dir.is_dir():
        problems.append(f"[{cat_dir.name}] missing test/")
        return problems

    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        images = sorted(defect_dir.glob("*.png"))
        if defect_type == "good":
            continue  # nominal test images — no ground truth expected
        gt_subdir = gt_dir / defect_type
        if not gt_subdir.is_dir():
            problems.append(f"[{cat_dir.name}] no ground_truth/{defect_type}/ for test images")
            continue
        for img in images:
            expected_mask = gt_subdir / f"{img.stem}_mask.png"
            if not expected_mask.exists():
                problems.append(f"[{cat_dir.name}] missing mask for test/{defect_type}/{img.name}")
    return problems

def main(root: str):
    root_path = Path(root)
    if not root_path.is_dir():
        print(f"ERROR: {root} does not exist.")
        sys.exit(1)

    found = {p.name for p in root_path.iterdir() if p.is_dir()}
    missing = EXPECTED_CATEGORIES - found
    extra = found - EXPECTED_CATEGORIES

    all_problems = []
    for cat in sorted(found & EXPECTED_CATEGORIES):
        all_problems.extend(check_category(root_path / cat))

    print(f"Categories found: {len(found & EXPECTED_CATEGORIES)}/15")
    if missing:
        print(f"  MISSING categories: {sorted(missing)}")
    if extra:
        print(f"  Unexpected extra entries: {sorted(extra)}")

    if all_problems:
        print(f"\n{len(all_problems)} problem(s) found:")
        for p in all_problems[:50]:
            print(f"  - {p}")
        if len(all_problems) > 50:
            print(f"  ... and {len(all_problems) - 50} more")
        sys.exit(1)
    else:
        print("\nAll checked categories look structurally correct "
              "(train/good present, every defect image has a matching ground-truth mask).")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_mvtec.py /path/to/data/raw/mvtec_ad")
        sys.exit(1)
    main(sys.argv[1])
