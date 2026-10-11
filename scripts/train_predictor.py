"""Thin wrapper for ``python -m ijepa_ad.predictor.training``."""

import sys

from ijepa_ad.predictor.training import main

if __name__ == "__main__":
    sys.exit(main())
