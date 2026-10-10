#!/usr/bin/env python3
"""Generate latent caches (thin wrapper over ``python -m ijepa_ad.caching.cli``)."""

import sys

from ijepa_ad.caching.cli import main

if __name__ == "__main__":
    sys.exit(main())
