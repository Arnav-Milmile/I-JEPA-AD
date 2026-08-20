"""Smoke test for the initial repository scaffold."""

import ijepa_ad


def test_package_imports() -> None:
    """The source package is importable before implementation begins."""
    assert ijepa_ad.__name__ == "ijepa_ad"
