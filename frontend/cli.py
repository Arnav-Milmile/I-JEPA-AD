"""CLI launcher for the I-JEPA-AD frontend dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit dashboard using sys.argv or default app path."""
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.stderr.write(
            "Streamlit is required to launch the dashboard.\n"
            'Install frontend dependencies with: pip install -e ".[frontend]"\n'
        )
        sys.exit(1)

    app_path = str(Path(__file__).resolve().parent / "app.py")
    sys.argv = ["streamlit", "run", app_path] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
