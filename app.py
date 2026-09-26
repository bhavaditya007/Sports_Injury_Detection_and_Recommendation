"""Streamlit entry point for the synthetic injury-label classification demo."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from injury_demo.streamlit_app import render_app  # noqa: E402


if __name__ == "__main__":
    render_app()
