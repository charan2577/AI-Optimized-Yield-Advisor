"""
Tests to verify Streamlit application entry points and page loading.
"""

import importlib.util
from pathlib import Path
import pytest
import config


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    return module


def test_app_py_syntax():
    """Verify app.py compiles cleanly."""
    with open(config.PROJECT_ROOT / "app.py", "r", encoding="utf-8") as f:
        compile(f.read(), "app.py", "exec")


@pytest.mark.parametrize("page_file", sorted((config.PROJECT_ROOT / "pages").glob("*.py")))
def test_pages_syntax(page_file):
    """Verify all Streamlit pages compile cleanly without syntax errors."""
    with open(page_file, "r", encoding="utf-8") as f:
        compile(f.read(), str(page_file), "exec")
