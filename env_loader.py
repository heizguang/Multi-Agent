"""Simple .env loader without external dependencies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """Load key-value pairs from a .env file into os.environ.

    Existing environment variables are preserved and will not be overwritten.
    """
    root = Path(__file__).resolve().parent
    path = Path(env_path)
    if not path.is_absolute():
        path = root / env_path

    loaded: Dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if not key:
            continue

        os.environ.setdefault(key, value)
        loaded[key] = value

    return loaded
