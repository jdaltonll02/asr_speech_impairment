"""
Single entry point for all configuration in the ASR-sprint project.

Load the central YAML once with ``load_config()``.  Every module should call
``from src.utils.config import load_config`` and obtain a frozen namespace via
``cfg = load_config()``.  Attribute access is supported at every nesting level
(``cfg.models.ky.language_token``) thanks to the recursive ``_DotDict``
wrapper.

Resolution order for the config file path:
1. The ``path`` argument passed to ``load_config()``.
2. The ``CONFIG_PATH`` environment variable.
3. ``<repo_root>/config/base.yaml`` where *repo_root* is two levels above this
   file (``src/utils/config.py`` → ``src/utils`` → ``src`` → repo root).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _DotDict:
    """Recursive dot-access wrapper around a plain dict loaded from YAML."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            setattr(self, key, _DotDict(value) if isinstance(value, dict) else value)

    def __repr__(self) -> str:
        return f"_DotDict({self.__dict__!r})"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


# ---------------------------------------------------------------------------
# Module-level cache so repeated calls are free
# ---------------------------------------------------------------------------
_cfg: _DotDict | None = None


def _default_config_path() -> Path:
    """Return the default config path: <repo_root>/config/base.yaml."""
    return Path(__file__).parents[2] / "config" / "base.yaml"


def load_config(path: str | None = None) -> _DotDict:
    """Load and return the project configuration as a dot-accessible namespace.

    Parameters
    ----------
    path:
        Explicit path to a YAML config file.  If *None*, the function checks
        the ``CONFIG_PATH`` environment variable, then falls back to
        ``<repo_root>/config/base.yaml``.

    Returns
    -------
    _DotDict
        Frozen (by convention) dot-accessible config object.  Sub-dicts are
        also ``_DotDict`` instances, so callers can write
        ``cfg.models.ky.language_token``.
    """
    global _cfg

    if _cfg is not None and path is None:
        return _cfg

    if path is None:
        path = os.environ.get("CONFIG_PATH", None)
    if path is None:
        path = str(_default_config_path())

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Set CONFIG_PATH env var or pass path= explicitly."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    _cfg = _DotDict(raw)
    return _cfg


def reset_config() -> None:
    """Clear the cached config (useful in tests)."""
    global _cfg
    _cfg = None
