from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "data" / "semiconductor_taxonomy.yaml"


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, Any]:
    if not _TAXONOMY_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(_TAXONOMY_PATH.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def taxonomy_section(name: str, default: Any = None) -> Any:
    data = load_taxonomy()
    section = data.get(name)
    if section is None:
        return default
    return section


def taxonomy_collection(default: dict[str, Any] | None = None) -> dict[str, Any]:
    section = taxonomy_section("collection", default or {})
    return section if isinstance(section, dict) else (default or {})


def taxonomy_analysis(default: dict[str, Any] | None = None) -> dict[str, Any]:
    section = taxonomy_section("analysis", default or {})
    return section if isinstance(section, dict) else (default or {})
