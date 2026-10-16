"""
BGE-M3 embedder helpers.

This module lazily loads `BGEM3FlagModel` and keeps it as a singleton so the
rest of the RAG pipeline can request dense+sparse embeddings without paying
startup cost repeatedly.

It also hardens model initialization for Windows:
- uses the correct `devices=` argument expected by current FlagEmbedding
- sets a stable ASCII cache directory outside the OneDrive/Korean workspace path
- raises a structured error that callers can downgrade to lexical-only mode
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel  # noqa: F401

_log = logging.getLogger(__name__)

_MODEL_NAME = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
_DEVICE = os.getenv("BGE_DEVICE", "cpu").strip() or "cpu"
_BATCH_SIZE = int(os.getenv("BGE_BATCH_SIZE", "8"))

_CACHE_ROOT = Path(
    os.getenv(
        "BGE_CACHE_DIR",
        str(Path(os.getenv("LOCALAPPDATA", str(Path.home() / ".cache"))) / "semiconductor_agent" / "huggingface"),
    )
)
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_ROOT / "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_model: "BGEM3FlagModel | None" = None
_model_load_error: Exception | None = None


class EmbedderUnavailableError(RuntimeError):
    """Raised when BGE-M3 cannot be initialized in the current runtime."""


def _load_model() -> "BGEM3FlagModel":
    from FlagEmbedding import BGEM3FlagModel

    primary_kwargs = {
        "model_name_or_path": _MODEL_NAME,
        "use_fp16": (_DEVICE != "cpu"),
        "devices": _DEVICE,
        "cache_dir": str(_CACHE_ROOT),
    }
    fallback_kwargs = {
        "model_name_or_path": _MODEL_NAME,
        "use_fp16": False,
        "devices": "cpu",
        "cache_dir": str(_CACHE_ROOT),
    }

    attempts = [primary_kwargs]
    if fallback_kwargs != primary_kwargs:
        attempts.append(fallback_kwargs)

    last_error: Exception | None = None
    for attempt_idx, kwargs in enumerate(attempts, start=1):
        try:
            _log.info(
                "Loading BGE-M3 model '%s' | device=%s | cache_dir=%s | attempt=%d",
                kwargs["model_name_or_path"],
                kwargs["devices"],
                kwargs["cache_dir"],
                attempt_idx,
            )
            model = BGEM3FlagModel(**kwargs)
            _log.info("BGE-M3 loaded successfully on %s.", kwargs["devices"])
            return model
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _log.warning(
                "BGE-M3 load attempt %d failed | device=%s | error=%s: %s",
                attempt_idx,
                kwargs["devices"],
                type(exc).__name__,
                exc,
            )

    assert last_error is not None
    raise EmbedderUnavailableError(
        f"Failed to initialize BGE-M3 ({_MODEL_NAME}) on device={_DEVICE}. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


def get_embedder() -> "BGEM3FlagModel":
    """Return the singleton BGE-M3 model, loading it on first use."""
    global _model, _model_load_error
    if _model is not None:
        return _model
    if _model_load_error is not None:
        raise EmbedderUnavailableError(str(_model_load_error)) from _model_load_error

    try:
        _model = _load_model()
        return _model
    except Exception as exc:  # noqa: BLE001
        _model_load_error = exc
        raise


def encode_batch(texts: list[str]) -> list[tuple[list[float], dict[int, float]]]:
    """
    Encode a batch of texts into dense+sparse vectors.

    Returns:
        list[(dense_vector, sparse_token_weight_dict)]
    """
    if not texts:
        return []

    model = get_embedder()
    out = model.encode(
        texts,
        batch_size=_BATCH_SIZE,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vecs = out["dense_vecs"]
    lexical_weights = out["lexical_weights"]

    results: list[tuple[list[float], dict[int, float]]] = []
    for dense_vec, sparse_raw in zip(dense_vecs, lexical_weights, strict=False):
        dense = dense_vec.tolist()
        sparse = {int(k): float(v) for k, v in sparse_raw.items() if float(v) > 1e-5}
        results.append((dense, sparse))
    return results


def encode_single(text: str) -> tuple[list[float], dict[int, float]]:
    """Encode a single text into dense+sparse vectors."""
    return encode_batch([text])[0]


def encode_query(query: str) -> tuple[list[float], dict[int, float]]:
    """
    Encode a retrieval query.

    Some deployments like to prepend a query instruction string. We keep it
    configurable while defaulting to the raw query.
    """
    prefix = os.getenv("BGE_QUERY_PREFIX", "")
    text = f"{prefix}{query}" if prefix else query
    return encode_single(text)
