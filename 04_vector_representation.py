"""
04_vector_representation.py

OncoRAG Clinical Knowledge Assistant - Vector Representation Module.

Sole responsibility: construct and validate the embedding model used to
convert chunk text (from 03_chunking.py) into dense vector representations.
This module does NOT perform embedding of documents itself - it only
builds and sanity-checks a ready-to-use HuggingFaceEmbeddings instance for
downstream modules (vector store indexing, retrieval) to consume.

Model: sentence-transformers/all-MiniLM-L6-v2
    - 384-dimensional embeddings. Chosen over larger models (e.g.
      all-mpnet-base-v2) for CPU-only, memory-constrained deployments
      (e.g. Streamlit Community Cloud's free tier), where a smaller model
      meaningfully reduces RAM pressure and encoding latency at a modest
      cost to embedding quality.

Colab compatibility:
    - Auto-detects and uses a GPU (CUDA) if available, falling back to CPU
      without requiring any manual configuration.
    - The underlying model is downloaded from the Hugging Face Hub on first
      call (requires 'sentence-transformers' and internet access); Colab
      runtimes provide both by default.
"""

import logging
from typing import Optional

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as import_error:
        raise ImportError(
            "HuggingFaceEmbeddings could not be imported. Install it with: "
            "pip install langchain-huggingface sentence-transformers"
        ) from import_error


__all__ = ["EmbeddingModelLoadError", "create_embedding_model"]

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_VALIDATION_TEXT = "Stage III non-small cell lung carcinoma validation probe."


class EmbeddingModelLoadError(Exception):
    """Raised when the embedding model fails to load or fails validation."""


def _resolve_device(device: Optional[str]) -> str:
    """
    Resolve which device the embedding model should run on.

    If an explicit device is provided, it is used as-is. Otherwise, this
    attempts to detect a CUDA-capable GPU (common in Colab runtimes) and
    falls back to CPU if torch is unavailable or no GPU is present.

    Args:
        device: An explicit device string ('cuda', 'cpu', etc.), or None
            to auto-detect.

    Returns:
        The resolved device string.
    """
    if device is not None:
        return device

    try:
        import torch
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        logger.warning("torch not available for device detection; defaulting to CPU.")
        resolved = "cpu"

    logger.info("Auto-detected embedding device: '%s'.", resolved)
    return resolved


def _validate_embedding_model(embedding_model: HuggingFaceEmbeddings) -> int:
    """
    Sanity-check a constructed embedding model by embedding a probe string.

    Confirms the model actually produces a non-empty, numeric vector
    before it's handed off to downstream indexing/retrieval code - so
    failures surface immediately at setup time rather than deep inside a
    later pipeline stage.

    Args:
        embedding_model: The HuggingFaceEmbeddings instance to validate.

    Returns:
        The dimensionality of the produced embedding vector.

    Raises:
        EmbeddingModelLoadError: If embedding the probe string fails, or
            the result is empty or not a numeric vector.
    """
    try:
        vector = embedding_model.embed_query(_VALIDATION_TEXT)
    except Exception as exc:
        raise EmbeddingModelLoadError(
            "Embedding model loaded but failed to produce an embedding "
            f"for a validation probe: {exc}"
        ) from exc

    if not vector or not isinstance(vector, (list, tuple)):
        raise EmbeddingModelLoadError(
            "Embedding model returned an empty or invalid vector during "
            "validation."
        )
    if not all(isinstance(value, (int, float)) for value in vector):
        raise EmbeddingModelLoadError(
            "Embedding model returned a vector with non-numeric values "
            "during validation."
        )

    return len(vector)


def create_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
) -> HuggingFaceEmbeddings:
    """
    Create and validate a HuggingFaceEmbeddings model instance.

    Builds a sentence-transformers-backed embedding model, running on GPU
    automatically when available (e.g. in a Colab GPU runtime) and on CPU
    otherwise, then immediately validates it by embedding a short probe
    string so any loading/runtime problem is caught here rather than
    later in the pipeline.

    Args:
        model_name: Hugging Face Hub model identifier to load. Defaults
            to 'sentence-transformers/all-mpnet-base-v2'.
        device: Explicit device to run the model on ('cuda' or 'cpu').
            If None, the device is auto-detected.
        normalize_embeddings: Whether to L2-normalize output embeddings,
            recommended for cosine-similarity-based vector search.

    Returns:
        A validated, ready-to-use HuggingFaceEmbeddings instance.

    Raises:
        EmbeddingModelLoadError: If the model cannot be downloaded,
            instantiated, or fails validation (e.g. missing dependencies,
            no internet access, corrupted cache, invalid model name).
    """
    resolved_device = _resolve_device(device)

    logger.info(
        "Loading embedding model '%s' on device '%s' "
        "(normalize_embeddings=%s)...",
        model_name, resolved_device, normalize_embeddings,
    )

    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": resolved_device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )
    except Exception as exc:
        raise EmbeddingModelLoadError(
            f"Failed to load embedding model '{model_name}' on device "
            f"'{resolved_device}'. Ensure 'sentence-transformers' is "
            "installed and that you have internet access to download the "
            f"model from the Hugging Face Hub. Original error: {exc}"
        ) from exc

    embedding_dimension = _validate_embedding_model(embedding_model)

    logger.info(
        "Embedding model '%s' loaded and validated successfully "
        "(dimension=%d, device='%s').",
        model_name, embedding_dimension, resolved_device,
    )
    return embedding_model
