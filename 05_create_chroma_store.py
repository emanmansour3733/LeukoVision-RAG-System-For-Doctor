"""
05_create_chroma_store.py

OncoRAG Clinical Knowledge Assistant - Vector Store Module.

Sole responsibility: create, persist, load, query, and manage a ChromaDB
vector database via LangChain. This module does NOT implement retrieval
strategy, prompt construction, or LLM-based answer generation - it is a
thin, focused data-layer wrapper around Chroma.

Pipeline position:
    01_documents.py          -> raw Document objects
    02_preprocessing.py      -> cleaned Document objects
    03_chunking.py           -> chunked Document objects (chunks)
    04_vector_representation.py -> validated embedding_model
    05_create_chroma_store.py (this module) -> persistent, queryable Chroma index

Metadata contract:
    Every chunk indexed here is expected to already carry (from upstream
    modules): filename, source, page, page_label, chunk_id, chunk_index,
    total_chunks, original_filename. This module never rewrites, strips,
    or renames any metadata field - Documents are passed to Chroma exactly
    as received. Note that Chroma requires metadata values to be simple
    scalars (str, int, float, bool); if an upstream Document carries an
    incompatible value (e.g. None or a nested object), Chroma itself will
    raise on insertion and that failure is surfaced via
    VectorStoreCreationError rather than being silently patched here.

Colab compatibility:
    Uses a local, on-disk persistent directory (default: 'vector_database')
    which works out of the box in a Colab runtime/session, including
    across reconnects within the same session as long as the directory
    persists on the runtime's filesystem (or a mounted Google Drive path
    is passed as persist_directory).
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError as import_error:
        raise ImportError(
            "Chroma could not be imported. Install it with: "
            "pip install langchain-chroma chromadb"
        ) from import_error

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    from langchain.schema import Document

try:
    from langchain_core.embeddings import Embeddings
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    Embeddings = object  # type: ignore[assignment,misc]


__all__ = [
    "VectorStoreCreationError",
    "VectorStoreLoadError",
    "VectorStoreSearchError",
    "create_vector_store",
    "load_vector_store",
    "similarity_search",
    "similarity_search_with_scores",
    "count_documents",
    "delete_vector_store",
    "database_exists",
    "print_database_statistics",
]

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
DEFAULT_PERSIST_DIRECTORY: str = "vector_database"


# --------------------------------------------------------------------------
# Custom exceptions
# --------------------------------------------------------------------------
class VectorStoreCreationError(Exception):
    """Raised when a vector store cannot be created, persisted, or deleted."""


class VectorStoreLoadError(Exception):
    """Raised when an existing vector store cannot be loaded from disk."""


class VectorStoreSearchError(Exception):
    """Raised when a query, count, or statistics read against the vector store fails."""


# --------------------------------------------------------------------------
# Core functions
# --------------------------------------------------------------------------
def create_vector_store(
    chunks: List[Document],
    embedding_model: Embeddings,
    persist_directory: PathLike = DEFAULT_PERSIST_DIRECTORY,
) -> Chroma:
    """Create a new persistent Chroma vector store from a list of chunks.

    Embeds every chunk with the supplied embedding_model and writes the
    resulting index to disk at persist_directory. Metadata on each chunk
    (filename, source, page, page_label, chunk_id, chunk_index,
    total_chunks, original_filename) is passed through to Chroma
    unmodified.

    Args:
        chunks: List of chunked Document objects, typically the output of
            03_chunking.py::chunk_documents().
        embedding_model: A validated embeddings instance, typically the
            output of 04_vector_representation.py::create_embedding_model().
        persist_directory: Filesystem path where the Chroma database will
            be persisted. Defaults to 'vector_database'.

    Returns:
        A ready-to-use Chroma vector store instance backed by the newly
        created persistent database.

    Raises:
        VectorStoreCreationError: If chunks is empty, or if Chroma fails
            to embed/index/persist the chunks (e.g. incompatible metadata
            types, disk write failure, embedding model error).
    """
    if not chunks:
        raise VectorStoreCreationError(
            "Cannot create a vector store from an empty list of chunks."
        )

    persist_path = Path(persist_directory)
    logger.info(
        "Creating vector store at '%s' from %d chunk(s)...",
        persist_path, len(chunks),
    )

    try:
        persist_path.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(persist_path),
        )
    except Exception as exc:
        raise VectorStoreCreationError(
            f"Failed to create vector store at '{persist_path}' from "
            f"{len(chunks)} chunk(s): {exc}"
        ) from exc

    # Some LangChain/Chroma versions still expose an explicit persist()
    # call; newer chromadb backends auto-persist on write. Call it if
    # present, but never fail creation over a missing/deprecated method.
    persist_method = getattr(vector_store, "persist", None)
    if callable(persist_method):
        try:
            persist_method()
        except Exception as persist_exc:
            logger.debug(
                "Explicit persist() call skipped (likely auto-persisted "
                "by the current chromadb backend): %s", persist_exc,
            )

    indexed_count = count_documents(vector_store)
    logger.info(
        "Vector store created at '%s': %d chunk(s) indexed.",
        persist_path, indexed_count,
    )
    return vector_store


def load_vector_store(
    persist_directory: PathLike = DEFAULT_PERSIST_DIRECTORY,
    embedding_model: Optional[Embeddings] = None,
) -> Chroma:
    """Load an existing persistent Chroma vector store from disk.

    Args:
        persist_directory: Filesystem path of a previously created Chroma
            database. Defaults to 'vector_database'.
        embedding_model: A validated embeddings instance matching the one
            used at creation time. Required so future queries against the
            loaded store can be embedded consistently.

    Returns:
        A Chroma vector store instance backed by the existing persistent
        database at persist_directory.

    Raises:
        VectorStoreLoadError: If embedding_model is not provided, no
            database exists at persist_directory, or Chroma fails to open
            the existing database (e.g. corrupted files, version mismatch).
    """
    if embedding_model is None:
        raise VectorStoreLoadError(
            "An embedding_model must be provided to load an existing "
            "vector store, since Chroma needs it to embed future queries."
        )

    persist_path = Path(persist_directory)
    if not database_exists(persist_path):
        raise VectorStoreLoadError(
            f"No persistent vector store found at '{persist_path}'. "
            "Create one first with create_vector_store()."
        )

    logger.info("Loading vector store from '%s'...", persist_path)

    try:
        vector_store = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embedding_model,
        )
    except Exception as exc:
        raise VectorStoreLoadError(
            f"Failed to load vector store from '{persist_path}': {exc}"
        ) from exc

    indexed_count = count_documents(vector_store)
    if indexed_count == 0:
        logger.warning(
            "Vector store loaded from '%s' but contains 0 indexed chunk(s).",
            persist_path,
        )
    logger.info(
        "Vector store loaded from '%s': %d chunk(s) available.",
        persist_path, indexed_count,
    )
    return vector_store


def similarity_search(
    query: str,
    vector_store: Chroma,
    k: int = 5,
) -> List[Document]:
    """Retrieve the top-k most similar chunks to a query string.

    Args:
        query: Natural language query text.
        vector_store: A Chroma vector store instance, typically from
            create_vector_store() or load_vector_store().
        k: Number of top results to retrieve. Defaults to 5.

    Returns:
        A list of up to k Document objects (chunks), ordered from most to
        least similar, with metadata preserved exactly as indexed.

    Raises:
        VectorStoreSearchError: If query is empty, k is not positive, or
            the underlying Chroma search fails.
    """
    if not query or not query.strip():
        raise VectorStoreSearchError("Query text must not be empty.")
    if k <= 0:
        raise VectorStoreSearchError(f"k must be a positive integer, got {k}.")

    try:
        results = vector_store.similarity_search(query, k=k)
    except Exception as exc:
        raise VectorStoreSearchError(
            f"Similarity search failed for query '{query}': {exc}"
        ) from exc

    logger.info(
        "Similarity search retrieved %d chunk(s) for query (k=%d).",
        len(results), k,
    )
    return results


def similarity_search_with_scores(
    query: str,
    vector_store: Chroma,
    k: int = 5,
) -> List[Tuple[Document, float]]:
    """Retrieve the top-k most similar chunks with their similarity scores.

    Args:
        query: Natural language query text.
        vector_store: A Chroma vector store instance, typically from
            create_vector_store() or load_vector_store().
        k: Number of top results to retrieve. Defaults to 5.

    Returns:
        A list of up to k (Document, score) tuples, ordered from most to
        least similar. Score semantics (distance vs. similarity) follow
        the underlying Chroma backend's convention - lower generally means
        more similar for the default distance metric.

    Raises:
        VectorStoreSearchError: If query is empty, k is not positive, or
            the underlying Chroma search fails.
    """
    if not query or not query.strip():
        raise VectorStoreSearchError("Query text must not be empty.")
    if k <= 0:
        raise VectorStoreSearchError(f"k must be a positive integer, got {k}.")

    try:
        results = vector_store.similarity_search_with_score(query, k=k)
    except Exception as exc:
        raise VectorStoreSearchError(
            f"Similarity search with scores failed for query '{query}': {exc}"
        ) from exc

    logger.info(
        "Similarity search with scores retrieved %d chunk(s) for query (k=%d).",
        len(results), k,
    )
    return results


def count_documents(vector_store: Chroma) -> int:
    """Count the total number of chunks currently indexed in a vector store.

    Args:
        vector_store: A Chroma vector store instance.

    Returns:
        The total number of indexed chunks.

    Raises:
        VectorStoreSearchError: If the count cannot be read from the
            underlying Chroma collection.
    """
    try:
        collection = getattr(vector_store, "_collection", None)
        if collection is not None:
            return int(collection.count())
        return len(vector_store.get().get("ids", []))
    except Exception as exc:
        raise VectorStoreSearchError(
            f"Failed to count documents in vector store: {exc}"
        ) from exc


def delete_vector_store(persist_directory: PathLike = DEFAULT_PERSIST_DIRECTORY) -> None:
    """Permanently delete a persistent Chroma database from disk.

    Args:
        persist_directory: Filesystem path of the database to delete.
            Defaults to 'vector_database'.

    Raises:
        VectorStoreCreationError: If the directory exists but cannot be
            removed (e.g. permission error, file lock).
    """
    persist_path = Path(persist_directory)

    if not persist_path.exists():
        logger.warning(
            "Nothing to delete: '%s' does not exist.", persist_path,
        )
        return

    try:
        shutil.rmtree(persist_path)
    except Exception as exc:
        raise VectorStoreCreationError(
            f"Failed to delete vector store at '{persist_path}': {exc}"
        ) from exc

    logger.info("Vector store at '%s' deleted.", persist_path)


def database_exists(persist_directory: PathLike = DEFAULT_PERSIST_DIRECTORY) -> bool:
    """Check whether a persistent vector store exists at a given path.

    Args:
        persist_directory: Filesystem path to check. Defaults to
            'vector_database'.

    Returns:
        True if persist_directory exists, is a directory, and contains at
        least one file/subdirectory (i.e. a non-empty Chroma database);
        False otherwise.
    """
    persist_path = Path(persist_directory)
    if not persist_path.exists() or not persist_path.is_dir():
        return False
    return any(persist_path.iterdir())


def print_database_statistics(
    vector_store: Chroma,
    persist_directory: PathLike,
    embedding_model_name: str,
) -> None:
    """Print a human-readable summary of the vector store's contents.

    Reports: number of indexed chunks, number of unique source PDFs,
    average chunks per document, embedding model name, and the resolved
    persist directory.

    Args:
        vector_store: A Chroma vector store instance.
        persist_directory: Filesystem path where the database is persisted.
        embedding_model_name: Name/identifier of the embedding model used
            to build the store (for display purposes only).

    Raises:
        VectorStoreSearchError: If chunk metadata cannot be read from the
            vector store.
    """
    try:
        raw = vector_store.get(include=["metadatas"])
        metadatas = raw.get("metadatas", []) or []
    except Exception as exc:
        raise VectorStoreSearchError(
            f"Failed to retrieve metadata for statistics: {exc}"
        ) from exc

    total_chunks = len(metadatas)
    unique_sources = {
        metadata.get("original_filename") or metadata.get("filename") or "unknown"
        for metadata in metadatas
    }
    unique_count = len(unique_sources)
    average_chunks_per_document = (
        total_chunks / unique_count if unique_count else 0.0
    )

    separator = "=" * 60
    print(separator)
    print("OncoRAG Vector Store Statistics")
    print(separator)
    print(f"Number of indexed chunks : {total_chunks}")
    print(f"Unique source PDFs       : {unique_count}")
    print(f"Average chunks per doc   : {average_chunks_per_document:.2f}")
    print(f"Embedding model          : {embedding_model_name}")
    print(f"Persist directory        : {Path(persist_directory).resolve()}")
    print(separator)

    logger.info(
        "Statistics: %d chunk(s), %d unique source PDF(s), "
        "%.2f avg chunk(s)/doc.",
        total_chunks, unique_count, average_chunks_per_document,
    )