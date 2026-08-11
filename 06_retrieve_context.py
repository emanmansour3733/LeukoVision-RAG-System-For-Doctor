"""
06_retrieve_context.py

OncoRAG Clinical Knowledge Assistant - Retrieval Module.

Sole responsibility: retrieve the most relevant chunks from an existing
Chroma vector store (05_create_chroma_store.py) and prepare them for downstream
consumption. This module does NOT call an LLM, build prompts, or generate
answers - it stops at "here are the relevant chunks, formatted and ready."

Pipeline position:
    01_documents.py              -> raw Document objects
    02_preprocessing.py          -> cleaned Document objects
    03_chunking.py                -> chunked Document objects
    04_vector_representation.py  -> validated embedding_model
    05_create_chroma_store.py            -> persistent, queryable Chroma index
    06_retrieve_context.py (this module) -> retrieval, formatting, source collection

Metadata contract:
    Every retrieved chunk is expected to carry (from upstream modules):
    filename, source, page, page_label, chunk_id, chunk_index,
    total_chunks, original_filename. This module never rewrites, strips,
    or renames any metadata field on a Document - it only reads from it.

Colab compatibility:
    Pure LangChain/Python - no external services beyond the already-loaded
    Chroma vector store and embedding model, so it runs directly in a
    Colab cell with no additional setup.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    from langchain.schema import Document

try:
    from langchain_core.retrievers import BaseRetriever
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    BaseRetriever = object  # type: ignore[assignment,misc]

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    from langchain_community.vectorstores import Chroma


__all__ = [
    "RetrieverCreationError",
    "RetrievalError",
    "ContextStats",
    "create_retriever",
    "retrieve_documents",
    "retrieve_with_scores",
    "retrieve_by_metadata",
    "print_retrieved_documents",
    "format_context",
    "collect_sources",
    "estimate_context_length",
]

logger = logging.getLogger(__name__)

DEFAULT_K: int = 5
VALID_SEARCH_TYPES = frozenset({"similarity", "mmr"})
RetrievedItem = Union[Document, Tuple[Document, float]]


# --------------------------------------------------------------------------
# Custom exceptions
# --------------------------------------------------------------------------
class RetrieverCreationError(Exception):
    """Raised when a retriever cannot be constructed from a vector store."""


class RetrievalError(Exception):
    """Raised when a retrieval query against the vector store fails."""


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ContextStats:
    """Summary statistics describing a set of retrieved chunks.

    Attributes:
        character_count: Total number of characters across all chunks'
            page_content.
        word_count: Total number of whitespace-separated words across all
            chunks' page_content.
        chunk_count: Number of chunks the statistics were computed over.
    """

    character_count: int
    word_count: int
    chunk_count: int


# --------------------------------------------------------------------------
# Core functions
# --------------------------------------------------------------------------
def create_retriever(
    vector_store: Chroma,
    k: int = DEFAULT_K,
    search_type: str = "similarity",
) -> BaseRetriever:
    """Build a LangChain retriever backed by an existing Chroma vector store.

    Args:
        vector_store: A Chroma vector store instance, typically from
            05_create_chroma_store.py::create_vector_store() or
            load_vector_store().
        k: Number of chunks to retrieve per query. Defaults to 5.
        search_type: Retrieval strategy - either "similarity" (standard
            nearest-neighbor search) or "mmr" (Maximal Marginal Relevance,
            which balances relevance with result diversity). Defaults to
            "similarity".

    Returns:
        A configured LangChain retriever ready to accept queries.

    Raises:
        RetrieverCreationError: If k is not positive, search_type is not
            one of "similarity" or "mmr", or the underlying vector store
            fails to produce a retriever.
    """
    if k <= 0:
        raise RetrieverCreationError(f"k must be a positive integer, got {k}.")
    if search_type not in VALID_SEARCH_TYPES:
        raise RetrieverCreationError(
            f"search_type must be one of {sorted(VALID_SEARCH_TYPES)}, "
            f"got '{search_type}'."
        )

    try:
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )
    except Exception as exc:
        raise RetrieverCreationError(
            f"Failed to create a '{search_type}' retriever with k={k}: {exc}"
        ) from exc

    logger.info(
        "Retriever created (search_type='%s', k=%d).", search_type, k,
    )
    return retriever


def retrieve_documents(query: str, retriever: BaseRetriever) -> List[Document]:
    """Retrieve relevant chunks for a query using a pre-built retriever.

    Args:
        query: Natural language query text.
        retriever: A retriever instance, typically from create_retriever().

    Returns:
        A list of relevant Document objects (chunks), with metadata
        preserved exactly as indexed.

    Raises:
        RetrievalError: If query is empty or the underlying retrieval
            call fails.
    """
    if not query or not query.strip():
        raise RetrievalError("Query text must not be empty.")

    logger.info("Retrieving documents for query: '%s'", query)

    try:
        invoke_method = getattr(retriever, "invoke", None)
        if callable(invoke_method):
            results = invoke_method(query)
        else:  # pragma: no cover - legacy LangChain fallback
            results = retriever.get_relevant_documents(query)
    except Exception as exc:
        raise RetrievalError(
            f"Retrieval failed for query '{query}': {exc}"
        ) from exc

    logger.info("Retrieved %d chunk(s) for query: '%s'", len(results), query)
    return results


def retrieve_with_scores(
    query: str,
    vector_store: Chroma,
    k: int = DEFAULT_K,
) -> List[Tuple[Document, float]]:
    """Retrieve relevant chunks together with their similarity scores.

    Standard LangChain retrievers do not expose similarity scores, so this
    queries the vector store directly rather than going through a
    retriever object.

    Args:
        query: Natural language query text.
        vector_store: A Chroma vector store instance.
        k: Number of chunks to retrieve. Defaults to 5.

    Returns:
        A list of up to k (Document, score) tuples, ordered from most to
        least similar.

    Raises:
        RetrievalError: If query is empty, k is not positive, or the
            underlying vector store search fails.
    """
    if not query or not query.strip():
        raise RetrievalError("Query text must not be empty.")
    if k <= 0:
        raise RetrievalError(f"k must be a positive integer, got {k}.")

    logger.info("Retrieving documents with scores for query: '%s'", query)

    try:
        results = vector_store.similarity_search_with_score(query, k=k)
    except Exception as exc:
        raise RetrievalError(
            f"Retrieval with scores failed for query '{query}': {exc}"
        ) from exc

    logger.info(
        "Retrieved %d chunk(s) with scores for query: '%s'", len(results), query,
    )
    return results


def retrieve_by_metadata(
    query: str,
    vector_store: Chroma,
    metadata_filter: Dict[str, object],
    k: int = DEFAULT_K,
) -> List[Document]:
    """Retrieve relevant chunks restricted to those matching a metadata filter.

    Combines semantic similarity search with an exact-match metadata
    filter, e.g. restricting results to a single source PDF via
    {"filename": "WHO_ALL_Guidelines.pdf"}.

    Args:
        query: Natural language query text.
        vector_store: A Chroma vector store instance.
        metadata_filter: A dict of metadata field/value pairs that
            returned chunks must match exactly (passed through to
            Chroma's native filter mechanism).
        k: Number of chunks to retrieve. Defaults to 5.

    Returns:
        A list of up to k Document objects matching both the query and
        the metadata filter.

    Raises:
        RetrievalError: If query is empty, metadata_filter is empty, k is
            not positive, or the underlying vector store search fails.
    """
    if not query or not query.strip():
        raise RetrievalError("Query text must not be empty.")
    if not metadata_filter:
        raise RetrievalError("metadata_filter must be a non-empty dict.")
    if k <= 0:
        raise RetrievalError(f"k must be a positive integer, got {k}.")

    logger.info(
        "Retrieving documents for query: '%s' with metadata filter: %s",
        query, metadata_filter,
    )

    try:
        results = vector_store.similarity_search(query, k=k, filter=metadata_filter)
    except Exception as exc:
        raise RetrievalError(
            f"Metadata-filtered retrieval failed for query '{query}' with "
            f"filter {metadata_filter}: {exc}"
        ) from exc

    logger.info(
        "Retrieved %d chunk(s) for query: '%s' matching filter: %s",
        len(results), query, metadata_filter,
    )
    return results


def _split_retrieved_item(item: RetrievedItem) -> Tuple[Document, Optional[float]]:
    """Normalize a retrieved item into a (Document, optional score) pair."""
    if isinstance(item, tuple):
        document, score = item
        return document, score
    return item, None


def print_retrieved_documents(retrieved_documents: List[RetrievedItem]) -> None:
    """Pretty-print retrieved chunks for human review.

    Accepts either a plain list of Document objects (from
    retrieve_documents() / retrieve_by_metadata()) or a list of
    (Document, score) tuples (from retrieve_with_scores()); the score
    column is included automatically when available.

    For each chunk, displays: rank, similarity score (if available),
    filename, page number, chunk index, chunk ID, and the first 500
    characters of content.

    Args:
        retrieved_documents: A list of Document objects, or a list of
            (Document, score) tuples.

    Returns:
        None. Output is printed directly to stdout.
    """
    if not retrieved_documents:
        print("No documents retrieved.")
        return

    separator = "-" * 60
    for rank, item in enumerate(retrieved_documents, start=1):
        document, score = _split_retrieved_item(item)
        metadata = document.metadata

        print(separator)
        print(f"Rank          : {rank}")
        if score is not None:
            print(f"Score         : {score:.4f}")
        print(f"Filename      : {metadata.get('filename', 'unknown')}")
        print(f"Page          : {metadata.get('page', 'unknown')}")
        print(f"Chunk index   : {metadata.get('chunk_index', 'unknown')}")
        print(f"Chunk ID      : {metadata.get('chunk_id', 'unknown')}")
        print(f"Content       : {document.page_content[:500]}")
    print(separator)


def format_context(retrieved_documents: List[RetrievedItem]) -> str:
    """Format retrieved chunks into a single string ready for an LLM prompt.

    Each chunk is rendered as a "Source: <filename>\\nPage: <page>" header
    followed by its content, with chunks separated by a blank line.

    Args:
        retrieved_documents: A list of Document objects, or a list of
            (Document, score) tuples.

    Returns:
        A single formatted context string. Returns an empty string if
        retrieved_documents is empty.
    """
    if not retrieved_documents:
        return ""

    formatted_blocks: List[str] = []
    for item in retrieved_documents:
        document, _ = _split_retrieved_item(item)
        metadata = document.metadata
        filename = metadata.get("filename", "unknown")
        page = metadata.get("page", "unknown")
        block = f"Source: {filename}\nPage: {page}\n\n{document.page_content}"
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


def collect_sources(retrieved_documents: List[RetrievedItem]) -> List[str]:
    """Collect a unique, order-preserving list of human-readable sources.

    Each source is rendered as "<filename> (Page <page>)". Duplicate
    filename/page combinations (e.g. multiple chunks from the same page)
    are collapsed to a single entry.

    Args:
        retrieved_documents: A list of Document objects, or a list of
            (Document, score) tuples.

    Returns:
        A list of unique source strings, in first-seen order.
    """
    seen: set = set()
    sources: List[str] = []

    for item in retrieved_documents:
        document, _ = _split_retrieved_item(item)
        metadata = document.metadata
        filename = metadata.get("filename", "unknown")
        page = metadata.get("page", "unknown")
        source_label = f"{filename} (Page {page})"

        if source_label not in seen:
            seen.add(source_label)
            sources.append(source_label)

    return sources


def estimate_context_length(retrieved_documents: List[RetrievedItem]) -> ContextStats:
    """Estimate the size of a retrieved-chunk context prior to LLM submission.

    Args:
        retrieved_documents: A list of Document objects, or a list of
            (Document, score) tuples.

    Returns:
        A ContextStats instance with character_count, word_count, and
        chunk_count computed across all retrieved chunks' page_content.
    """
    character_count = 0
    word_count = 0
    chunk_count = 0

    for item in retrieved_documents:
        document, _ = _split_retrieved_item(item)
        content = document.page_content or ""
        character_count += len(content)
        word_count += len(content.split())
        chunk_count += 1

    return ContextStats(
        character_count=character_count,
        word_count=word_count,
        chunk_count=chunk_count,
    )