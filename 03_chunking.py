"""
03_chunking.py

OncoRAG Clinical Knowledge Assistant - Chunking Module.

Consumes the cleaned LangChain Document objects produced by
02_preprocessing.py::preprocess_documents() and splits each one into
semantic chunks suitable for embedding, using LangChain's
RecursiveCharacterTextSplitter. All original metadata (filename, source,
page, etc.) is preserved on every resulting chunk, plus new chunk-level
metadata for retrieval traceability.
"""

import logging
import uuid
from typing import List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    from langchain.schema import Document


__all__ = [
    "create_text_splitter",
    "split_document",
    "chunk_documents",
]

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter configured for clinical text.

    RecursiveCharacterTextSplitter tries a prioritized list of separators
    (paragraph, then line, then sentence-ish punctuation, then word) so
    splits fall on natural semantic boundaries wherever possible, rather
    than cutting mid-sentence or mid-term.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters carried between
            consecutive chunks, to preserve context across chunk
            boundaries.

    Returns:
        A configured RecursiveCharacterTextSplitter instance.

    Raises:
        ValueError: If chunk_overlap is negative or not smaller than
            chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_document(
    document: Document,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """
    Split a single Document's page_content into chunk-level Documents.

    Every resulting chunk carries a full copy of the original document's
    metadata, plus:
        - chunk_id: a globally unique identifier (uuid4 hex) for this chunk.
        - chunk_index: 0-indexed position of this chunk within its source document.
        - total_chunks: total number of chunks produced from this document.
        - original_filename: the source document's filename, mirrored from
          the 'filename' metadata for convenient access at the chunk level.

    Args:
        document: A cleaned Document (typically one page) to split.
        splitter: A configured RecursiveCharacterTextSplitter.

    Returns:
        A list of chunk-level Document objects. Returns an empty list if
        the document has no content to split.
    """
    content = document.page_content or ""
    if not content.strip():
        logger.warning(
            "Skipping document with empty content (metadata: %s).",
            document.metadata,
        )
        return []

    text_chunks = splitter.split_text(content)
    total_chunks = len(text_chunks)
    original_filename = document.metadata.get("filename", "unknown")

    chunk_documents_list: List[Document] = []
    for chunk_index, chunk_text in enumerate(text_chunks):
        chunk_metadata = dict(document.metadata)
        chunk_metadata.update(
            {
                "chunk_id": f"{original_filename}_{document.metadata.get('page')}_{chunk_index}",
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "original_filename": original_filename,
            }
        )
        chunk_documents_list.append(
            Document(page_content=chunk_text, metadata=chunk_metadata)
        )

    logger.info(
        "Split document '%s' (page %s) into %d chunk(s).",
        original_filename,
        document.metadata.get("page", "?"),
        total_chunks,
    )
    return chunk_documents_list


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Orchestrator: split every Document in a list into semantic chunks.

    Builds a single shared splitter (avoids reconstructing it per
    document) and applies it across all input documents. Fail-soft: a
    single document that fails to split is logged and skipped rather
    than aborting the entire run.

    Args:
        documents: List of cleaned Document objects, typically the output
            of 02_preprocessing.py::preprocess_documents().
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A flat list of chunk-level Document objects across all input
        documents, each with original metadata preserved plus chunk_id,
        chunk_index, total_chunks, and original_filename.
    """
    splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks: List[Document] = []
    failed_count = 0

    for index, document in enumerate(documents):
        try:
            all_chunks.extend(split_document(document, splitter))
        except Exception as exc:
            filename = document.metadata.get("filename", "unknown") if hasattr(
                document, "metadata"
            ) else "unknown"
            logger.warning(
                "Skipping document at index %d (filename='%s') due to "
                "chunking error: %s", index, filename, exc,
            )
            failed_count += 1

    if failed_count:
        logger.warning(
            "%d document(s) failed to chunk and were skipped out of %d total.",
            failed_count,
            len(documents),
        )

    logger.info(
        "Chunking complete: %d document(s) produced %d chunk(s) total.",
        len(documents) - failed_count,
        len(all_chunks),
    )
    return all_chunks