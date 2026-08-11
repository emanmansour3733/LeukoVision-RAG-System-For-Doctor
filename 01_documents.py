"""
01_documents.py

OncoRAG Clinical Knowledge Assistant - Document Ingestion Module.

Responsible for reading every PDF inside the `documents/` folder using
LangChain's PyPDFLoader and returning LangChain Document objects, each
carrying traceable metadata (filename, page number, source path).

This module is designed to fail-soft: a single missing, empty, or corrupted
PDF must never crash the entire ingestion run. Problems are logged and the
pipeline continues with whatever documents were successfully loaded.

Two loading APIs are provided:
    - load_all_documents(...)  -> List[Document]   (simple, in-memory)
    - iter_all_documents(...)  -> Iterator[Document] (memory-efficient,
      streams one file's pages at a time; preferred for large corpora)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError as import_error:  # pragma: no cover
    raise ImportError(
        "PyPDFLoader could not be imported. Install it with: "
        "pip install langchain-community pypdf"
    ) from import_error

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    from langchain.schema import Document


__all__ = [
    "DocumentsFolderNotFoundError",
    "EmptyDocumentsFolderError",
    "NoDocumentsLoadedError",
    "validate_documents_folder",
    "get_pdf_files",
    "load_single_pdf",
    "load_all_documents",
    "iter_all_documents",
    "summarize_documents",
]

PathLike = Union[str, Path]

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Custom exceptions
# --------------------------------------------------------------------------
class DocumentsFolderNotFoundError(FileNotFoundError):
    """Raised when the documents folder does not exist on disk."""


class EmptyDocumentsFolderError(Exception):
    """Raised when the documents folder exists but contains no PDF files."""


class NoDocumentsLoadedError(Exception):
    """Raised when every PDF in the folder failed to load."""


# --------------------------------------------------------------------------
# Core functions
# --------------------------------------------------------------------------
def validate_documents_folder(documents_dir: Path) -> None:
    """
    Validate that the documents folder exists and is a directory.

    Args:
        documents_dir: Path to the folder expected to contain PDF files.

    Raises:
        DocumentsFolderNotFoundError: If the folder does not exist.
        NotADirectoryError: If the path exists but is not a directory.
    """
    if not documents_dir.exists():
        raise DocumentsFolderNotFoundError(
            f"Documents folder not found: '{documents_dir}'. "
            f"Create the folder and add your medical PDFs before running ingestion."
        )
    if not documents_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but found a file at: '{documents_dir}'."
        )


def get_pdf_files(documents_dir: Path) -> List[Path]:
    """
    Collect all PDF file paths inside the documents folder.

    Matching is case-insensitive ('*.pdf' and '*.PDF' both count), which
    matters on case-sensitive filesystems (Linux) where mixed-case
    extensions are common in files sourced from different origins.

    Args:
        documents_dir: Validated path to the documents folder.

    Returns:
        A sorted, de-duplicated list of Path objects pointing to PDF files.

    Raises:
        EmptyDocumentsFolderError: If no PDF files are found.
    """
    matches = {p.resolve() for p in documents_dir.iterdir()
               if p.is_file() and p.suffix.lower() == ".pdf"}
    pdf_files = sorted(matches, key=lambda p: p.name.lower())

    if not pdf_files:
        raise EmptyDocumentsFolderError(
            f"No PDF files found in '{documents_dir}'. "
            f"Add PDFs (ALL, AML, WHO Classification, CBC, etc.) before running ingestion."
        )
    return pdf_files


def attach_custom_metadata(pages: List[Document], file_path: Path) -> List[Document]:
    """
    Enrich each page-level Document with standardized, traceable metadata.

    Ensures every Document carries:
        - filename: the PDF's file name (for display/citation)
        - source: the full file path (for traceability/debugging)
        - page: a 1-indexed page number (human-readable, PyPDFLoader is 0-indexed)

    Args:
        pages: Raw Document objects returned by PyPDFLoader for one PDF.
        file_path: Path of the source PDF these pages came from.

    Returns:
        The same list of Document objects with metadata updated in place.
    """
    for index, page in enumerate(pages):
        original_page_number = page.metadata.get("page", index)
        page.metadata["filename"] = file_path.name
        page.metadata["source"] = str(file_path)
        page.metadata["page"] = original_page_number + 1
    return pages


def load_single_pdf(file_path: Path) -> List[Document]:
    """
    Load a single PDF file into a list of page-level Document objects.

    Any failure (corrupted file, unsupported encoding, encrypted PDF, etc.)
    is caught, logged as a warning, and results in an empty list rather than
    raising - so the caller can skip this file and continue with the rest.

    Args:
        file_path: Path to the PDF file to load.

    Returns:
        A list of Document objects (one per page), or an empty list if the
        file could not be read or contained no extractable pages.
    """
    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
    except Exception as exc:
        logger.warning("Skipping unreadable/corrupted PDF '%s': %s", file_path.name, exc)
        return []

    if not pages:
        logger.warning("No extractable pages found in '%s'; skipping.", file_path.name)
        return []

    pages = attach_custom_metadata(pages, file_path)
    logger.info("Loaded %d page(s) from '%s'.", len(pages), file_path.name)
    return pages


def _resolve_pdf_files(documents_dir: PathLike) -> List[Path]:
    """Shared validation/discovery step used by both loading APIs."""
    dir_path = Path(documents_dir)
    validate_documents_folder(dir_path)
    pdf_files = get_pdf_files(dir_path)
    logger.info("Found %d PDF file(s) in '%s'.", len(pdf_files), dir_path)
    return pdf_files


def iter_all_documents(
    documents_dir: PathLike = "documents",
) -> Iterator[Document]:
    """
    Memory-efficient generator: yields Document objects one page at a time
    as each PDF is processed, instead of materializing the entire corpus
    in memory at once. Preferred for large document sets that will be
    streamed straight into a chunker/embedder.

    Note: since NoDocumentsLoadedError depends on knowing whether *any*
    document loaded across the whole run, that check cannot be enforced by
    a generator (it doesn't know until exhausted). Callers who need that
    guarantee should use load_all_documents instead, or track yielded
    count themselves.

    Args:
        documents_dir: Path (as a string or Path) to the folder containing PDFs.

    Yields:
        Document objects with 'filename', 'source', and 'page' metadata.

    Raises:
        DocumentsFolderNotFoundError: If the folder does not exist.
        EmptyDocumentsFolderError: If the folder contains no PDF files.
    """
    pdf_files = _resolve_pdf_files(documents_dir)
    for pdf_file in pdf_files:
        yield from load_single_pdf(pdf_file)


def load_all_documents(
    documents_dir: PathLike = "documents",
    max_workers: Optional[int] = None,
) -> List[Document]:
    """
    Orchestrator: load every PDF inside the documents folder into Document objects.

    Validates the folder, discovers all PDFs, loads each one (continuing
    past any single-file failure), and returns the combined list.

    Args:
        documents_dir: Path (as a string or Path) to the folder containing PDFs.
            Defaults to "documents".
        max_workers: If set to an int > 1, PDFs are loaded concurrently using
            a thread pool (I/O- and C-extension-bound PDF parsing benefits
            from this). Defaults to None, which loads sequentially -
            deterministic order and simplest behavior for small corpora.

    Returns:
        A list of LangChain Document objects covering every successfully
        loaded page across all PDFs, each with 'filename', 'source', and
        'page' metadata. Order matches sorted filename order when run
        sequentially; order is not guaranteed when max_workers > 1.

    Raises:
        DocumentsFolderNotFoundError: If the folder does not exist.
        EmptyDocumentsFolderError: If the folder contains no PDF files.
        NoDocumentsLoadedError: If every PDF failed to load.
    """
    pdf_files = _resolve_pdf_files(documents_dir)

    all_documents: List[Document] = []
    failed_files: List[str] = []

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(load_single_pdf, pdf_file): pdf_file
                for pdf_file in pdf_files
            }
            for future in as_completed(future_to_file):
                pdf_file = future_to_file[future]
                pages = future.result()  # load_single_pdf never raises
                if pages:
                    all_documents.extend(pages)
                else:
                    failed_files.append(pdf_file.name)
    else:
        for pdf_file in pdf_files:
            pages = load_single_pdf(pdf_file)
            if pages:
                all_documents.extend(pages)
            else:
                failed_files.append(pdf_file.name)

    if not all_documents:
        raise NoDocumentsLoadedError(
            "No documents were successfully loaded from any PDF in "
            f"'{documents_dir}'. Check that the files are valid, non-encrypted PDFs."
        )

    if failed_files:
        logger.warning(
            "%d file(s) failed to load and were skipped: %s",
            len(failed_files),
            sorted(failed_files),
        )

    logger.info(
        "Ingestion summary: %d page(s) loaded from %d/%d PDF file(s).",
        len(all_documents),
        len(pdf_files) - len(failed_files),
        len(pdf_files),
    )
    return all_documents


def summarize_documents(documents: List[Document]) -> Dict[str, int]:
    """
    Build a simple per-file page-count summary, useful for logging and testing.

    Args:
        documents: List of loaded Document objects.

    Returns:
        A dictionary (sorted by filename) mapping filename -> number of
        pages loaded for that file.
    """
    summary: Dict[str, int] = {}
    for doc in documents:
        filename = doc.metadata.get("filename", "unknown")
        summary[filename] = summary.get(filename, 0) + 1
    return dict(sorted(summary.items()))


# --------------------------------------------------------------------------
# Manual test entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        documents = load_all_documents("documents")
        summary = summarize_documents(documents)

        print(f"\nTotal pages loaded: {len(documents)}")
        print("Pages per file:")
        for filename, page_count in summary.items():
            print(f"  - {filename}: {page_count} page(s)")

        print("\nSample metadata from first loaded document:")
        print(documents[0].metadata)

    except (DocumentsFolderNotFoundError, EmptyDocumentsFolderError, NoDocumentsLoadedError) as known_error:
        logger.error("Ingestion stopped: %s", known_error)
    except Exception as unexpected_error:
        logger.error("Unexpected failure during ingestion test: %s", unexpected_error)