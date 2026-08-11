"""
02_preprocessing.py

OncoRAG Clinical Knowledge Assistant - Document Preprocessing Module.

Consumes the list of LangChain Document objects produced by
01_documents.py::load_all_documents() and returns a new list of Document
objects with cleaned `page_content`, while preserving metadata exactly as
received (filename, source, page, etc.).

Cleaning is deliberately conservative for a clinical RAG corpus:
    - No lowercasing (case can be diagnostically meaningful, e.g. "ALL" vs "all").
    - No punctuation removal (dosages, ratios, and citations rely on it).
    - No stripping of medical terminology, abbreviations, or numerals.

Only structural/formatting noise is removed:
    - Repeated spaces/tabs collapsed to a single space.
    - Repeated blank lines collapsed to a single blank line.
    - Trailing whitespace stripped from each line.
    - Unicode normalized (NFKC) to fold visually-identical characters
      (e.g. full-width digits, curly quotes) into consistent forms.
    - Unnecessary control/format characters removed (newline and tab kept).
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import List

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    from langchain.schema import Document


__all__ = [
    "normalize_unicode",
    "normalize_whitespace",
    "remove_extra_blank_lines",
    "clean_document",
    "preprocess_documents",
]

logger = logging.getLogger(__name__)

# Unicode categories treated as "unnecessary control characters":
#   Cc = control (e.g. \x00-\x1f, \x7f)
#   Cf = format (e.g. zero-width space, byte-order mark, joiners)
# Newline and tab are explicitly exempted since they carry structure.
_UNNECESSARY_CONTROL_CATEGORIES = frozenset({"Cc", "Cf"})
_REPEATED_SPACES_PATTERN = re.compile(r"[ \t]{2,}")
_TRAILING_SPACES_PATTERN = re.compile(r"[ \t]+\n")
_REPEATED_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text to NFKC form and strip stray control characters.

    NFKC (Compatibility Composition) folds visually-equivalent characters
    into a single canonical representation - e.g. full-width digits,
    ligatures, and curly quotes become their standard ASCII/Unicode
    counterparts. This does not alter medical terminology or wording;
    it only normalizes how characters are encoded. Control (Cc) and
    format (Cf) characters - e.g. zero-width spaces left over from PDF
    text extraction - are stripped; newline and tab are preserved since
    they carry document structure.

    Args:
        text: Raw text to normalize.

    Returns:
        Unicode-normalized text with unnecessary control/format
        characters removed.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(
        char for char in normalized
        if char in ("\n", "\t")
        or unicodedata.category(char) not in _UNNECESSARY_CONTROL_CATEGORIES
    )
    return normalized


def normalize_whitespace(text: str) -> str:
    """
    Collapse repeated spaces/tabs and strip trailing whitespace per line.

    Only horizontal whitespace (spaces, tabs) is touched here; newline
    structure is left intact for remove_extra_blank_lines() to handle.

    Args:
        text: Text to normalize.

    Returns:
        Text with repeated spaces/tabs collapsed to a single space and
        trailing whitespace removed from the end of each line.
    """
    if not text:
        return ""
    collapsed = _REPEATED_SPACES_PATTERN.sub(" ", text)
    # Handle trailing space before a newline...
    collapsed = _TRAILING_SPACES_PATTERN.sub("\n", collapsed)
    # ...and on the final line, which has no trailing \n for the regex above.
    collapsed = "\n".join(line.rstrip(" \t") for line in collapsed.split("\n"))
    return collapsed


def remove_extra_blank_lines(text: str) -> str:
    """
    Collapse three or more consecutive newlines into a single blank line.

    PDF extraction frequently leaves runs of empty lines from page
    headers/footers, table gutters, or figure captions. A single blank
    line is preserved as a paragraph separator; anything beyond that is
    redundant noise.

    Args:
        text: Text to normalize.

    Returns:
        Text with excessive blank lines collapsed to a maximum of one.
    """
    if not text:
        return ""
    return _REPEATED_BLANK_LINES_PATTERN.sub("\n\n", text)


def clean_document(document: Document) -> Document:
    """
    Apply the full cleaning pipeline to a single Document's page_content.

    Metadata is preserved exactly as received (copied, untouched) - only
    `page_content` is replaced with its cleaned version.

    Args:
        document: A LangChain Document with raw, extracted page_content.

    Returns:
        A new Document instance with cleaned page_content and identical
        metadata to the input.
    """
    original_content = document.page_content or ""

    cleaned = normalize_unicode(original_content)
    cleaned = normalize_whitespace(cleaned)
    cleaned = remove_extra_blank_lines(cleaned)
    cleaned = cleaned.strip()

    return Document(page_content=cleaned, metadata=dict(document.metadata))


def preprocess_documents(documents: List[Document]) -> List[Document]:
    """
    Orchestrator: clean page_content for a list of Document objects.

    Fail-soft by design, consistent with 01_documents.py: a single
    malformed Document (e.g. unexpected page_content type) is logged and
    skipped rather than aborting the entire preprocessing run.

    Args:
        documents: List of Document objects, typically the output of
            01_documents.py::load_all_documents().

    Returns:
        A new list of Document objects with cleaned page_content and
        metadata preserved exactly as received. Order is preserved.
    """
    cleaned_documents: List[Document] = []
    skipped_count = 0

    for index, document in enumerate(documents):
        try:
            cleaned_documents.append(clean_document(document))
        except Exception as exc:
            filename = getattr(document, "metadata", {}).get("filename", "unknown")
            logger.warning(
                "Skipping document at index %d (filename='%s') due to cleaning "
                "error: %s", index, filename, exc,
            )
            skipped_count += 1

    if skipped_count:
        logger.warning(
            "%d document(s) skipped during preprocessing out of %d total.",
            skipped_count,
            len(documents),
        )

    logger.info(
        "Preprocessing complete: %d/%d document(s) cleaned successfully.",
        len(cleaned_documents),
        len(documents),
    )
    return cleaned_documents