"""
07_prompting.py

OncoRAG Clinical Knowledge Assistant - Prompt Construction Module.

Sole responsibility: build the system and user prompts that will
eventually be sent to an LLM. This module does NOT call any LLM API, does
NOT perform retrieval, and does NOT touch the vector database - it only
turns a question plus already-retrieved chunks (from 06_retrieve_context.py)
into well-formed prompt text.

Pipeline position:
    ...
    06_retrieve_context.py                 -> retrieved Document chunks
    07_prompting.py (this)    -> system prompt + user prompt (+ citations)
    (a future module would send this prompt to an LLM)

Metadata contract:
    This module never modifies chunk metadata. It relies on
    06_retrieve_context.py::format_context() and collect_sources() to read
    filename/page metadata for context formatting and citation, and
    treats their output as read-only.
"""

import logging
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - fallback for older LangChain versions
    from langchain.schema import Document


def _load_retriever_module():
    """Dynamically load 06_retrieve_context.py (its filename starts with a digit,
    so it cannot be imported with a normal `import` statement) and return
    the module object so we can reuse format_context()/collect_sources()
    without duplicating their logic here.
    """
    import importlib.util

    module_path = Path(__file__).resolve().parent / "06_retrieve_context.py"
    spec = importlib.util.spec_from_file_location("_oncorag_retriever", module_path)
    retriever_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retriever_module)
    return retriever_module


_retriever_module = _load_retriever_module()
format_context = _retriever_module.format_context
collect_sources = _retriever_module.collect_sources


__all__ = [
    "PromptConstructionError",
    "PromptTooLargeError",
    "PromptStats",
    "build_system_prompt",
    "build_user_prompt",
    "build_complete_prompt",
    "count_prompt_tokens_estimate",
    "validate_prompt_length",
    "preview_prompt",
]

logger = logging.getLogger(__name__)

RetrievedItem = Union[Document, Tuple[Document, float]]
TOKENS_PER_WORD_ESTIMATE: float = 1.3

# Custom exceptions
class PromptConstructionError(Exception):
    """Raised when a prompt cannot be constructed from the given inputs."""


class PromptTooLargeError(Exception):
    """Raised when an estimated prompt size exceeds an allowed token limit."""


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptStats:
    """Estimated size statistics for a prompt string.

    Attributes:
        character_count: Total number of characters in the prompt.
        word_count: Total number of whitespace-separated words in the prompt.
        estimated_tokens: Approximate token count, computed as
            round(word_count * 1.3).
    """

    character_count: int
    word_count: int
    estimated_tokens: int


# --------------------------------------------------------------------------
# Core functions
# --------------------------------------------------------------------------
def build_system_prompt() -> str:
    """Build the system-level instruction prompt for the clinical assistant.

    Encodes the assistant's operating rules: answer only from retrieved
    context, never fabricate medical facts, clearly flag missing
    information, always cite filename and page number, never claim
    personal medical expertise, never diagnose patients, and maintain a
    professional medical tone.

    Returns:
        The complete system prompt as a string.
    """
    return (
        "You are OncoRAG, a clinical knowledge assistant that helps "
        "oncologists and cancer-care researchers navigate uploaded "
        "oncology literature (guidelines, staging systems, trial data, "
        "treatment protocols, and related medical documents).\n\n"
        "You must follow these rules at all times:\n"
        "1. Answer ONLY using the information provided in the retrieved "
        "context below. Do not use outside knowledge, even if you "
        "believe it is correct.\n"
        "2. Never invent, assume, or infer medical facts, dosages, "
        "staging criteria, or outcomes that are not explicitly present "
        "in the retrieved context.\n"
        "3. If the retrieved context does not contain enough information "
        "to answer the question, clearly state that the uploaded "
        "literature does not contain enough information to answer it - "
        "do not guess or fill gaps with general oncology knowledge.\n"
        "4. Always cite the filename and page number of every source you "
        "use to support a statement.\n"
        "5. Never claim personal medical expertise, professional "
        "judgment, or clinical experience of your own - you are "
        "summarizing documents, not practicing medicine.\n"
        "6. Never diagnose a patient, recommend a specific treatment for "
        "a named patient, or provide a definitive clinical decision - you "
        "may summarize what the literature says about diagnostic "
        "criteria, staging, or treatment options, but the decision "
        "belongs to the treating clinician.\n"
        "7. Flag any information related to dosages, contraindications, "
        "or drug interactions as literature-reported values that must be "
        "independently verified before clinical use.\n"
        "8. Maintain a professional, precise, medical tone at all times."
    )


def build_user_prompt(
    question: str,
    context: str,
    conversation_history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Build the user-facing prompt combining the question, context, and instructions.

    Args:
        question: The clinical question being asked.
        context: The formatted retrieved context (typically the output of
            06_retrieve_context.py::format_context()).
        conversation_history: Optional list of (previous_question,
            previous_answer) pairs, oldest first, used so follow-up
            questions ("what about pediatric patients?") can be
            understood in context. Only used to interpret the current
            question - the model is still instructed to answer solely
            from the Retrieved Context, never from unretrieved parts of
            the earlier conversation.

    Returns:
        A formatted user prompt string with an optional "Previous
        Conversation" section, followed by "Question", "Retrieved
        Context", and "Instructions" sections.

    Raises:
        PromptConstructionError: If question is empty.
    """
    if not question or not question.strip():
        raise PromptConstructionError("question must not be empty.")

    context_section = context.strip() if context and context.strip() else (
        "No relevant context was retrieved from the uploaded literature."
    )

    history_section = ""
    if conversation_history:
        turns = "\n\n".join(
            f"Previous Question: {prev_q.strip()}\nPrevious Answer: {prev_a.strip()}"
            for prev_q, prev_a in conversation_history
        )
        history_section = (
            "Previous Conversation (for understanding follow-up "
            "questions only - do not treat this as source material)\n"
            f"{turns}\n\n"
        )

    return (
        f"{history_section}"
        "Question\n"
        f"{question.strip()}\n\n"
        "Retrieved Context\n"
        f"{context_section}\n\n"
        "Instructions\n"
        "Answer the question above using only the information in the "
        "Retrieved Context (use the Previous Conversation only to "
        "understand what a follow-up question like 'what about ...' is "
        "referring to). Cite the filename and page number for every "
        "fact you use. If the Retrieved Context does not contain enough "
        "information to answer the question, say so explicitly instead "
        "of guessing."
    )


def build_complete_prompt(
    question: str,
    retrieved_documents: List[RetrievedItem],
    conversation_history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Build the full prompt (system prompt + user prompt + citations).

    Uses 06_retrieve_context.py::format_context() to render the retrieved chunks
    into context text and 06_retrieve_context.py::collect_sources() to build a
    citation list, without modifying any chunk metadata.

    Args:
        question: The clinical question being asked.
        retrieved_documents: A list of Document objects, or a list of
            (Document, score) tuples, typically from 06_retrieve_context.py's
            retrieval functions.

    Returns:
        The complete prompt string: system prompt, followed by the user
        prompt (question, context, instructions), followed by a "Sources"
        section listing each unique filename/page citation.

    Raises:
        PromptConstructionError: If question is empty, retrieved_documents
            is empty, or context/citation formatting fails.
    """
    if not question or not question.strip():
        raise PromptConstructionError("question must not be empty.")
    if not retrieved_documents:
        raise PromptConstructionError(
            "retrieved_documents must not be empty; at least one "
            "retrieved chunk is required to build a grounded prompt."
        )

    try:
        context = format_context(retrieved_documents)
        sources = collect_sources(retrieved_documents)
    except Exception as exc:
        raise PromptConstructionError(
            f"Failed to format context/sources for question '{question}': {exc}"
        ) from exc

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(question, context, conversation_history)

    sources_section = "Sources\n" + "\n".join(sources) if sources else (
        "Sources\nNo sources available."
    )

    complete_prompt = f"{system_prompt}\n\n{user_prompt}\n\n{sources_section}"

    stats = count_prompt_tokens_estimate(complete_prompt)
    logger.info(
        "Built prompt for question: '%s' | retrieved chunks: %d | "
        "estimated tokens: %d",
        question, len(retrieved_documents), stats.estimated_tokens,
    )

    return complete_prompt


def count_prompt_tokens_estimate(prompt: str) -> PromptStats:
    """Estimate the size of a prompt in characters, words, and tokens.

    Token count is approximated as word_count * 1.3, a common rough
    heuristic for English text with typical LLM tokenizers.

    Args:
        prompt: The prompt text to measure.

    Returns:
        A PromptStats instance with character_count, word_count, and
        estimated_tokens.
    """
    character_count = len(prompt)
    word_count = len(prompt.split())
    estimated_tokens = round(word_count * TOKENS_PER_WORD_ESTIMATE)

    return PromptStats(
        character_count=character_count,
        word_count=word_count,
        estimated_tokens=estimated_tokens,
    )


def validate_prompt_length(prompt: str, max_tokens: int) -> None:
    """Validate that a prompt's estimated token count does not exceed a limit.

    Args:
        prompt: The prompt text to validate.
        max_tokens: The maximum allowed estimated token count.

    Raises:
        PromptTooLargeError: If max_tokens is not positive, or the
            prompt's estimated token count exceeds max_tokens.
    """
    if max_tokens <= 0:
        raise PromptTooLargeError(
            f"max_tokens must be a positive integer, got {max_tokens}."
        )

    stats = count_prompt_tokens_estimate(prompt)
    if stats.estimated_tokens > max_tokens:
        raise PromptTooLargeError(
            f"Estimated prompt size ({stats.estimated_tokens} tokens) "
            f"exceeds the allowed limit of {max_tokens} tokens. "
            f"(characters={stats.character_count}, words={stats.word_count})"
        )

    logger.info(
        "Prompt length validated: %d estimated tokens (limit: %d).",
        stats.estimated_tokens, max_tokens,
    )


def preview_prompt(prompt: str) -> None:
    """Print a complete prompt in a readable, bordered format for manual review.

    Args:
        prompt: The prompt text to display.

    Returns:
        None. Output is printed directly to stdout.
    """
    stats = count_prompt_tokens_estimate(prompt)
    separator = "=" * 70

    print(separator)
    print("PROMPT PREVIEW")
    print(separator)
    print(prompt)
    print(separator)
    print(
        f"[{stats.character_count} characters | {stats.word_count} words | "
        f"~{stats.estimated_tokens} tokens]"
    )
    print(separator)
