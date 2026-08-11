"""
streamlit_app.py

OncoRAG - Oncology Clinical Knowledge Assistant.

Streamlit UI that wires together the 01-08 pipeline modules into a RAG
assistant for oncologists: upload oncology literature (guidelines, trial
data, protocols), build a searchable index, and ask grounded, cited
questions against it.

Beyond the base RAG loop, this UI adds a few features aimed at making the
tool genuinely useful in a clinical-reading workflow rather than a demo:

    - Multi-turn memory: follow-up questions ("what about pediatric
      patients?") are understood in context of the last few turns.
    - Confidence flagging: low retrieval-similarity answers are visibly
      flagged as weak evidence instead of presented with false confidence.
    - Source page preview: each citation can be expanded into the actual
      rendered PDF page, so the doctor can verify a claim without leaving
      the app.
    - Suggested follow-up questions: after each answer, the model proposes
      3 natural next questions a clinician might ask.
    - Arabic quick-translation of any answer, on demand, with citations
      left untouched.
    - Session export to PDF, for tumor-board handouts or documentation.

Deployment:
    In Streamlit Cloud, set these under "Manage app -> Secrets" (TOML):

        OPENROUTER_API_KEY = "your_openrouter_key_here"
        OPENROUTER_MODEL = "openai/gpt-4o-mini"

    Never write a real API key into this file or any other Python file
    in the repo. Locally, put the same two values in a .env file
    (see .env.example) - 08_llm.py loads it automatically.
"""

import importlib.util
import os
import re
import time
import types
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import streamlit as st

try:
    import pymupdf as fitz  # PyMuPDF, used only for rendering a cited page as an image
except ImportError:  # pragma: no cover
    try:
        import fitz  # older PyMuPDF versions expose only this name
    except ImportError:
        fitz = None

try:
    from fpdf import FPDF
except ImportError:  # pragma: no cover
    FPDF = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE_FILES = [
    "01_documents.py",
    "02_preprocessing.py",
    "03_chunking.py",
    "04_vector_representation.py",
    "05_create_chroma_store.py",
    "06_retrieve_context.py",
    "07_prompting.py",
    "08_llm.py",
]

DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "vector_database")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
CONVERSATION_MEMORY_TURNS = 4          # how many previous Q&A pairs to keep as context
LOW_CONFIDENCE_THRESHOLD = 0.45        # below this avg. similarity, flag the answer

st.set_page_config(
    page_title="OncoRAG - Oncology Clinical Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Modern theme (pure CSS - no extra dependency, no effect on pipeline logic).
# Uses Streamlit's stable data-testid hooks so it degrades gracefully rather
# than breaking if a future Streamlit release renames internal classes.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    :root {
        --oncorag-primary: #1f6f78;
        --oncorag-primary-dark: #124c53;
        --oncorag-accent: #e8927c;
        --oncorag-bg-soft: #f4f8f8;
    }

    /* App background */
    .stApp { background: linear-gradient(180deg, #f7fafa 0%, #eef3f4 100%); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--oncorag-primary-dark) 0%, var(--oncorag-primary) 100%);
    }
    section[data-testid="stSidebar"] * { color: #eef7f7 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

    /* Title */
    h1 { font-weight: 700 !important; color: var(--oncorag-primary-dark); letter-spacing: -0.5px; }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.4rem 0.2rem;
        margin-bottom: 0.4rem;
        box-shadow: 0 1px 3px rgba(18, 76, 83, 0.08);
        background: #ffffff;
    }

    /* Chat input */
    div[data-testid="stChatInput"] {
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(18, 76, 83, 0.10);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(18, 76, 83, 0.15);
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        border-color: var(--oncorag-primary);
        color: var(--oncorag-primary);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--oncorag-primary) 0%, var(--oncorag-primary-dark) 100%);
        border: none;
    }

    /* Metrics as soft cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 4px rgba(18, 76, 83, 0.08);
        border: 1px solid rgba(18, 76, 83, 0.06);
    }

    /* Expanders (sources) */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(18, 76, 83, 0.10);
        overflow: hidden;
    }

    /* Caption under title */
    .stApp > header + div p { color: #4a6367; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Pipeline loading (cached so modules/models are only loaded once per
# running process, not on every Streamlit rerun/interaction)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline() -> types.ModuleType:
    """Load the 01-08 pipeline files into one merged namespace.

    They are numbered (can't be imported with a normal `import`
    statement), so each is loaded from disk with importlib and its
    public names are copied onto a single merged module object.
    """
    merged = types.ModuleType("oncorag_pipeline")
    load_errors = []
    for filename in PIPELINE_FILES:
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            load_errors.append(f"{filename}: file not found")
            continue
        module_name = filename.replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not hidden
            load_errors.append(f"{filename}: {exc}")
            continue
        for attr in dir(mod):
            if not attr.startswith("_"):
                setattr(merged, attr, getattr(mod, attr))
    merged._load_errors = load_errors  # type: ignore[attr-defined]
    return merged


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    pipeline = get_pipeline()
    return pipeline.create_embedding_model()


@st.cache_resource(show_spinner=False)
def get_vector_store(_persist_directory: str):
    pipeline = get_pipeline()
    embedding_model = get_embedding_model()
    if not pipeline.database_exists(_persist_directory):
        return None
    return pipeline.load_vector_store(_persist_directory, embedding_model)


@st.cache_resource(show_spinner=False)
def get_chunk_count(_persist_directory: str) -> int:
    pipeline = get_pipeline()
    vector_store = get_vector_store(_persist_directory)
    if vector_store is None:
        return 0
    return pipeline.count_documents(vector_store)


@st.cache_resource(show_spinner=False)
def get_doc_count() -> int:
    pipeline = get_pipeline()
    try:
        return len(pipeline.load_all_documents(DOCUMENTS_DIR))
    except Exception:
        return 0


def refresh_index_caches() -> None:
    """Clear cached read-through state after rebuilding the index."""
    get_vector_store.clear()
    get_chunk_count.clear()
    get_doc_count.clear()


# --------------------------------------------------------------------------
# Credentials: Streamlit secrets (deployed) / .env (local) -> env vars.
# NOTE: no key is ever written into this file. See README.md.
# --------------------------------------------------------------------------
pipeline = get_pipeline()

try:
    if not pipeline.OPENROUTER_API_KEY:
        pipeline.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    if st.secrets.get("OPENROUTER_MODEL"):
        pipeline.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL")
except Exception:
    pass  # st.secrets raises if no secrets.toml exists locally - that's fine

has_api_key = bool(pipeline.OPENROUTER_API_KEY)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # full display history (incl. sources/follow-ups)
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # [(question, answer), ...] fed back for memory
if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = 0.0
if "arabic_translations" not in st.session_state:
    st.session_state.arabic_translations = {}   # message index -> Arabic text
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# --------------------------------------------------------------------------
# System status
# --------------------------------------------------------------------------
chunk_count = get_chunk_count(PERSIST_DIRECTORY)
doc_count = get_doc_count()
pipeline_ready = chunk_count > 0 and has_api_key and not pipeline._load_errors


# --------------------------------------------------------------------------
# Feature helpers
# --------------------------------------------------------------------------
def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma's squared-L2 distance to an approximate 0-1 similarity.

    Embeddings are L2-normalized (see 04_vector_representation.py), so for
    normalized vectors: cosine_similarity = 1 - (squared_l2_distance / 2).
    """
    return max(0.0, min(1.0, 1 - (distance / 2)))


def generate_follow_up_questions(question: str, answer: str, api_key: str, model: str) -> List[str]:
    """Ask the model for 3 natural follow-up questions a clinician might ask next.

    Best-effort: any failure here (network hiccup, parsing issue) simply
    yields no suggestions rather than breaking the main answer.
    """
    prompt = (
        "Based on this oncology question-and-answer exchange, suggest "
        "exactly 3 short, specific, clinically relevant follow-up "
        "questions a treating oncologist might naturally ask next. "
        "Return ONLY a numbered list (1., 2., 3.), no other text.\n\n"
        f"Question: {question}\n\nAnswer: {answer}"
    )
    try:
        raw = pipeline.generate_answer(api_key, model, prompt, max_output_tokens=150)
    except Exception:
        return []

    questions = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line.strip())
        if cleaned:
            questions.append(cleaned)
    return questions[:3]


def translate_to_arabic(answer: str, api_key: str, model: str) -> str:
    """Translate a generated answer into clinical Arabic, keeping citations intact."""
    prompt = (
        "Translate the following clinical answer into clear, professional "
        "Modern Standard Arabic suitable for a physician. Keep every "
        "citation (filename and page number) exactly as written in the "
        "original - do not translate filenames or page numbers. Return "
        "only the translated answer.\n\n"
        f"{answer}"
    )
    return pipeline.generate_answer(api_key, model, prompt, max_output_tokens=800)


@st.cache_data(show_spinner=False)
def render_pdf_page(source_path: str, page_number, zoom: float = 1.6) -> Optional[bytes]:
    """Render one page of a source PDF to PNG bytes, for on-demand verification.

    Returns None (rather than raising) if PyMuPDF isn't installed, the
    file can't be found, or the page number is out of range - the caller
    shows a friendly message instead of crashing the page.
    """
    if fitz is None or not source_path or not os.path.exists(source_path):
        return None
    try:
        page_index = int(page_number) - 1  # metadata page numbers are 1-indexed
    except (TypeError, ValueError):
        return None
    if page_index < 0:
        return None
    try:
        with fitz.open(source_path) as doc:
            if page_index >= len(doc):
                return None
            pixmap = doc[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            return pixmap.tobytes("png")
    except Exception:
        return None


def build_session_pdf(chat_history: List[Dict]) -> Optional[bytes]:
    """Export the visible chat session as a simple PDF report."""
    if FPDF is None or not chat_history:
        return None

    def _latin1(text: str) -> str:
        # fpdf2's core fonts only support latin-1; anything outside that
        # range (e.g. Arabic translations) is replaced rather than crashing.
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "OncoRAG Session Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, _latin1("Generated from the OncoRAG oncology literature assistant."), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for message in chat_history:
        role_label = "Clinician question" if message["role"] == "user" else "OncoRAG answer"
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 7, _latin1(f"{role_label}:"))
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _latin1(message["content"]))

        if message.get("sources"):
            pdf.set_font("Helvetica", "I", 9)
            for src in message["sources"]:
                pdf.multi_cell(
                    0, 5,
                    _latin1(f"  - Source: {src['filename']} (page {src['page']})"),
                )
        pdf.ln(3)

    return bytes(pdf.output())


def render_source_block(sources: List[Dict], key_prefix: str) -> None:
    """Render the expandable sources list, each with an on-demand page preview."""
    with st.expander(f"Sources ({len(sources)})"):
        for i, src in enumerate(sources):
            st.markdown(
                f"**{src['filename']}** (page {src['page']}) - "
                f"similarity {src['similarity']:.0%}"
            )
            st.caption(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))

            if fitz is not None and src.get("source_path"):
                preview_key = f"{key_prefix}_preview_{i}"
                if st.button("View PDF page", key=preview_key):
                    st.session_state[f"{preview_key}_show"] = True
                if st.session_state.get(f"{preview_key}_show"):
                    image_bytes = render_pdf_page(src["source_path"], src["page"])
                    if image_bytes:
                        st.image(image_bytes, caption=f"{src['filename']}, page {src['page']}")
                    else:
                        st.caption("Could not render this page (file missing or unreadable).")
            st.divider()


# --------------------------------------------------------------------------
# Core query functions
# --------------------------------------------------------------------------
def prepare_query(question: str, conversation_history: List[Tuple[str, str]]) -> Dict:
    """Retrieval + prompt construction only - fast, done before any LLM call."""
    vector_store = get_vector_store(PERSIST_DIRECTORY)
    retrieved_with_scores = pipeline.retrieve_with_scores(question, vector_store, k=TOP_K)
    prompt = pipeline.build_complete_prompt(question, retrieved_with_scores, conversation_history)

    api_key = pipeline.create_openrouter_client(pipeline.OPENROUTER_API_KEY)
    model = pipeline.OPENROUTER_MODEL

    sources = [
        {
            "filename": doc.metadata.get("filename", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "source_path": doc.metadata.get("source", ""),
            "text": doc.page_content,
            "similarity": _distance_to_similarity(distance),
        }
        for doc, distance in retrieved_with_scores
    ]
    avg_similarity = (
        sum(src["similarity"] for src in sources) / len(sources) if sources else 0.0
    )

    return {
        "prompt": prompt,
        "api_key": api_key,
        "model": model,
        "sources": sources,
        "avg_similarity": avg_similarity,
    }


def run_query(question: str, conversation_history: List[Tuple[str, str]]) -> Dict:
    """Non-streaming variant kept for callers that need one complete result
    (e.g. programmatic use outside the chat UI, which streams instead)."""
    prep = prepare_query(question, conversation_history)
    start_time = time.monotonic()
    answer = "".join(pipeline.generate_answer_stream(prep["api_key"], prep["model"], prep["prompt"]))
    elapsed = time.monotonic() - start_time
    follow_ups = generate_follow_up_questions(question, answer, prep["api_key"], prep["model"])
    return {
        "answer": answer,
        "sources": prep["sources"],
        "elapsed": elapsed,
        "avg_similarity": prep["avg_similarity"],
        "follow_ups": follow_ups,
    }


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### OncoRAG")
    st.caption("Oncology Clinical Knowledge Assistant")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Chat", "Knowledge Base", "Settings"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**System status**")
    st.markdown(f"**Vector index:** {'Ready' if chunk_count > 0 else 'Not built'} ({chunk_count} chunks)")
    st.markdown(f"**OpenRouter API key:** {'Configured' if has_api_key else 'Missing'}")
    if pipeline._load_errors:
        st.markdown("**Pipeline modules:** Load errors detected")
        with st.expander("Module load errors"):
            for err in pipeline._load_errors:
                st.caption(err)

    if not has_api_key:
        st.warning(
            "No OPENROUTER_API_KEY found. Add it under Streamlit "
            "'Manage app -> Secrets' (or a local .env file) - see README.md."
        )

    if st.session_state.chat_history:
        st.divider()
        pdf_bytes = build_session_pdf(st.session_state.chat_history)
        if pdf_bytes:
            st.download_button(
                "Export session as PDF",
                data=pdf_bytes,
                file_name="oncorag_session.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_history = []
            st.session_state.arabic_translations = {}
            st.rerun()

st.title("OncoRAG")
st.caption(
    "Ask grounded, cited questions against your uploaded oncology literature. "
    "This tool summarizes documents - it does not diagnose or make treatment decisions."
)

# --------------------------------------------------------------------------
# Chat page
# --------------------------------------------------------------------------
if page == "Chat":
    if not pipeline_ready:
        st.info(
            "The assistant isn't ready yet. Add oncology PDFs and build the "
            "index in **Knowledge Base**, and make sure an OpenRouter API "
            "key is configured (see sidebar)."
        )

    if pipeline_ready and not st.session_state.chat_history:
        st.caption("Try one of these to get started:")
        example_questions = [
            "What are the key presenting features of acute leukemia?",
            "How is tumor lysis syndrome recognized and managed?",
            "What does the ANC threshold mean for febrile neutropenia?",
            "What's the difference between clinical and pathologic staging?",
        ]
        chip_cols = st.columns(len(example_questions))
        for chip_col, example_question in zip(chip_cols, example_questions):
            with chip_col:
                if st.button(example_question, key=f"example_{example_question}", use_container_width=True):
                    st.session_state.pending_question = example_question
                    st.rerun()

    history_length = len(st.session_state.chat_history)
    for idx, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                if message.get("avg_similarity", 1.0) < LOW_CONFIDENCE_THRESHOLD:
                    st.warning(
                        "Low retrieval confidence - the source material for this "
                        "answer was only weakly related to the question. Verify "
                        "against primary literature before relying on it."
                    )

                if message.get("sources"):
                    render_source_block(message["sources"], key_prefix=f"msg{idx}")

                translation_col, _ = st.columns([1, 4])
                with translation_col:
                    if st.button("Translate to Arabic", key=f"arabic_btn_{idx}"):
                        if idx not in st.session_state.arabic_translations:
                            api_key = pipeline.create_openrouter_client(pipeline.OPENROUTER_API_KEY)
                            with st.spinner("Translating..."):
                                st.session_state.arabic_translations[idx] = translate_to_arabic(
                                    message["content"], api_key, pipeline.OPENROUTER_MODEL
                                )
                if idx in st.session_state.arabic_translations:
                    st.markdown(st.session_state.arabic_translations[idx])

                # Only offer follow-up suggestions on the most recent answer.
                if idx == history_length - 1 and message.get("follow_ups"):
                    st.caption("Suggested follow-ups:")
                    for f_idx, follow_up in enumerate(message["follow_ups"]):
                        if st.button(follow_up, key=f"followup_{idx}_{f_idx}"):
                            st.session_state.pending_question = follow_up
                            st.rerun()

    typed_question = st.chat_input(
        "Ask about a treatment protocol, staging system, trial finding..."
        if pipeline_ready else "Set up the knowledge base and API key first",
        disabled=not pipeline_ready,
    )

    user_query = st.session_state.pending_question or typed_question
    st.session_state.pending_question = None

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving relevant context..."):
                    prep = prepare_query(
                        user_query,
                        st.session_state.conversation_history[-CONVERSATION_MEMORY_TURNS:],
                    )

                start_time = time.monotonic()
                answer_text = st.write_stream(
                    pipeline.generate_answer_stream(prep["api_key"], prep["model"], prep["prompt"])
                )
                st.session_state.last_response_time = time.monotonic() - start_time

                with st.spinner("Preparing follow-up suggestions..."):
                    follow_ups = generate_follow_up_questions(
                        user_query, answer_text, prep["api_key"], prep["model"]
                    )

                result = {
                    "answer": answer_text,
                    "sources": prep["sources"],
                    "avg_similarity": prep["avg_similarity"],
                    "follow_ups": follow_ups,
                }
            except Exception as exc:  # noqa: BLE001
                result = {
                    "answer": f"The assistant hit an error and could not generate an answer: {exc}",
                    "sources": [],
                    "avg_similarity": 1.0,
                    "follow_ups": [],
                }
                st.markdown(result["answer"])

            if result["avg_similarity"] < LOW_CONFIDENCE_THRESHOLD and result["sources"]:
                st.warning("Low retrieval confidence - verify against primary literature.")
            if result["sources"]:
                render_source_block(result["sources"], key_prefix="latest")

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "avg_similarity": result["avg_similarity"],
                "follow_ups": result["follow_ups"],
            }
        )
        st.session_state.conversation_history.append((user_query, result["answer"]))
        st.rerun()

# --------------------------------------------------------------------------
# Knowledge Base page
# --------------------------------------------------------------------------
elif page == "Knowledge Base":
    st.subheader("Knowledge Base")
    st.write(
        "Upload oncology PDFs (guidelines, staging references, trial "
        "papers, protocols), then build the index so the Chat tab can "
        "retrieve grounded, cited answers from them."
    )

    col_a, col_b = st.columns(2)
    col_a.metric("PDF pages loaded", doc_count)
    col_b.metric("Indexed chunks", chunk_count)

    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    uploaded_files = st.file_uploader(
        "Upload oncology PDFs", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(DOCUMENTS_DIR, uploaded_file.name), "wb") as out_file:
                out_file.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to the documents folder.")
        get_doc_count.clear()

    existing_pdfs = sorted(Path(DOCUMENTS_DIR).glob("*.pdf"))
    if existing_pdfs:
        with st.expander(f"Documents currently in the folder ({len(existing_pdfs)})"):
            for pdf_path in existing_pdfs:
                st.caption(pdf_path.name)

    st.divider()
    if st.button("Build / Rebuild Index", type="primary", use_container_width=True):
        if not existing_pdfs:
            st.error("No PDF files found in the documents folder. Upload some first.")
        else:
            with st.spinner("Loading, chunking, embedding, and indexing documents..."):
                try:
                    docs = pipeline.load_all_documents(DOCUMENTS_DIR)
                    cleaned = pipeline.preprocess_documents(docs)
                    chunks = pipeline.chunk_documents(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)

                    if pipeline.database_exists(PERSIST_DIRECTORY):
                        pipeline.delete_vector_store(PERSIST_DIRECTORY)

                    embedding_model = get_embedding_model()
                    pipeline.create_vector_store(chunks, embedding_model, PERSIST_DIRECTORY)

                    refresh_index_caches()
                    st.success(f"Index built successfully: {len(chunks)} chunks indexed.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Index build failed: {exc}")

# --------------------------------------------------------------------------
# Settings page
# --------------------------------------------------------------------------
elif page == "Settings":
    st.subheader("Settings")
    st.caption("Read-only view of the current configuration.")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("LLM model (OpenRouter)", value=pipeline.OPENROUTER_MODEL, disabled=True)
        st.text_input(
            "Embedding model",
            value="sentence-transformers/all-MiniLM-L6-v2",
            disabled=True,
        )
    with col2:
        st.number_input("Top-K retrieval", value=TOP_K, disabled=True)
        st.number_input("Chunk size (chars)", value=CHUNK_SIZE, disabled=True)
        st.number_input(
            "Conversation memory (turns)", value=CONVERSATION_MEMORY_TURNS, disabled=True
        )

    st.divider()
    st.caption(
        f"Last response time: {st.session_state.last_response_time:.2f}s"
        if st.session_state.last_response_time else "No queries run yet this session."
    )
