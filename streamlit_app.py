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

import concurrent.futures
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
    from fpdf.enums import WrapMode
except ImportError:  # pragma: no cover
    FPDF = None
    WrapMode = None

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
MIN_SOURCE_SIMILARITY = 0.35           # chunks below this are noise, not real matches - drop them
FOLLOWUP_TIMEOUT_SECONDS = 20          # bail on the follow-up-question call rather than hang forever
TRANSLATE_TIMEOUT_SECONDS = 30         # translation is a longer generation, give it a bit more room
AVATAR_USER = "🩺"
AVATAR_ASSISTANT = "⚕️"

st.set_page_config(
    page_title="OncoRAG - Oncology Clinical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Theme state. Streamlit resolves a keyed widget's value into
# st.session_state BEFORE the script body runs, so reading it here (ahead
# of where the toggle widget is actually drawn, in the sidebar below) still
# reflects the click that triggered this rerun - no extra rerun/flash.
# --------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

_THEME = {
    "light": {
        "paper": "#f3f5f1",
        "surface": "#ffffff",
        "surface_raised": "#ffffff",
        "ink": "#101917",
        "ink_soft": "#57645f",
        "border": "#dde3dd",
        "primary": "#0e5c52",
        "primary_deep": "#093f38",
        "primary_soft_bg": "rgba(14,92,82,0.07)",
        "primary_soft_border": "rgba(14,92,82,0.22)",
        "accent": "#d1652b",
        "good": "#1f8f5f",
        "warn": "#c17a1f",
        "bad": "#b23b30",
        "sidebar_text": "#eef3f0",
        "sidebar_border": "rgba(255,255,255,0.14)",
        "shadow": "0 1px 2px rgba(16,25,23,0.04), 0 10px 28px -16px rgba(16,25,23,0.16)",
        "scheme": "light",
    },
    "dark": {
        "paper": "#0b1211",
        "surface": "#121a18",
        "surface_raised": "#16201d",
        "ink": "#e9efec",
        "ink_soft": "#93a29c",
        "border": "rgba(255,255,255,0.10)",
        "primary": "#39cbb4",
        "primary_deep": "#1f8f7e",
        "primary_soft_bg": "rgba(57,203,180,0.10)",
        "primary_soft_border": "rgba(57,203,180,0.28)",
        "accent": "#ff9257",
        "good": "#45c98a",
        "warn": "#e0a53f",
        "bad": "#e2645a",
        "sidebar_text": "#eaf2ef",
        "sidebar_border": "rgba(255,255,255,0.10)",
        "shadow": "0 1px 2px rgba(0,0,0,0.3), 0 10px 28px -16px rgba(0,0,0,0.55)",
        "scheme": "dark",
    },
}
_T = _THEME["dark"] if st.session_state.dark_mode else _THEME["light"]

# --------------------------------------------------------------------------
# Modern theme (pure CSS - no extra dependency, no effect on pipeline logic).
# Uses Streamlit's stable data-testid hooks so it degrades gracefully rather
# than breaking if a future Streamlit release renames internal classes.
# Signature detail: a slow-drifting "trace" sweep under the header, styled
# after a monitor readout - a quiet nod to continuous, evidence-based
# reading rather than a one-off answer.
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        color-scheme: {_T["scheme"]};
        --oncorag-primary: {_T["primary"]};
        --oncorag-primary-deep: {_T["primary_deep"]};
        --oncorag-primary-soft-bg: {_T["primary_soft_bg"]};
        --oncorag-primary-soft-border: {_T["primary_soft_border"]};
        --oncorag-accent: {_T["accent"]};
        --oncorag-paper: {_T["paper"]};
        --oncorag-surface: {_T["surface"]};
        --oncorag-ink: {_T["ink"]};
        --oncorag-ink-soft: {_T["ink_soft"]};
        --oncorag-border: {_T["border"]};
        --oncorag-good: {_T["good"]};
        --oncorag-warn: {_T["warn"]};
        --oncorag-bad: {_T["bad"]};
        --oncorag-shadow: {_T["shadow"]};
        --font-display: 'Fraunces', 'Source Serif 4', Georgia, serif;
        --font-body: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, monospace;
    }}

    html, body, [class*="css"] {{
        font-family: var(--font-body);
        color: var(--oncorag-ink);
    }}

    /* Theme-swap transition, scoped to the handful of elements whose
       colors actually change with dark mode - NOT a universal `*` rule.
       That was the real cause of the lag: `*` makes the browser watch
       every property on every node (including the hundreds Streamlit
       redraws on each rerun), which forces a full style recalculation on
       almost every interaction. */
    .stApp, section[data-testid="stSidebar"], .block-container,
    div[data-testid="stChatMessage"], div[data-testid="stChatInput"],
    div[data-testid="stMetric"], div[data-testid="stExpander"],
    section[data-testid="stFileUploaderDropzone"], .oncorag-badge, body {{
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }}

    @keyframes oncorag-rise {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes oncorag-trace {{
        0%   {{ transform: translateX(-220px); }}
        100% {{ transform: translateX(calc(100% + 220px)); }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; transition: none !important; }}
    }}

    /* ---------------------------------------------------------------- */
    /* Overall canvas                                                     */
    /* ---------------------------------------------------------------- */
    .stApp {{ background: var(--oncorag-paper); }}
    .block-container {{ padding-top: 2rem; max-width: 1120px; }}

    hr {{ border: none; border-top: 1px solid var(--oncorag-border); margin: 1.1rem 0; }}

    ::selection {{ background: var(--oncorag-primary-soft-border); }}

    /* Visible keyboard focus everywhere - accessibility floor */
    a:focus-visible, button:focus-visible, input:focus-visible,
    [role="radio"]:focus-visible, [role="button"]:focus-visible {{
        outline: 2px solid var(--oncorag-primary) !important;
        outline-offset: 2px;
    }}

    /* ---------------------------------------------------------------- */
    /* Sidebar - flat deep panel                                         */
    /* ---------------------------------------------------------------- */
    section[data-testid="stSidebar"] {{
        background: var(--oncorag-primary-deep);
        border-right: none;
    }}
    section[data-testid="stSidebar"] * {{ color: {_T["sidebar_text"]} !important; font-family: var(--font-body); }}
    section[data-testid="stSidebar"] hr {{ border-top-color: {_T["sidebar_border"]}; margin: 0.9rem 0; }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

    /* Dark-mode toggle - styled as a deliberate switch, not a stray checkbox */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-family: var(--font-mono) !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        opacity: 0.75;
    }}

    /* Sidebar nav - an index list with a left rule, not a pill/glass control */
    section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 0; display: flex; flex-direction: column; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        background: transparent;
        border: none;
        border-left: 2px solid transparent;
        border-radius: 0;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0;
        transition: border-color 0.15s ease-in-out, background 0.15s ease-in-out;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.05); }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background: rgba(255,255,255,0.06);
        border-left-color: var(--oncorag-accent);
        font-weight: 600;
    }}

    /* Status badges - footnote/evidence-tag style: mono, square, hairline */
    .oncorag-badge {{
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-family: var(--font-mono);
        font-size: 0.76rem; font-weight: 500; letter-spacing: 0.02em;
        padding: 0.24rem 0.55rem; border-radius: 5px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.18);
        margin: 0.15rem 0.3rem 0.15rem 0;
    }}
    .oncorag-dot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
    .oncorag-dot-good {{ background: var(--oncorag-good); }}
    .oncorag-dot-warn {{ background: var(--oncorag-warn); }}
    .oncorag-dot-bad  {{ background: var(--oncorag-bad); }}

    /* ---------------------------------------------------------------- */
    /* Header banner - serif headline, animated trace rule as signature   */
    /* ---------------------------------------------------------------- */
    .oncorag-header {{
        position: relative;
        background: var(--oncorag-primary-deep);
        border-radius: 10px;
        overflow: hidden;
        padding: 1.6rem 1.9rem 1.4rem;
        margin-bottom: 1.9rem;
        box-shadow: var(--oncorag-shadow);
        animation: oncorag-rise 0.5s ease both;
    }}
    .oncorag-header::after {{
        content: "";
        position: absolute; left: 0; bottom: 0; height: 3px; width: 220px;
        background: repeating-linear-gradient(
            90deg,
            var(--oncorag-accent) 0px, var(--oncorag-accent) 26px,
            transparent 26px, transparent 40px
        );
        will-change: transform;
        animation: oncorag-trace 5.5s linear infinite;
        opacity: 0.9;
    }}
    .oncorag-header h1 {{
        font-family: var(--font-display) !important;
        color: #ffffff !important;
        margin: 0 0 0.35rem 0 !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px;
        font-size: 2.05rem !important;
    }}
    .oncorag-header p {{
        color: rgba(255,255,255,0.76) !important;
        margin: 0; font-size: 0.93rem; max-width: 66ch;
        font-family: var(--font-body);
    }}

    /* Section headers - serif display face carries the identity, with a
       thin accent rule underneath as a quiet signature detail */
    h2, h3 {{
        font-family: var(--font-display) !important;
        color: var(--oncorag-primary-deep) !important;
        font-weight: 600 !important;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--oncorag-border);
        margin-bottom: 1rem !important;
    }}

    /* ---------------------------------------------------------------- */
    /* Chat bubbles - soft-lifted card, gentle rise-in on new messages   */
    /* ---------------------------------------------------------------- */
    div[data-testid="stChatMessage"] {{
        border-radius: 12px;
        padding: 0.85rem 1.05rem;
        margin-bottom: 0.6rem;
        box-shadow: var(--oncorag-shadow);
        border: 1px solid var(--oncorag-border);
        background: var(--oncorag-surface);
        animation: oncorag-rise 0.35s ease both;
    }}

    /* Avatar circles - flat palette colors instead of Streamlit's default clash */
    div[data-testid="stChatMessageAvatarUser"] {{
        background-color: var(--oncorag-primary) !important;
    }}
    div[data-testid="stChatMessageAvatarAssistant"] {{
        background-color: var(--oncorag-primary-deep) !important;
    }}
    div[data-testid="stChatMessageAvatarUser"] svg,
    div[data-testid="stChatMessageAvatarAssistant"] svg {{ fill: #ffffff !important; }}

    div[data-testid="stChatInput"] {{
        border-radius: 10px;
        box-shadow: none;
        border: 1px solid var(--oncorag-border);
        background: var(--oncorag-surface);
    }}
    div[data-testid="stChatInput"]:focus-within {{
        border-color: var(--oncorag-primary);
        box-shadow: 0 0 0 3px var(--oncorag-primary-soft-bg);
    }}

    /* ---------------------------------------------------------------- */
    /* Buttons - understated, with a real hover lift                     */
    /* ---------------------------------------------------------------- */
    .stButton > button {{
        border-radius: 8px;
        border: 1px solid var(--oncorag-border);
        font-family: var(--font-body);
        font-weight: 600;
        background: var(--oncorag-surface);
        transition: border-color 0.15s ease-in-out, color 0.15s ease-in-out,
                    transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }}
    .stButton > button:hover {{
        border-color: var(--oncorag-primary);
        color: var(--oncorag-primary);
        transform: translateY(-1px);
        box-shadow: var(--oncorag-shadow);
    }}
    .stButton > button:active {{ transform: translateY(0); }}
    .stButton > button[kind="primary"] {{
        background: var(--oncorag-primary);
        border: 1px solid var(--oncorag-primary);
        color: #fff;
        font-weight: 700;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: var(--oncorag-primary-deep);
        border-color: var(--oncorag-primary-deep);
        color: #fff;
    }}

    .stButton > button p {{ font-size: 0.86rem; }}

    /* Give button rows (example-question chips, follow-ups) breathing room */
    div[data-testid="stHorizontalBlock"] {{ gap: 0.6rem; }}

    /* ---------------------------------------------------------------- */
    /* Metrics - lifted card, mono figures                                */
    /* ---------------------------------------------------------------- */
    div[data-testid="stMetric"] {{
        background: var(--oncorag-surface);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        box-shadow: var(--oncorag-shadow);
        border: 1px solid var(--oncorag-border);
        border-left: 3px solid var(--oncorag-primary);
    }}
    div[data-testid="stMetricLabel"] {{ color: var(--oncorag-ink-soft) !important; font-family: var(--font-body) !important; }}
    div[data-testid="stMetricValue"] {{ font-family: var(--font-mono) !important; }}

    /* ---------------------------------------------------------------- */
    /* Expanders (sources)                                                */
    /* ---------------------------------------------------------------- */
    div[data-testid="stExpander"] {{
        border-radius: 10px;
        border: 1px solid var(--oncorag-border);
        overflow: hidden;
        background: var(--oncorag-surface);
        box-shadow: none;
    }}

    div[data-testid="stAlert"] {{ border-radius: 8px; }}

    .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--oncorag-ink-soft) !important; }}

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {{
        border-radius: 10px;
        border: 1.5px dashed var(--oncorag-border);
        background: var(--oncorag-surface);
        transition: border-color 0.15s ease-in-out;
    }}
    section[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--oncorag-primary); }}
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

INDEX_BATCH_SIZE = 40  # chunks embedded/inserted per batch - keeps peak memory down


def build_index_from_chunks(chunks: List, embedding_model, progress_callback=None) -> None:
    """Embed and insert chunks in small batches instead of all at once.

    Encoding hundreds of chunks in a single call spikes memory (the full
    batch of vectors + intermediate tensors sit in RAM at once on top of
    the embedding model itself). Batching keeps peak memory roughly
    constant regardless of corpus size - important on memory-constrained
    hosts like Streamlit Community Cloud's free tier.
    """
    if pipeline.database_exists(PERSIST_DIRECTORY):
        pipeline.delete_vector_store(PERSIST_DIRECTORY)

    total = len(chunks)
    first_batch, rest = chunks[:INDEX_BATCH_SIZE], chunks[INDEX_BATCH_SIZE:]
    vector_store = pipeline.create_vector_store(first_batch, embedding_model, PERSIST_DIRECTORY)
    done = len(first_batch)
    if progress_callback:
        progress_callback(done, total)

    # Only continue in batches if the pipeline handed back a store we can
    # append to (langchain's Chroma wrapper exposes add_documents). If not,
    # fall back to a single call with everything - still correct, just
    # without the memory benefit for this one run.
    if rest:
        if vector_store is not None and hasattr(vector_store, "add_documents"):
            for i in range(0, len(rest), INDEX_BATCH_SIZE):
                batch = rest[i : i + INDEX_BATCH_SIZE]
                vector_store.add_documents(batch)
                done += len(batch)
                if progress_callback:
                    progress_callback(done, total)
        else:
            pipeline.create_vector_store(chunks, embedding_model, PERSIST_DIRECTORY)
            if progress_callback:
                progress_callback(total, total)


# --------------------------------------------------------------------------
# System status
# --------------------------------------------------------------------------
chunk_count = get_chunk_count(PERSIST_DIRECTORY)
doc_count = get_doc_count()

# If PDFs already exist in the documents folder (e.g. shipped in the repo)
# but no index exists yet - typically because the persisted vector store
# doesn't survive a fresh deploy on Streamlit Cloud - build it once
# automatically instead of forcing a manual click on every cold start.
if (
    chunk_count == 0
    and has_api_key
    and not pipeline._load_errors
    and not st.session_state.get("_auto_build_attempted")
):
    st.session_state._auto_build_attempted = True
    existing_pdfs_at_start = sorted(Path(DOCUMENTS_DIR).glob("*.pdf")) if os.path.isdir(DOCUMENTS_DIR) else []
    if existing_pdfs_at_start:
        with st.spinner("Setting up the knowledge base for the first time..."):
            try:
                docs = pipeline.load_all_documents(DOCUMENTS_DIR)
                cleaned = pipeline.preprocess_documents(docs)
                chunks = pipeline.chunk_documents(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)
                embedding_model = get_embedding_model()
                build_index_from_chunks(chunks, embedding_model)
                refresh_index_caches()
                chunk_count = get_chunk_count(PERSIST_DIRECTORY)
                doc_count = get_doc_count()
            except Exception:
                pass  # fall through - Knowledge Base page still offers a manual rebuild

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


def _call_with_timeout(fn, timeout_seconds: float, *args, **kwargs):
    """Run fn(*args, **kwargs) in a worker thread and bail after timeout_seconds.

    Raises on both real exceptions and timeouts (concurrent.futures.TimeoutError
    inherits from Exception too) - callers should catch broadly. The
    worker thread is not joined - if it's stuck on a network call it's
    left to finish or die on its own rather than blocking the app.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    finally:
        executor.shutdown(wait=False)


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
        raw = _call_with_timeout(
            pipeline.generate_answer, FOLLOWUP_TIMEOUT_SECONDS, api_key, model, prompt, max_output_tokens=150
        )
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
    try:
        return _call_with_timeout(
            pipeline.generate_answer, TRANSLATE_TIMEOUT_SECONDS, api_key, model, prompt, max_output_tokens=800
        )
    except Exception:
        return "_Translation timed out or failed - please try again._"


@st.cache_data(show_spinner=False, max_entries=20)
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


@st.cache_data(show_spinner=False)
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
        pdf.multi_cell(0, 7, _latin1(f"{role_label}:"), wrapmode=WrapMode.CHAR)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _latin1(message["content"]), wrapmode=WrapMode.CHAR)

        if message.get("sources"):
            pdf.set_font("Helvetica", "I", 9)
            for src in message["sources"]:
                pdf.multi_cell(
                    0, 5,
                    _latin1(f"  - Source: {src['filename']} (page {src['page']})"),
                    wrapmode=WrapMode.CHAR,
                )
        pdf.ln(3)

    return bytes(pdf.output())


def render_source_block(sources: List[Dict], key_prefix: str) -> None:
    """Render the expandable sources list, each with an on-demand page preview."""
    with st.expander(f"Sources ({len(sources)})"):
        for i, src in enumerate(sources):
            st.markdown(
                f"**{src['filename']}** &nbsp;"
                f"<span style='font-family:var(--font-mono);font-size:0.82rem;color:var(--oncorag-ink-soft);'>"
                f"p.{src['page']} · {src['similarity']:.0%} match</span>",
                unsafe_allow_html=True,
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

    all_sources = [
        {
            "filename": doc.metadata.get("filename", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "source_path": doc.metadata.get("source", ""),
            "text": doc.page_content,
            "similarity": _distance_to_similarity(distance),
        }
        for doc, distance in retrieved_with_scores
    ]

    # Chroma always returns k results even when nothing is actually
    # relevant. Drop chunks below MIN_SOURCE_SIMILARITY so the UI never
    # presents "noise" chunks as if they were real matches.
    sources = [src for src in all_sources if src["similarity"] >= MIN_SOURCE_SIMILARITY]

    avg_similarity = (
        sum(src["similarity"] for src in sources) / len(sources) if sources else 0.0
    )

    return {
        "prompt": prompt,
        "api_key": api_key,
        "model": model,
        "sources": sources,
        "avg_similarity": avg_similarity,
        "no_relevant_match": len(sources) == 0,
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
        "no_relevant_match": prep["no_relevant_match"],
        "follow_ups": follow_ups,
    }


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.1rem;'>"
        "<span style='font-size:1.5rem;'>🩺</span>"
        "<span style='font-size:1.3rem;font-weight:800;letter-spacing:-0.4px;'>OncoRAG</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Oncology Clinical Knowledge Assistant")
    st.toggle("Dark mode", key="dark_mode")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Chat", "Knowledge Base", "Settings"],
        format_func=lambda p: {"Chat": "💬  Chat", "Knowledge Base": "📚  Knowledge Base", "Settings": "⚙️  Settings"}[p],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**System status**")

    index_dot = "oncorag-dot-good" if chunk_count > 0 else "oncorag-dot-bad"
    key_dot = "oncorag-dot-good" if has_api_key else "oncorag-dot-bad"
    badges_html = (
        f"<span class='oncorag-badge'><span class='oncorag-dot {index_dot}'></span>"
        f"Index: {'Ready' if chunk_count > 0 else 'Not built'} ({chunk_count} chunks)</span>"
        f"<span class='oncorag-badge'><span class='oncorag-dot {key_dot}'></span>"
        f"API key: {'Configured' if has_api_key else 'Missing'}</span>"
    )
    if pipeline._load_errors:
        badges_html += (
            "<span class='oncorag-badge'><span class='oncorag-dot oncorag-dot-warn'></span>"
            "Pipeline: load errors</span>"
        )
    st.markdown(badges_html, unsafe_allow_html=True)

    if pipeline._load_errors:
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
        try:
            pdf_bytes = build_session_pdf(st.session_state.chat_history)
        except Exception:
            pdf_bytes = None
            st.caption("Session PDF export isn't available right now.")
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

st.markdown(
    """
    <div class="oncorag-header">
        <h1>🩺 OncoRAG</h1>
        <p>Ask grounded, cited questions against your uploaded oncology literature.
        This tool summarizes documents — it does not diagnose or make treatment decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
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
        with st.chat_message(message["role"], avatar=AVATAR_USER if message["role"] == "user" else AVATAR_ASSISTANT):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                if message.get("sources"):
                    conf = message.get("avg_similarity", 0.0)
                    conf_class = (
                        "oncorag-dot-good" if conf >= LOW_CONFIDENCE_THRESHOLD else "oncorag-dot-warn"
                    )
                    st.markdown(
                        f"<span class='oncorag-badge' style='background:var(--oncorag-primary-soft-bg);"
                        f"border-color:var(--oncorag-primary-soft-border);color:var(--oncorag-ink) !important;'>"
                        f"<span class='oncorag-dot {conf_class}'></span>Confidence: {conf:.0%}</span>",
                        unsafe_allow_html=True,
                    )
                    if conf < LOW_CONFIDENCE_THRESHOLD:
                        st.warning(
                            "Low retrieval confidence - the source material for this "
                            "answer was only weakly related to the question. Verify "
                            "against primary literature before relying on it."
                        )
                    render_source_block(message["sources"], key_prefix=f"msg{idx}")
                elif message.get("no_relevant_match"):
                    st.caption("No matching passages found in the knowledge base for this question.")

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
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
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
                    "no_relevant_match": prep["no_relevant_match"],
                    "follow_ups": follow_ups,
                }
            except Exception as exc:  # noqa: BLE001
                result = {
                    "answer": f"The assistant hit an error and could not generate an answer: {exc}",
                    "sources": [],
                    "avg_similarity": 0.0,
                    "no_relevant_match": True,
                    "follow_ups": [],
                }
                st.markdown(result["answer"])

            if result["sources"]:
                conf = result["avg_similarity"]
                conf_class = "oncorag-dot-good" if conf >= LOW_CONFIDENCE_THRESHOLD else "oncorag-dot-warn"
                st.markdown(
                    f"<span class='oncorag-badge' style='background:var(--oncorag-primary-soft-bg);"
                    f"border-color:var(--oncorag-primary-soft-border);color:var(--oncorag-ink) !important;'>"
                    f"<span class='oncorag-dot {conf_class}'></span>Confidence: {conf:.0%}</span>",
                    unsafe_allow_html=True,
                )
                if conf < LOW_CONFIDENCE_THRESHOLD:
                    st.warning("Low retrieval confidence - verify against primary literature.")
                render_source_block(result["sources"], key_prefix="latest")
            elif result["no_relevant_match"]:
                st.caption("No matching passages found in the knowledge base for this question.")

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "avg_similarity": result["avg_similarity"],
                "no_relevant_match": result["no_relevant_match"],
                "follow_ups": result["follow_ups"],
            }
        )
        st.session_state.conversation_history.append((user_query, result["answer"]))
        st.rerun()

# --------------------------------------------------------------------------
# Knowledge Base page
# --------------------------------------------------------------------------
elif page == "Knowledge Base":
    st.subheader("📚 Knowledge Base")
    st.write(
        "Upload oncology PDFs (guidelines, staging references, trial "
        "papers, protocols), then build the index so the Chat tab can "
        "retrieve grounded, cited answers from them."
    )

    col_a, col_b = st.columns(2)
    col_a.metric("PDF pages loaded", doc_count)
    col_b.metric("Indexed chunks", chunk_count)

    def build_index() -> bool:
        """Load -> preprocess -> chunk -> embed -> index (in memory-friendly batches)."""
        try:
            with st.spinner("Loading and chunking documents..."):
                docs = pipeline.load_all_documents(DOCUMENTS_DIR)
                cleaned = pipeline.preprocess_documents(docs)
                chunks = pipeline.chunk_documents(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)
                embedding_model = get_embedding_model()

            progress_bar = st.progress(0.0, text="Embedding and indexing 0 / 0 chunks...")

            def _on_progress(done: int, total: int) -> None:
                progress_bar.progress(
                    done / total if total else 1.0,
                    text=f"Embedding and indexing {done} / {total} chunks...",
                )

            build_index_from_chunks(chunks, embedding_model, progress_callback=_on_progress)
            progress_bar.empty()

            refresh_index_caches()
            st.session_state.processed_pdf_names = {p.name for p in Path(DOCUMENTS_DIR).glob("*.pdf")}
            st.success(f"Index built successfully: {len(chunks)} chunks indexed.")
            return True
        except Exception as exc:  # noqa: BLE001
            st.error(f"Index build failed: {exc}")
            return False

    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    if "processed_pdf_names" not in st.session_state:
        # Assume whatever's already on disk was indexed in a prior session/run.
        st.session_state.processed_pdf_names = (
            {p.name for p in Path(DOCUMENTS_DIR).glob("*.pdf")} if chunk_count > 0 else set()
        )

    uploaded_files = st.file_uploader(
        "Upload oncology PDFs", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        new_names = []
        for uploaded_file in uploaded_files:
            with open(os.path.join(DOCUMENTS_DIR, uploaded_file.name), "wb") as out_file:
                out_file.write(uploaded_file.getbuffer())
            new_names.append(uploaded_file.name)
        get_doc_count.clear()

        # Auto-chunk/embed/index right away - no separate manual step needed.
        unindexed = set(new_names) - st.session_state.processed_pdf_names
        if unindexed:
            st.success(f"Saved {len(uploaded_files)} file(s). Indexing automatically...")
            if build_index():
                st.rerun()
        else:
            st.info("These file(s) are already indexed.")

    existing_pdfs = sorted(Path(DOCUMENTS_DIR).glob("*.pdf"))
    if existing_pdfs:
        with st.expander(f"Documents currently in the folder ({len(existing_pdfs)})"):
            for pdf_path in existing_pdfs:
                indexed = pdf_path.name in st.session_state.processed_pdf_names
                st.caption(f"{'✅' if indexed else '⏳'} {pdf_path.name}")

    st.divider()
    if st.button("Rebuild Index", use_container_width=True):
        if not existing_pdfs:
            st.error("No PDF files found in the documents folder. Upload some first.")
        elif build_index():
            st.rerun()

# --------------------------------------------------------------------------
# Settings page
# --------------------------------------------------------------------------
elif page == "Settings":
    st.subheader("⚙️ Settings")
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
