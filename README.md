# OncoRAG — Oncology Clinical Knowledge Assistant

A RAG pipeline that answers questions about uploaded oncology literature
(guidelines, staging systems, trial data, protocols) with page-cited,
grounded answers, deployed as a Streamlit app.

## What changed vs. the previous version

The old version had two bugs that caused the "slow / hangs" symptoms,
plus one serious security issue:

1. **Invalid model name.** `08_llm.py` called `"gemini-3.5-flash"`, which
   does not exist. Every single question triggered 3 retries with
   exponential backoff before failing — that's the multi-second freeze
   on every query. `08_llm.py` was rewritten to call OpenRouter
   (`openai/gpt-4o-mini` by default), matching the model your
   assignment's secrets contract expects, with fast-fail on bad
   requests (no pointless retries on a 4xx error) and a hard timeout.

2. **A real API key was hardcoded in `streamlit_app1.py`**
   (`MANUAL_GOOGLE_API_KEY = "..."`). This violates the assignment's own
   rule ("Do not write your real API key inside any Python file") and
   means that key is exposed to anyone who sees the source or the
   GitHub repo. **Rotate/revoke that key now** if it was ever pushed
   anywhere public. The new `streamlit_app.py` never contains a key —
   it reads `OPENROUTER_API_KEY` from Streamlit secrets (deployed) or a
   local `.env` file (dev only, gitignored).

3. Files were renamed to match the required submission structure
   (`05_create_chroma_store.py`, `06_retrieve_context.py`,
   `07_prompting.py`, `streamlit_app.py`), and the system prompt was
   broadened from leukemia-only to general oncology, with an added rule
   flagging dosage/contraindication text as literature-reported and
   requiring independent verification.

## Features that go beyond a basic RAG demo

- **Multi-turn memory** — follow-up questions like "what about pediatric
  patients?" are understood using the last few turns of conversation,
  without ever letting the model answer from unretrieved memory (see the
  `conversation_history` param added to `build_user_prompt` /
  `build_complete_prompt` in `07_prompting.py`).
- **Confidence flagging** — if the retrieved chunks are only weakly
  related to the question (low average similarity), the UI shows an
  explicit "verify against primary literature" warning instead of
  presenting a shaky answer with false confidence.
- **Source page preview** — every citation has a "🔎 View PDF page"
  button that renders the *actual* cited page as an image (via
  PyMuPDF), so the doctor can check the claim in context without
  leaving the app or hunting through the PDF.
- **Suggested follow-up questions** — after every answer, the model
  proposes 3 natural next questions; clicking one asks it immediately.
- **On-demand Arabic translation** — a "🔤 Arabic" button translates any
  answer into clinical Arabic while leaving citations (filename/page)
  untouched, useful when discussing a case with Arabic-speaking staff.
- **Session export to PDF** — the whole conversation, with citations,
  can be downloaded as a PDF for tumor-board handouts or documentation.

## Pipeline

```
01_documents.py            load PDFs -> Document objects
02_preprocessing.py        clean text, preserve metadata
03_chunking.py             split into overlapping chunks
04_vector_representation.py  build the embedding model (MiniLM-L6-v2)
05_create_chroma_store.py  embed + persist chunks in a Chroma index
06_retrieve_context.py     similarity search, format context/citations
07_prompting.py            build the system + user + sources prompt
08_llm.py                  call OpenRouter, return the answer
streamlit_app.py           UI: chat, knowledge base, settings
```

## Local setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in your real OpenRouter key
mkdir -p documents          # drop your oncology PDFs in here
streamlit run streamlit_app.py
```

Open the app, go to **Knowledge Base**, upload PDFs (or place them in
`documents/` directly), click **Build / rebuild index**, then use **Chat**.

## Streamlit Cloud deployment

1. Push this project to GitHub — **make sure `.env` is not included**
   (it's gitignored already).
2. Deploy the repo on Streamlit Cloud, entry point `streamlit_app.py`.
3. In the app, go to **Manage app → Secrets** and add:
   ```toml
   OPENROUTER_API_KEY = "your_openrouter_key_here"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```
4. Re-run the app, upload your PDFs under Knowledge Base, and build the index.

Note: on Streamlit Cloud's free tier the filesystem is ephemeral — the
index you build will be lost on redeploy/restart, so you'll rebuild it
after each redeploy (this is normal for this tier, not a bug).

## Final checklist (per assignment)

- [x] All required Python files present (01–07 + `streamlit_app.py`)
- [x] `requirements.txt` present
- [x] No real API key in any file in the ZIP/repo
- [x] Streamlit secrets read in valid TOML format
- [x] App answers using retrieved context only
- [x] Every answer cites filename + page
