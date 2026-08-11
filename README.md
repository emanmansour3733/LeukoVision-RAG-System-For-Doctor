# OncoRAG — Oncology Clinical Knowledge Assistant

OncoRAG is a Retrieval-Augmented Generation (RAG) system that answers
questions about a user-supplied library of oncology literature —
clinical guidelines, staging references, trial data, and treatment
protocols — with grounded, page-cited answers. The system is delivered
as a Streamlit application backed by a modular ingestion, indexing, and
retrieval pipeline.

## Architecture

```
01_documents.py               PDF ingestion -> Document objects
02_preprocessing.py           Text cleaning, metadata preservation
03_chunking.py                Semantic chunking with overlap
04_vector_representation.py   Embedding model (sentence-transformers/all-MiniLM-L6-v2)
05_create_chroma_store.py     Chroma vector store: create, persist, load, query
06_retrieve_context.py        Similarity search, context and citation formatting
07_prompting.py                System and user prompt construction
08_llm.py                     LLM communication via the OpenRouter API
streamlit_app.py              Application UI: Chat, Knowledge Base, Settings
```

Each pipeline module has a single, isolated responsibility, and no
module performs work outside its stated scope (for example,
`06_retrieve_context.py` does not call an LLM, and `07_prompting.py`
does not touch the vector store). This keeps the pipeline testable
stage by stage and makes it straightforward to swap any single
component — embedding model, vector store, or LLM provider — without
touching the others.

## Capabilities

- **Grounded, cited answers.** Every response is generated strictly from
  retrieved context and cites filename and page number for each source
  used.
- **Multi-turn conversation memory.** Follow-up questions are interpreted
  using recent conversation history, while answers are still generated
  only from retrieved context, never from unretrieved prior turns.
- **Confidence flagging.** Answers backed by weakly related source
  material are flagged for independent verification instead of being
  presented with unwarranted confidence.
- **Source page verification.** Each citation can be expanded to display
  the actual cited PDF page as a rendered image.
- **Suggested follow-up questions.** Each answer is accompanied by three
  proposed next questions.
- **On-demand Arabic translation.** Any answer can be translated into
  clinical Arabic on request, with citations left unmodified.
- **Session export.** The full conversation, including citations, can be
  exported as a PDF document.
- **Streamed responses.** Answers are streamed token-by-token from the
  LLM as they are generated, rather than returned as a single blocking
  call.

## Configuration

Credentials are never stored in source files. They are read from
environment variables, populated either by a local `.env` file (see
`.env.example`) or by Streamlit secrets in a deployed environment.

Required configuration:

```
OPENROUTER_API_KEY   OpenRouter API key
OPENROUTER_MODEL     Model identifier (default: openai/gpt-4o-mini)
```

## Local setup

```bash
pip install -r requirements.txt
cp .env.example .env
# populate OPENROUTER_API_KEY in .env
streamlit run streamlit_app.py
```

The `documents/` directory ships with a set of sample oncology
reference PDFs so the pipeline can be exercised end to end immediately.
In the running application, open **Knowledge Base**, confirm the
documents are listed (or upload additional PDFs), and select
**Build / Rebuild Index**. Once the index is built, use **Chat** to
query the knowledge base.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub. `.env` is excluded via `.gitignore`
   and must never be committed.
2. Create a new app on Streamlit Community Cloud pointing at this
   repository, with `streamlit_app.py` as the entry point.
3. Under **Manage app -> Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "your_openrouter_key"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```
4. Once deployed, open **Knowledge Base** and select
   **Build / Rebuild Index**.

On the free tier, the filesystem is ephemeral: the vector index is
lost on redeploy or restart and must be rebuilt from the **Knowledge
Base** page afterward. This is expected behavior on that tier, not a
defect.

## Document sourcing

The bundled `documents/` folder contains original, self-authored
educational reference material (acute leukemia overview, WHO
classification summary, CBC interpretation, staging systems, systemic
therapy principles, oncologic emergencies) intended to demonstrate the
pipeline end to end. For clinical or production use, replace these with
the licensed guideline documents (NCCN, WHO, ASCO, ESMO, or
institutional protocols) your organization is authorized to use. Public
domain sources (for example, NCI PDQ summaries) and open-access
literature (for example, PubMed Central) are also compatible with the
ingestion pipeline without licensing restrictions.

## Operating constraints

The system prompt enforces the following at the model level:

- Answers are generated exclusively from retrieved context; no external
  or parametric medical knowledge is used to fill gaps.
- Insufficient retrieved context is explicitly reported rather than
  answered speculatively.
- Every factual claim is cited by filename and page.
- The system does not diagnose patients, recommend patient-specific
  treatment, or claim clinical judgment; it summarizes literature only.
- Dosage, contraindication, and drug-interaction information is flagged
  as literature-reported and subject to independent verification before
  clinical use.

## Requirements

See `requirements.txt`. Core dependencies: `streamlit`, `langchain`
(with `langchain-community`, `langchain-chroma`, `langchain-huggingface`,
`langchain-text-splitters`), `chromadb`, `sentence-transformers`,
`pypdf`, `pymupdf`, `fpdf2`, `requests`.
