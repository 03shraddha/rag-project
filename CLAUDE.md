# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the CLI demo (one-shot, runs hardcoded test queries)
python rag_system.py

# Run the web UI (FastAPI + single-page frontend)
uvicorn api:app --reload
# Then open http://localhost:8000
```

There are no tests, linter, or build steps configured.

## Architecture

This is a from-scratch RAG (Retrieval-Augmented Generation) system with two entry points:

**`rag_system.py`** — Standalone CLI script. Runs the full pipeline linearly: load knowledge → chunk → embed → index → query → generate. Executes hardcoded test queries and prints results. Not imported by anything else.

**`api.py`** — FastAPI web server that exposes the same RAG pipeline as an API. Uses a `rag` global dict to hold all state (chunks, embeddings, FAISS index, models). The `build_index()` function populates this dict at startup via the lifespan handler and is also called when knowledge is updated via POST.

**`static/index.html`** — Single-file frontend (HTML/CSS/JS, no build tools). Dark terminal-aesthetic dashboard. Left panel shows editable knowledge base and chunk cards; right panel has query input, animated pipeline visualization, and expandable step-by-step results with timing.

### RAG Pipeline (4 steps)

1. **Embed** — `SentenceTransformer('all-MiniLM-L6-v2')` encodes query to 384-dim vector
2. **Retrieve** — `faiss.IndexFlatL2` finds top-k nearest chunks by L2 distance
3. **Augment** — Builds a grounded prompt with retrieved context + instruction to refuse if answer not in context
4. **Generate** — `google/flan-t5-small` via HuggingFace transformers produces the answer

### API Endpoints

- `GET /` — serves the frontend
- `GET /api/health` — system stats (chunk count, embedding dim, index size)
- `GET /api/chunks` — all chunks with embedding previews (first 8 dims)
- `GET /api/knowledge` — raw knowledge text
- `POST /api/knowledge` — replace knowledge text and rebuild the entire index
- `POST /api/query` — run full RAG pipeline, returns answer + per-step timing + intermediate outputs

### Key Data Flow

`my_knowledge.txt` → `RecursiveCharacterTextSplitter` (chunk_size=150, overlap=20) → sentence-transformers encode → FAISS index. Query embeds → FAISS search → context injected into prompt template → FLAN-T5 generates answer.

## Platform

Windows environment. Use `ls` (not `dir /b`) in shell commands.
