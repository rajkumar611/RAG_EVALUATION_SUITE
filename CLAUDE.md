# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**AI Learning Hub** — a FastAPI web app that demonstrates RAG strategies and LangChain concepts interactively. A single `index.html` frontend calls backend endpoints; results are displayed in the browser. No database, no auth, no tests.

## Running the app

```bash
pip install -r requirements.txt   # first time only (downloads ~80 MB embedding model)
python main.py                    # serves on http://localhost:8000
```

`main.py` loads `.env` then delegates to `uvicorn app:app`. The `.env` must contain `ANTHROPIC_API_KEY`.

## Architecture

```
app.py          ← FastAPI app; mounts both routers; serves index.html at /
rag/routes.py   ← 5 RAG strategies + /upload
lc/routes.py    ← 10 LangChain concept demos
index.html      ← entire SPA (single file, ~175 KB)
extracted.js    ← JS extracted from index.html (reference artifact, not loaded by the app)
```

### RAG module (`rag/routes.py`)

Uses the **Anthropic SDK directly** (`anthropic.Anthropic()`, model `claude-sonnet-4-6`).

Global in-memory state holds the uploaded document corpus — **reset on every server restart**. Upload a PDF or TXT via `POST /upload` before calling any RAG endpoint.

| Endpoint | Strategy |
|---|---|
| `POST /rag/naive` | Embed → vector search → generate |
| `POST /rag/advanced` | Query rewrite → hybrid search → RRF → LLM re-rank → generate |
| `POST /rag/agentic` | Tool-calling agent that searches iteratively (up to 2 rounds, `MAX_SEARCH_ROUNDS = 2`) |
| `POST /rag/hybrid` | Dense (FAISS cosine) + sparse (TF-IDF BM25) fused via RRF |
| `POST /rag/graph` | Seed retrieval + 2-hop BFS on a sequential graph → re-score |

Shared utilities in the same file: `vsearch` (dense), `bsearch` (TF-IDF), `rrf` (Reciprocal Rank Fusion), `chunk_text`, `llm`, `rebuild_indexes`, `no_docs_response`, `ctx_prompt`.

Global state variables: `DOCS` (chunks), `DOC_EMBS` (numpy embeddings), `TFIDF_MAT` (sparse matrix), `G` (NetworkX sequential graph), `embedder` (`all-MiniLM-L6-v2`, 384-dim, loaded at startup), `tfidf` (TfidfVectorizer).

Upload limits: max 300 chunks, max chunk size 400 chars.

### LangChain module (`lc/routes.py`)

Uses **`langchain-anthropic`** (`ChatAnthropic`, model `claude-haiku-4-5-20251001`).

All imports are deferred (inside each route function) — this keeps startup fast and makes each endpoint self-contained.

| Endpoint | Concept |
|---|---|
| `POST /lc/prompt` | Prompt templates + `StrOutputParser` |
| `POST /lc/chaining` | Sequential 3-step chain (translate → summarise → JSON) |
| `POST /lc/rag` | RAG with a fixed 7-doc FAISS vectorstore (lazy-loaded, in-memory) |
| `POST /lc/memory` | Per-session conversation history via `MessagesPlaceholder` |
| `POST /lc/tools` | `bind_tools` with calculator / weather / word-count |
| `POST /lc/documents` | `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter` |
| `POST /lc/parsers` | `StrOutputParser`, `JsonOutputParser`, `CommaSeparatedListOutputParser` |
| `POST /lc/agent` | ReAct agent via `langgraph.prebuilt.create_react_agent` |
| `POST /lc/multiagent` | Two sequential LLM calls (researcher → blog writer) |
| `POST /lc/langgraph` | `StateGraph` with manager → research → writer → reviewer + conditional edge (max 2 revisions) |

`LC_SESSIONS` (dict) stores per-session chat history in memory — cleared with `DELETE /lc/memory/{session_id}`.

## Key constraints

- **All state is in-memory.** Restarting the server clears uploaded docs and chat sessions.
- The RAG module and LangChain module use **different Claude models** — don't conflate them.
- The LangChain fixed vectorstore (`_lc_vectorstore`) contains only 7 hardcoded AI/ML facts; it is not the uploaded document store.
- `index.html` is a single-file SPA — all JS, CSS, and HTML in one file. Edit it directly.
- `app.py` enables CORS with wildcard (`allow_origins=["*"]`) — fine for local dev, not for production.
- The agentic RAG tool (`search_knowledge_base`) accepts `top_k` (default 3, max 5); the agent runs at most 2 search rounds.
- LangChain agent tools: `calculator`, `get_exchange_rate`, `get_country_info`. LangChain tools endpoint tools: `calculator`, `get_weather`, `word_count`.
