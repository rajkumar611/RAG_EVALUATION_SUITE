# RAG Evaluation Suite

An interactive FastAPI application demonstrating **5 RAG retrieval strategies**, **10 LangChain orchestration patterns**, and **LLM-based RAG evaluation** — built to production-grade standards.

---

## Why This Project

Most RAG demos are toy examples. This project shows:
- **Retrieval depth** — five strategies from naive to graph-based, each with visible pipeline steps
- **Orchestration breadth** — ten LangChain patterns from prompt templates to LangGraph stateful workflows
- **Evaluation** — LLM-as-Judge scoring across Faithfulness, Answer Relevancy, and Context Utilization
- **Observability** — LangSmith tracing integration (optional, zero-config when API key is set)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     index.html (SPA)                    │
│          RAG Strategies │ Evaluation │ LangChain         │
└────────────────┬──────────────┬───────────────┬─────────┘
                 │              │               │
         ┌───────▼──────┐  ┌───▼────┐  ┌───────▼──────────┐
         │  rag/routes  │  │  eval  │  │ langchain_        │
         │              │  │ (same) │  │ orchestration/    │
         │ • naive      │  └────────┘  │ routes            │
         │ • advanced   │              │                   │
         │ • agentic    │              │ • prompt templates │
         │ • hybrid     │              │ • chaining        │
         │ • graph      │              │ • RAG             │
         │ • evaluate   │              │ • memory          │
         └──────┬───────┘              │ • tools           │
                │                      │ • documents       │
         ┌──────▼───────┐              │ • parsers         │
         │ Anthropic SDK│              │ • agent (ReAct)   │
         │ claude-sonnet│              │ • multi-agent     │
         │ (direct)     │              │ • LangGraph       │
         └──────────────┘              └────────┬──────────┘
                                                │
                                       ┌────────▼────────┐
                                       │ langchain-       │
                                       │ anthropic        │
                                       │ claude-haiku     │
                                       └─────────────────┘
```

---

## RAG Strategies

| Strategy | When to Use | Key Technique |
|---|---|---|
| **Naive RAG** | Baseline / simple Q&A | Embed → cosine search → generate |
| **Advanced RAG** | Production Q&A | Query rewrite + hybrid search + LLM re-rank |
| **Agentic RAG** | Multi-hop questions | Tool-calling agent with iterative search |
| **Hybrid RAG** | Keyword + semantic needs | Dense (FAISS) + Sparse (BM25) fused via RRF |
| **Graph RAG** | Relational / connected data | Seed retrieval + BFS graph expansion |

---

## LangChain Orchestration Patterns

| Pattern | Production Use Case |
|---|---|
| Prompt Templates | Standardised, reusable prompt management |
| Chaining | Multi-step pipelines (translate → summarise → format) |
| RAG | Document Q&A with FAISS retriever |
| Memory | Session-aware conversation with `MessagesPlaceholder` |
| Tools | External API integration via `bind_tools` |
| Document Splitters | `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter` |
| Output Parsers | `StrOutputParser`, `JsonOutputParser`, `CommaSeparatedListOutputParser` |
| Agent (ReAct) | Autonomous reasoning with `create_react_agent` |
| Multi-Agent | Role-based sequential agents (researcher → writer) |
| LangGraph | Stateful workflows with conditional edges and revision loops |

---

## RAG Evaluation (LLM-as-Judge)

Three RAGAS-aligned metrics evaluated by Claude:

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Is the answer grounded in context? (hallucination detection) |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Utilization** | Did the retrieved chunks contain the right information? |
| **Correctness** *(optional)* | Accuracy vs ground truth answer |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 3. (Optional) Enable LangSmith tracing
# Add LANGCHAIN_API_KEY to .env — tracing activates automatically

# 4. Run
python main.py
# → http://localhost:8000
```

---

## Project Structure

```
.
├── app.py                              # FastAPI app — mounts routers, enables LangSmith
├── main.py                             # Uvicorn entrypoint
├── requirements.txt
├── .env.example
│
├── src/                                # All application source code
│   ├── rag/
│   │   └── routes.py                  # 5 RAG strategies + /rag/evaluate endpoint
│   └── langchain_orchestration/
│       └── routes.py                  # 10 LangChain orchestration patterns
│
├── frontend/
│   └── index.html                     # Single-file SPA
│
└── tests/                             # Test suite
```

---

## Key Design Decisions

- **Two Claude models intentionally**: RAG uses `claude-sonnet-4-6` (higher reasoning for retrieval tasks); LangChain demos use `claude-haiku-4-5` (faster, cheaper for concept demos)
- **LLM-as-Judge over RAGAS library**: Avoids heavy dependencies; the same evaluation principle, implemented directly with the Anthropic SDK
- **LangSmith zero-config**: Set `LANGCHAIN_API_KEY` in `.env` — all LangChain calls trace automatically, no code changes required
- **All state in-memory**: Intentional for simplicity; upload a PDF before using RAG endpoints (resets on server restart)

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Anthropic Claude (Sonnet + Haiku) |
| Orchestration | LangChain + LangGraph |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense Search | FAISS |
| Sparse Search | TF-IDF / BM25 (scikit-learn) |
| Graph | NetworkX |
| Observability | LangSmith (optional) |

---

## Author

**Raj Kumar** — [rajkumar.novsix@gmail.com](mailto:rajkumar.novsix@gmail.com)
