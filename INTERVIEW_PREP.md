# Interview Prep — RAG & LangChain

Based on the AI Learning Hub project.

---

## RAG (Retrieval-Augmented Generation)

### Fundamentals

**Q: What is RAG and why do we use it?**
RAG (Retrieval-Augmented Generation) combines a retrieval step with an LLM generation step. Instead of relying solely on the model's training data, we fetch relevant context from an external document store and include it in the prompt. This lets the model answer questions about private or up-to-date documents it was never trained on, while also reducing hallucinations by grounding responses in retrieved facts.

---

**Q: Walk me through the naive RAG pipeline in this project.**
1. At upload time, the document is chunked (max 400 chars), embedded with `all-MiniLM-L6-v2`, and stored as numpy arrays.
2. At query time, the query is embedded using the same model.
3. Cosine similarity is computed between the query embedding and all chunk embeddings (`vsearch`, k=3).
4. The top-3 chunks are injected into the prompt as context.
5. The prompt is sent to `claude-sonnet-4-6` via the Anthropic SDK, and the answer is returned.

---

**Q: What embedding model is used and why?**
`all-MiniLM-L6-v2` from Sentence Transformers. It produces 384-dimensional dense vectors and is a strong balance between speed, size (~80 MB), and retrieval quality. It's widely used for semantic search in production-grade RAG systems.

---

**Q: What is the difference between dense and sparse retrieval?**
- **Dense retrieval** (`vsearch`): Uses neural embeddings (vectors) and cosine similarity. Captures semantic meaning — "car" and "automobile" are close in vector space.
- **Sparse retrieval** (`bsearch`): Uses TF-IDF (BM25-style). Relies on exact keyword overlap weighted by term frequency and inverse document frequency. Fast and interpretable.
- Dense is better for paraphrase matching; sparse is better for exact keyword recall. Hybrid combines both.

---

**Q: What is Reciprocal Rank Fusion (RRF) and how is it used here?**
RRF merges ranked lists from multiple retrieval sources without needing calibrated scores. For each document, its RRF score is the sum of `1 / (rank + k)` across all lists it appears in (k=60 is a smoothing constant that prevents very high scores for rank-1 hits). In this project, dense and BM25 results are each ranked, then RRF fuses them into a single ranked list. This is used in both the `advanced` and `hybrid` RAG endpoints.

---

**Q: How does the hybrid RAG endpoint differ from the naive one?**
Naive RAG uses only dense vector search (k=3). Hybrid RAG runs dense search (k=5) AND TF-IDF BM25 search (k=5) in parallel, then fuses the two ranked lists using RRF, keeping the top 3 results. Each retrieved chunk is annotated with its source (`[vector]` or `[bm25]`), making the retrieval more robust to both semantic and keyword-based queries.

---

**Q: What is advanced RAG and what extra steps does it add over naive RAG?**
Advanced RAG adds three extra stages:
1. **Query rewriting** — the LLM first rewrites the user query into a more search-friendly form.
2. **Hybrid retrieval + RRF** — runs both dense and sparse retrieval and fuses results.
3. **LLM re-ranking** — the fused candidates are sent to the LLM which outputs a JSON-ranked list of the most relevant doc IDs. Only the re-ranked top docs are used for generation.

---

**Q: What is agentic RAG and when would you use it over naive RAG?**
Agentic RAG gives the LLM a `search_knowledge_base` tool and lets it decide when and how to search, iteratively. The agent can issue multiple search queries with different phrasings (up to `MAX_SEARCH_ROUNDS = 2`), accumulate results, and decide when it has enough context to answer. Use agentic RAG for multi-part questions or when the right search query isn't obvious upfront — it trades latency for retrieval quality.

---

**Q: Explain the graph RAG approach.**
1. An initial vector seed search retrieves 2 closely related chunks.
2. A sequential graph `G` is built at upload time where each chunk is a node and edges connect adjacent chunks (d0→d1, d1→d2, etc.), preserving document order.
3. From each seed node, a 2-hop BFS traversal finds neighboring chunks.
4. All visited chunks are re-scored by cosine similarity against the query, and the top 3 are used for generation.

This is useful when the answer requires context that spans multiple consecutive passages, not just isolated chunks.

---

**Q: What are the tradeoffs between chunk size and retrieval quality?**
- **Smaller chunks** (e.g., 200 chars): More precise retrieval, but may cut off essential context mid-thought.
- **Larger chunks** (e.g., 800 chars): More context per chunk, but the embedding represents a broader topic, reducing precision.
- This project uses 400 chars, a common middle ground. The `RecursiveCharacterTextSplitter` in the LangChain module is smarter — it tries to split on paragraph → sentence → word boundaries before hard-cutting.

---

**Q: What is the max number of chunks this project supports, and why does that limit exist?**
300 chunks (`chunks[:300]` in `rebuild_indexes()`). This is a practical safeguard — all chunk embeddings are held in memory as a numpy array. Without a limit, uploading very large documents could exhaust RAM and slow cosine similarity computation significantly.

---

## LangChain

### Chains & Prompt Templates

**Q: What is a LangChain chain and how is it different from a direct LLM call?**
A chain is a composable pipeline of steps using the `|` pipe operator (LCEL — LangChain Expression Language). Each step's output becomes the next step's input. A direct LLM call is a one-shot request. Chains enable modular, reusable pipelines — e.g., `prompt | llm | parser` — where you can swap any component without rewriting the flow.

---

**Q: What does `StrOutputParser` do and why use it?**
It extracts the plain text string from the LLM's response object (an `AIMessage`). Without it, the chain returns the full `AIMessage` object. Using a parser makes the output directly usable as a string — important when piping output into the next step of a chain.

---

**Q: How does the chaining endpoint work?**
It runs a 3-step sequential chain on a piece of text:
1. Translate to English.
2. Summarize in one sentence.
3. Wrap the summary in JSON with a `summary` key.

Each step's output is piped directly as input to the next step. The final output is a JSON string.

---

### Memory

**Q: How is per-session conversation memory implemented here?**
`LC_SESSIONS` is a plain Python dict mapping `session_id → list of messages` (HumanMessage / AIMessage objects). On each request, the session's history is passed to a `ChatPromptTemplate` via `MessagesPlaceholder`, the LLM responds, and both the new human message and AI reply are appended to the list. This is in-memory only — sessions are lost on server restart.

---

**Q: What is `MessagesPlaceholder` in LangChain?**
It's a slot in a `ChatPromptTemplate` that gets filled at runtime with a list of message objects (conversation history). It allows you to inject a dynamic, growing conversation history into a fixed prompt template structure without rebuilding the template each turn.

---

**Q: What is the risk of using an in-memory dict for session state?**
- Lost on server restart.
- Unbounded growth — sessions are never expired automatically, only on explicit `DELETE /lc/memory/{session_id}`.
- Not shareable across multiple server processes/instances.
In production you'd use Redis, a database, or a managed session store.

---

### Tools & Agents

**Q: What is `bind_tools` in LangChain?**
It attaches tool definitions (schemas) to the LLM so the model knows what tools are available and can emit structured tool-call requests. The model doesn't execute the tools — your code must detect tool calls in the response and invoke the actual functions. `bind_tools` is a lower-level primitive; `create_react_agent` handles the tool-call / execute / loop automatically.

---

**Q: What tools are available in the LangChain agent endpoint and how are they defined?**
Three tools: `calculator`, `get_exchange_rate`, `get_country_info`. They are Python functions decorated with LangChain's `@tool` decorator, which auto-generates the JSON schema from the docstring and type annotations. The `create_react_agent` from `langgraph.prebuilt` wraps the LLM + tools in a ReAct loop automatically.

---

**Q: What is the ReAct pattern?**
ReAct (Reasoning + Acting) is an agent pattern where the LLM alternates between:
1. **Reason** — think about what to do next (often visible in the model's response).
2. **Act** — call a tool.
3. **Observe** — receive the tool result and reason again.

This loop continues until the model emits a final answer. It's implemented here via `langgraph.prebuilt.create_react_agent`.

---

**Q: What's the difference between the `tools` endpoint and the `agent` endpoint in the LangChain module?**
- `/lc/tools` uses `bind_tools` manually — it sends one request, extracts tool calls from the response, executes them in Python, then sends a second request with the tool results. It runs one round of tool use.
- `/lc/agent` uses `create_react_agent`, which handles the ReAct loop automatically — the agent can call tools multiple times in sequence until it decides it has enough information.

---

### Output Parsers

**Q: What output parsers does this project demonstrate?**
Three parsers from `langchain_core.output_parsers`:
- `StrOutputParser` — extracts the plain text string.
- `JsonOutputParser` — parses the LLM output as JSON into a Python dict.
- `CommaSeparatedListOutputParser` — splits the LLM output on commas, returning a Python list.

Each parser is chained as `prompt | llm | parser`.

---

**Q: What happens if `JsonOutputParser` receives malformed JSON from the LLM?**
It raises a parsing exception. In practice you'd add retry logic (LangChain's `OutputFixingParser` or `RetryWithErrorOutputParser`) or wrap the call in a try/except. Prompting the LLM to output only valid JSON with no surrounding text greatly reduces this failure mode.

---

### Document Splitters

**Q: What is the difference between `CharacterTextSplitter` and `RecursiveCharacterTextSplitter`?**
- `CharacterTextSplitter`: Splits purely on a single separator (default `"\n\n"`). If a chunk exceeds `chunk_size`, it hard-cuts it.
- `RecursiveCharacterTextSplitter`: Tries a list of separators in order (`"\n\n"` → `"\n"` → `" "` → `""`) and only falls back to the next separator if the current one can't produce small-enough chunks. This preserves natural language boundaries better.

`RecursiveCharacterTextSplitter` is almost always the better default for prose text.

---

### Multi-Agent & LangGraph

**Q: How does the multi-agent endpoint work?**
Two sequential LLM calls with distinct system prompts:
1. A **research assistant** LLM summarizes facts about the topic.
2. A **blog writer** LLM takes the researcher's output and writes a blog post.

There is no coordination framework — the output of the first `invoke()` is passed as input to the second. This is the simplest form of multi-agent composition.

---

**Q: What is LangGraph and how is it used in this project?**
LangGraph is a library for building stateful, multi-step LLM workflows as directed graphs. Nodes are Python functions, edges define the flow, and a shared `TypedDict` state object passes data between nodes. In this project it implements a blog-writing workflow:
- **Nodes**: `manager`, `research`, `writer`, `reviewer`
- **Conditional edge**: after the reviewer node, a function checks `state["revisions"] >= 2` — if true it routes to `END`, otherwise back to `writer` for another revision pass.

---

**Q: What is a `StateGraph` and what does `TypedDict` have to do with it?**
`StateGraph` takes a `TypedDict` class as its state schema. Every node receives the current state dict and returns a partial dict of updated keys. LangGraph merges these updates into the state automatically. Using `TypedDict` provides type hints so you know what keys exist (e.g., `topic`, `research`, `draft`, `feedback`, `revisions`).

---

**Q: What is a conditional edge in LangGraph?**
A conditional edge calls a routing function after a node completes. The function inspects the current state and returns the name of the next node (or `END`). In this project, after the reviewer node runs, the router checks the revision count — if it has hit the max (2), it ends the graph; otherwise it loops back to the writer node for another draft.

---

## General / System Design

**Q: Why do the RAG module and LangChain module use different Claude models?**
- RAG uses `claude-sonnet-4-6` — a more capable model, appropriate for re-ranking, query rewriting, and tool-calling agent loops where reasoning quality matters.
- LangChain uses `claude-haiku-4-5-20251001` — a faster, cheaper model appropriate for demos that prioritize responsiveness over depth.
This reflects a real-world pattern: use cheaper models for high-volume or low-stakes tasks, stronger models for complex reasoning.

---

**Q: How would you scale this app for production?**
Key changes:
1. Replace in-memory `DOCS`/`DOC_EMBS` with a persistent vector database (Pinecone, Weaviate, pgvector).
2. Replace `LC_SESSIONS` dict with Redis or a database-backed session store.
3. Remove wildcard CORS — restrict to known frontend origins.
4. Add authentication and per-user document isolation.
5. Move embedding computation off the request thread into a background worker.
6. Cap chunk count per user, not globally.

---

**Q: What are the limitations of TF-IDF compared to BM25?**
TF-IDF doesn't normalize for document length — a long chunk with many term occurrences can dominate unfairly. BM25 adds length normalization and a saturation factor so term frequency has diminishing returns. This project's `bsearch` uses `TfidfVectorizer` (sklearn) which approximates BM25 behavior but doesn't implement the full BM25 formula.

---

**Q: Why are LangChain imports deferred (inside route functions) in this project?**
Deferred imports prevent every LangChain dependency from loading at server startup. Since each endpoint uses different splitters, parsers, and agent configurations, loading them all upfront would slow cold start. Deferring also makes each route fully self-contained and easier to reason about in isolation.

---

**Q: What is FAISS and what kind of similarity search does it perform?**
FAISS (Facebook AI Similarity Search) is a library optimized for fast nearest-neighbor search over dense float vectors. In the LangChain RAG endpoint, it builds an index over 7 document embeddings and returns the k=2 most similar chunks to the query embedding using cosine (or inner product) similarity. For small corpora like this, it's essentially an exact search; FAISS's real advantage is approximate nearest-neighbor search at billion-vector scale.
