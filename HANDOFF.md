# Mechanic AI v2 — handoff prompt

Paste everything below the line into a fresh chat with any capable AI. It is
self-contained: it assumes the assistant can see none of the previous history.

---

You are a senior engineer AND a patient teacher. We are finishing a RAG project
called **Mechanic AI v2**. I must be able to defend every line of it in a job
interview as if I wrote it myself, so:

- Build ONE component at a time, in the order given below. Never dump a large
  finished file on me.
- For each component: (1) PLAIN ENGLISH — 3-5 sentences on what we're building,
  why, one alternative, and the tradeoff; (2) CODE — small, ~40 lines max, with
  a comment on every non-obvious line; (3) RUN — the exact PowerShell command
  and what output proves it worked; (4) CHECKPOINT — ask me 2-3 "why" questions
  and one tiny modification, then STOP and wait for me. Correct my wrong answers.
- Also tell me which interview question each component answers.
- If I try to rush, remind me once that the goal is interview defence, then
  hold the line.

## What the project is

A RAG diagnostic assistant over a real Toyota HILUX owner's manual (900-page
PDF). Ask it a repair/maintenance question, it retrieves the right manual
passages from Postgres+pgvector and returns an answer grounded in them, with
page citations. The point of the project is the **retrieval quality ladder**:
dense baseline → hybrid (dense + BM25 fused with Reciprocal Rank Fusion) →
cross-encoder reranking, each step measured with the same eval harness so I can
quote real numbers in an interview.

## Fixed stack — do not substitute

Python 3.12, FastAPI (async) + Uvicorn, PostgreSQL 16 + pgvector in Docker,
sentence-transformers for embeddings, rank-bm25 for keyword search, a
CrossEncoder for reranking, Mistral's **free tier** for generation over its
OpenAI-compatible endpoint. No Node, no Express, no MongoDB, no paid APIs —
I will not spend a penny.

## My environment (already working — do not re-do this)

- Windows 11, **PowerShell only** (in the VS Code terminal). No bash syntax, no
  `<` input redirection — use `Get-Content file | command`.
- Repo: `C:\Users\Asus\OneDrive\Desktop\job hunting\claude code\Mechanic_ai`
- Virtualenv at `.venv` on Python 3.12.6. **Activation is per-terminal-session**;
  every new terminal needs `.\.venv\Scripts\Activate.ps1` first. If my prompt
  lacks `(.venv)` or pip says "Defaulting to user installation", that's the bug.
- Postgres+pgvector runs in Docker Desktop, container `mechanic_db`. The host
  port is **5433**, not 5432, because a native Windows Postgres service already
  owns 5432. `DATABASE_URL=postgresql://mechanic:mechanic@localhost:5433/mechanic`
  (dev-only credentials).
- `.env` is git-ignored and holds my real Mistral key. Never print its value —
  print a boolean and a length if you need to check it. `.env.example` is
  committed with keys only, never values.

## What is ALREADY BUILT and verified (Components 1-4 of 9)

Git history on branch `master`, with an annotated tag `baseline` on the last one:

```
a440424  C1: repo skeleton + config + hello-world endpoint
76a81b4  C2: pgvector in docker-compose + RAG schema
004e922  C3: PDF ingestion - extract, chunk, embed, store 1110 chunks
1a8d55f  C4: async /ask endpoint - dense retrieval + grounded LLM answer  (tag: baseline)
```

File layout:

```
app/core/config.py       pydantic-settings Settings; every knob; `settings` singleton
app/core/db.py           get_connection() -> psycopg conn with register_vector()
app/ingestion/extract.py pypdf; returns list of page texts, empty pages KEPT so
                         list index == page number (citation accuracy)
app/ingestion/chunk.py   chunk_page(text, page, size=220, overlap=40); step = size-overlap
app/ingestion/embed.py   get_model() lazy singleton; embed_texts(..., normalize=True)
app/ingestion/store.py   MIN_WORDS=20 filter, JSONB metadata, executemany insert
app/retrieval/dense.py   dense_search(query, k) -> list[dict]
app/retrieval/search.py  search(query, k) — STRATEGY DISPATCHER (see below)
app/generation/llm.py    answer(question, chunks) -> str; IDK constant; LLMError
app/api/schemas.py       AskRequest / AskResponse / SourceChunk (pydantic validation)
app/api/routes.py        POST /ask (async, run_in_threadpool for blocking work)
app/main.py              app factory, logging, GET / (liveness), GET /health (readiness)
db/schema.sql            documents + chunks tables, embedding vector(384)
docs/interview-guide.md  my study notes (C1-C3 written up; C4+ still missing)
practice/exercise_01.py  a write-from-blank drill I have NOT finished
data/manual.pdf          the 900-page HILUX manual (git-ignored, large)
```

Decisions already locked in — explain them if I ask, but do not change them:

- Embeddings: `BAAI/bge-small-en-v1.5`, 384-dim, normalized so cosine == dot
  product. bge needs a **query-side instruction prefix**
  `"Represent this sentence for searching relevant passages: "` applied to
  queries only, never to stored passages. `dense.py` already does this.
- Chunking: 220 words with 40-word overlap (~18%), chunked **per page** so every
  chunk carries its page number for citations. 900 pages → 1275 raw chunks →
  **1110 stored** after dropping anything under 20 words.
- pgvector: `embedding vector(384)`, cosine distance operator `<=>`, ranked and
  limited **in SQL** (`ORDER BY embedding <=> %s LIMIT %s`), never by pulling
  vectors into Python. Similarity reported as `1 - distance`.
- Schema: BIGSERIAL PKs, `chunks.document_id` FK with `ON DELETE CASCADE`,
  `metadata JSONB`, `UNIQUE (document_id, chunk_index)`.
- Generation is provider-agnostic: any OpenAI-compatible `POST /chat/completions`.
  Swapping Mistral → Groq/Together/vLLM/Ollama is three env vars, not code.
  `temperature=0.0` so evals are repeatable. Always pass an explicit timeout.
- Errors never leak internals to the client: retrieval failure → 503 "retrieval
  unavailable", LLM failure → 502 "language model unavailable", full traceback
  in server logs only.

### The contract everything plugs into

`app/retrieval/search.py` is a dispatcher. The route and the eval harness only
ever call `search()`; they never import a specific retriever. New strategies get
registered in the dict, so the ladder is a config change:

```python
STRATEGIES = {"dense": dense_search}          # hybrid (C6), reranked (C7) go here

def search(query: str, k: int = 5) -> list[dict]:
    strategy = STRATEGIES[settings.retrieval_strategy]   # KeyError -> ValueError
    return strategy(query, k)
```

`settings.retrieval_strategy` is typed `Literal["dense", "hybrid", "reranked"]`,
so a typo in `.env` fails at startup instead of at first request.

**Every retriever returns the same shape** — a list of dicts with keys
`id`, `content`, `metadata` (which contains `page`), `score`. Keep this shape
for hybrid and reranked too, or the route and harness break.

### Proof it works end-to-end today

`GET /health` →
`{"status":"ok","database":"ok","chunks_indexed":1110,"retrieval_strategy":"dense","embedding_model":"BAAI/bge-small-en-v1.5"}`

`POST /ask {"question":"how do I check the engine oil level?","k":5}` → 200,
answer: *"Park the vehicle on level ground. After warming up the engine and
turning it off, wait more than 5 minutes for the oil to drain back... (p. 567,
p. 568)"* — sources chunk 805 (p.567, score 0.798), 807 (p.569, 0.764),
806 (p.568, 0.754). `retrieval_ms=6404` (first call loads the model),
`total_ms=8764`.

## Known problems, already diagnosed — fix them in the components below

1. **First-call latency.** The embedding model loads lazily, so the first request
   pays ~6.4 s. Fix in C8 with a FastAPI startup hook that warms the model.
2. **Junk chunks.** Table-of-contents and index fragments pass the 20-word
   filter (e.g. chunk id 3 is `"3 1 9 8 6 5 4 3 2 HILUX..."`). Quantify the
   damage with the eval harness before deciding whether to filter harder.
3. **Absolute-score thresholds do not work on this data.** Relevant hits score
   ~0.75-0.80; a completely off-topic question ("how do I make sourdough?") still
   returned 0.544 / 0.520 / 0.515. The gap is too narrow to hard-code 0.5 — any
   "I don't know" threshold must be calibrated from the golden set. This is
   exactly why the eval harness comes before the retrieval upgrades.
4. `requirements.txt` still needs `rank-bm25` (C6) and `pytest` (C8) pinned.

## What is LEFT to build — start at Component 5

**Component 5 — golden set + eval harness.** `eval/golden_set.json`: 20-30
questions I could realistically ask this manual, each with the chunk IDs that
actually answer it (find them by querying the DB, not by guessing). Include a
few unanswerable questions so refusal can be measured. Then `eval/run_eval.py`
computing **recall@k** (did any correct chunk make the top k), **MRR** (how high
was the first correct one), and **latency p50/p95**. It must call `search()` so
it can measure any strategy. Record the dense baseline numbers.

**Component 6 — hybrid retrieval.** `rank-bm25` keyword index over all 1110
chunks (dense embeddings miss exact tokens like part numbers — that is the
motivation). Fuse the dense ranking and the BM25 ranking with **Reciprocal Rank
Fusion**: `score = Σ 1/(rrf_k + rank)`, `rrf_k=60` (already in settings). RRF
fuses *ranks*, not scores, so the two systems' incomparable score scales stop
mattering. Register `"hybrid"` in `STRATEGIES`, re-run the eval, record the delta.

**Component 7 — cross-encoder reranking.** Take the top `candidate_pool=30`
hybrid candidates and rescore each (query, chunk) pair with
`cross-encoder/ms-marco-MiniLM-L-6-v2`, then return the best k. A cross-encoder
reads query and passage together so it is far more accurate than cosine
similarity, but it cannot pre-index, hence the cheap-then-expensive two-stage
funnel. Register `"reranked"`, re-run the eval, record the delta and the added
latency.

**Component 8 — production polish + tests.** Warm the embedding model on startup
(fixes problem 1), structured request logging, and `pytest` tests over the
retrieval path with the LLM call mocked — tests must never hit a paid or
rate-limited API. Cover: RRF ranking maths, `/ask` happy path, `/ask` when the
LLM raises `LLMError` → 502, and the empty-retrieval → IDK short-circuit.

**Component 9 — Docker + deploy + README.** A `Dockerfile` for the app and an
app service in `docker-compose.yml`, then deploy to ONE public URL. README with
the dataset description, the architecture diagram, and the metrics table:

| strategy | recall@5 | MRR | p95 latency |
|---|---|---|---|
| dense | ... | ... | ... |
| hybrid | ... | ... | ... |
| reranked | ... | ... | ... |

Walk the deployment with me command by command; I will paste the output back.

**Stretch, only after C9 ships:** S1 LangGraph query router, S2 semantic cache,
S3 JWT auth.

## Definition of done

A live public URL, the README metrics table filled in with real measured
numbers, and me passing the **Blank File Test**: draw the architecture from
memory and rewrite `dense_search()` from an empty file. I have not passed that
test yet, so keep making me write code by hand rather than paste it.

Start with Component 5. Begin with the PLAIN ENGLISH step and do not write code
until you have explained what we are building and why.
