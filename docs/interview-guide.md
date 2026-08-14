# Mechanic AI v2 — Interview Study Guide

> **How to use this:** Read it top-to-bottom — it's the *story* of how the system
> was built, not a pile of facts. Each component section has the same shape:
> **What I built → Decisions & why → Concepts (and why they showed up here) →
> Interview questions it answers.**
>
> Memorize the **decisions** and the **architecture shape**. The code itself is
> disposable — you regenerate or look it up. The reasoning is what you defend.

---

## 🎤 Elevator pitch (grows as we build)

> Mechanic AI v2 is a Retrieval-Augmented Generation (RAG) assistant that answers
> repair questions from a real vehicle manual. It runs a fully local pipeline —
> local embeddings + local LLM — so no query or document ever leaves the host.
> *(We'll extend this with the hybrid-retrieval + reranking metrics story.)*

## 🗺️ Architecture (be able to draw this from memory)

```
INGESTION (offline, run once)
  data/manual.pdf ──pypdf──▶ 900 pages of text
        │
        ▼  chunk_page(): 220-word windows, 40-word overlap, per page
     1275 chunks ──filter <20 words──▶ 1110 chunks
        │
        ▼  bge-small-en-v1.5 (batched)
     1110 × 384-dim vectors
        │
        ▼
  Postgres: documents ─┬─< chunks[content, metadata{page,local_index}, embedding vector(384)]

SERVING (online, per request)
  Browser ──HTTP──▶ FastAPI (async) ──reads──▶ Settings (.env, typed & validated)
                        │
                        ▼
                   PostgreSQL + pgvector   (Docker container, data in a named volume)
```
*(Retrieval, LLM, hybrid search, and reranking get added to this diagram as we build them.)*

---

## Component 1 — Repo skeleton & configuration

**What I built:** an isolated Python 3.12 environment, a pinned dependency list, a
folder skeleton, git, and a running async FastAPI app whose config comes from a
validated settings object instead of hard-coded values.

**Decisions & why (this is the interview gold):**
- **`venv` + `requirements.txt` over Poetry/conda** — I wanted the most *transparent*
  setup: nothing hidden, one obvious file listing exactly what's installed. Poetry is
  more convenient (lockfiles, nicer CLI) but adds a tool and abstraction I'd have to
  explain. For a project I must defend line-by-line, transparency wins.
- **Pinned exact versions (`==`)** — makes the environment **reproducible**: my laptop,
  a teammate, CI, and the Docker build all get *identical* versions. Unpinned = "same
  command, different result on different days" = the classic works-on-my-machine bug.
- **`pydantic-settings` over scattered `os.getenv()`** — one typed, validated,
  self-documenting config object. If `APP_PORT` isn't a valid int, the app refuses to
  start with a clear error, instead of failing weirdly deep inside a request later.
- **Async FastAPI over Flask** — the spec's pipeline (DB + model calls) benefits from
  async concurrency, and FastAPI generates interactive API docs for free.

**Concepts (and why they appeared *here*):**
- **Virtualenv** — a project's private `site-packages`, so different projects can use
  different versions of the same package without colliding. *(Showed up when we isolated the env.)*
- **Transitive dependencies** — I asked for 2 packages, `pip list` showed ~20. The rest
  are dependencies-of-my-dependencies (FastAPI → starlette, pydantic; uvicorn → h11,
  websockets…). *(Showed up when reading `pip freeze`.)*
- **`.env` vs `.env.example`** — real values live in `.env` (git-ignored); a keys-only
  `.env.example` is committed as documentation. Committing a real `.env` leaks secrets
  into git history *permanently* — deleting the file later doesn't remove them; you must
  rotate the secret. *(Showed up when wiring config.)*
- **Required vs default settings** — a field with no default (`app_name: str`) is
  *required*: missing → startup crash. A field with a default (`app_port: int = 8000`)
  is optional. Loud early crash > silent bad value. *(Showed up testing the settings.)*
- **Config ≠ running server** — setting `APP_PORT=9000` just changes a *number*.
  Nothing listens until **uvicorn** binds the port. "Port already in use" is only
  possible at that bind moment. *(Showed up in the port checkpoint.)*
- **git tracks files, not empty folders** — empty `app/`/`tests/` were invisible until a
  file landed in them; force one with a `.gitkeep`. *(Showed up in `git status`.)*

**Interview questions this answers:**
- *"How is your project reproducible — how would someone stand it up from scratch?"*
- *"How do you manage configuration and secrets across environments?"*
- *"Why FastAPI / why async?"*

---

## Component 2 — Postgres + pgvector & the RAG schema

**What I built:** a PostgreSQL database (with the pgvector extension) running in Docker
via docker-compose, plus a version-controlled SQL schema modeling documents, their
chunks, chunk metadata, and each chunk's embedding vector.

**Decisions & why (interview gold):**
- **Postgres in Docker over a native install** — one command starts a Postgres that's
  *identical* on my laptop and the deploy server; no manual install drift, and the
  `pgvector/pgvector` image ships the vector extension prebuilt (no compiling it myself).
- **One database (pgvector) for BOTH rows and vectors, not a separate vector DB** — the
  data is relational (documents own chunks) *and* needs similarity search. pgvector lets
  one Postgres do both, so I avoid running/syncing a second system. A dedicated vector DB
  (Pinecone/Chroma) scales further, but that's complexity I don't need at this size.
- **Embedding as a column on `chunks`, not a separate `embeddings` table** — exactly one
  embedding per chunk from one model = a 1:1 relationship, so a separate table would only
  add joins for no benefit. I'd split it out only if I wanted *multiple* embedding models
  per chunk to compare — then the extra table earns its keep.
- **Schema-as-code (`db/schema.sql`)** — same principle as `requirements.txt`: the DB
  structure is reproducible and reviewable, not typed by hand each time.
- **Local `sentence-transformers` embeddings (384-dim) over Ollama `nomic-embed-text`** —
  sentence-transformers runs *in-process* (just a pip package); Ollama needs a separate
  model server to run and monitor. Fewer moving parts. *(Privacy doesn't separate them —
  both are local; simplicity does. Privacy is the argument for the local LLM choice.)*

**Concepts (and why they appeared *here*):**
- **Docker image vs container** — image = frozen template; container = a running instance
  of it. One image → many containers. *(Showed up starting Postgres.)*
- **Port mapping `"HOST:CONTAINER"`** — left = your machine's port, right = the port
  inside the container. Change the left if it's taken; the right stays fixed. *(Checkpoint.)*
- **Volumes = persistence** — a container's filesystem is disposable; data you want to
  survive restarts lives in a named volume. `docker compose down` keeps it; `down -v`
  deletes it. *(Checkpoint.)*
- **pgvector** — adds a `vector` column type + similarity operators to Postgres; that's
  what lets a *relational* DB do embedding search. *(Verified with `SELECT '[1,2,3]'::vector;`.)*
- **`BIGSERIAL PRIMARY KEY`** — auto-incrementing unique id identifying each row.
- **Foreign key + `ON DELETE CASCADE`** — a chunk *belongs to* a document; deleting the
  document auto-deletes its chunks, so there are never orphan chunks. *(Schema design.)*
- **`JSONB`** — a queryable JSON column for *variable* metadata (page, section) you don't
  want a rigid column for.
- **`vector(384)`** — the number is the embedding dimension and is a hard contract with
  the model: a 768-dim model would be rejected (`expected 384 dimensions, got 768`).
- **Vector *index* is a later, separate decision** (ivfflat vs hnsw), built *after* data
  exists.

**Interview questions this answers:**
- *"Why pgvector / why not a dedicated vector database?"*
- *"How did you model your data for RAG?"*
- *"Why store the embedding as a column vs a separate table?"*
- *"Why local embeddings — and where does privacy actually come into the design?"*

---

## Component 3 — Ingestion pipeline (PDF → chunks → embeddings → Postgres)

**What I built:** a pipeline that extracts text from a real 900-page Toyota HILUX owner's
manual, splits it into overlapping ~220-word chunks tagged with page numbers, embeds each
chunk into a 384-dim vector locally, and stores all 1110 chunks with metadata in Postgres —
logging the chunk-size distribution so the chunking choice is *measured*, not asserted.

**The numbers (memorize these — interviewers love concrete figures):**
900 pages → 1275 raw chunks → **1110 chunks** after filtering (min 20, median 142, max 220 words)
→ 1110 × 384-dim vectors. Chunk size **220 words**, overlap **40 words (~18%)**.

**Decisions & why (interview gold):**
- **Chunk size 220 words / overlap 40** — the embedder (bge-small) has a **512-token** window,
  and words ≈ tokens × 1.3, so ~220 words ≈ ~290 tokens: comfortably under the limit with
  headroom, but big enough to hold a complete instruction. Overlap ≈18% so an instruction
  split across a boundary survives intact in at least one chunk.
- **Chunk per page, not across pages** — every chunk then belongs to exactly one page, so the
  answer can cite *"see page 42."* The cost: an instruction spanning a page break gets cut.
  Deliberate trade: accurate citation > perfect continuity, for a manual.
- **`bge-small-en-v1.5` over `all-MiniLM-L6-v2`** — both 384-dim (fits my schema), but bge is
  retrieval-tuned and has a 512-token window vs MiniLM's 256. MiniLM is a bit faster; bge's
  better retrieval quality is worth it, and the bigger window is what permits 220-word chunks.
- **Filter chunks under 20 words** — the raw distribution had a `min` of **1 word** (page
  numbers, index fragments, diagram labels). A 1-word chunk is retrieval noise forever, and
  it's cheaper to exclude at ingest than to work around at query time. *Known risk:* a short
  line can be a real safety warning or a spec value, so the threshold is a tunable I'd revisit
  against eval numbers, not a law.
- **One batched embedding call, not per-chunk** — model invocation overhead dominates on CPU;
  batching 1110 chunks is dramatically faster than 1110 separate calls.
- **Keep empty pages in the page list** — so a page's *index* equals its *page number*. Drop
  them and every later page's cited number silently shifts by one.

**Concepts (and why they appeared *here*):**
- **Why chunk at all** — embedders/LLMs have max input lengths, and retrieval works best when
  each unit is ~one idea. Chunking is the granularity dial for retrieval quality.
- **Chunk-size trade-off, stated as two distinct failures.** Too big: (1) **truncation** — text
  past the token window is silently *not embedded*, so it's invisible to search; (2) **diluted
  matches** — a multi-topic chunk matches many queries weakly instead of one strongly. Too
  small: context is lost (a step separated from its warning).
- **Embeddings are contextual, then pooled** — the model reads the whole chunk at once, so a
  word's vector depends on its neighbours ("engine oil" vs "cooking oil"); per-token vectors
  are then pooled into one fixed-size 384-dim vector. *(Note: transformers do **not** drop
  stopwords — that's a BM25/keyword-search idea, coming in Component 6.)*
- **384 dimensions, not 3** — the count of numbers in one vector *is* the dimensionality of the
  space. Can't visualize it; the math (dot product, distance) is identical regardless. Don't
  confuse the *array shape* (1110 × 384, a container) with the *vector space* (384-D).
- **Normalized embeddings** — unit length, so **cosine similarity = dot product** (cheaper), and
  it lines up with pgvector's distance operators. Demonstrated live: oil-question ↔ oil-answer
  scored **0.836** while oil-question ↔ audio-sentence scored **0.395**, despite the Q and A
  sharing almost no words — that gap *is* semantic search, and keyword search would score ~0.
- **Dense embeddings capture semantics but can miss exact tokens** (part numbers, error codes) —
  precisely the gap BM25 fills in Component 6.
- **Lazy singleton model load** — loading a model is expensive: load once, reuse. Not at import
  (slows startup), not per call (catastrophic).
- **`RETURNING id`** — Postgres hands back the new primary key in the same statement, no second
  query. **`executemany`** — many rows in one driver call instead of N round-trips.
- **PDF text isn't guaranteed** — scanned PDFs are images and need OCR. Always eyeball the
  extraction before trusting it.
- **`x or ""`** — Python's fallback idiom; `page.extract_text()` can return `None`, and the
  `or ""` stops that `None` from crashing the next string operation.
- **`python -m package.module`** runs a file as part of its package so imports resolve.

**Interview questions this answers:**
- *"Why did you choose that chunk size and overlap?"* ← the one you're guaranteed to be asked
- *"Which embedding model did you use and why?"*
- *"How do you handle citations / knowing which page an answer came from?"*
- *"How did you validate that your ingestion was any good?"* (the distribution log)

---

## 🧰 Debugging war stories (keep these — "tell me about something that broke")

**1. Port collision: Docker Postgres vs a native Postgres service.**
`psycopg` failed with `FATAL: password authentication failed for user "mechanic"`. Key insight:
that message means a **real Postgres answered and rejected me** — so it wasn't a code bug.
`netstat -ano | findstr :5432` showed **two different PIDs** listening on 5432; `tasklist` showed
a native Windows `postgres.exe` service. My connection was landing on the wrong server. Fix:
remap the **host** side of the port pair (`"5433:5432"`) — the container's internal port never
changes — update `DATABASE_URL`, and `docker compose up -d` to recreate. The named volume (and
my schema) survived because I didn't pass `-v`.

**2. Reading which layer an error names.**
- `password authentication failed` → reached a server, got rejected → creds / wrong server.
- `Socket is not connected` / `connection refused` → **nothing listening** → service down or wrong port.
Two very different fixes; the wording tells you which.

**3. Virtualenv not activated → installs went to the wrong Python.**
After reopening the terminal, `import psycopg` failed. Three tells: the prompt had lost its
`(.venv)` prefix; pip said *"Defaulting to user installation"*; and the traceback path read
`...\Python314\site-packages\...` — global 3.14, not my 3.12 venv. **Activation is per-terminal
session** and must be redone every time. Secondary symptom: I'd installed bare `psycopg` instead
of `psycopg[binary]`, so the bundled `libpq` was missing → `no pq wrapper available`.
