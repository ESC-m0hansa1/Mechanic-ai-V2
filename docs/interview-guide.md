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
(so far)
Browser / client
      │  HTTP
      ▼
FastAPI (async)  ──reads──▶ Settings (.env, typed & validated)
      │
      ▼
PostgreSQL + pgvector  (documents ─┬─< chunks[content, metadata, embedding vector(384)])
   (running in Docker, data in a named volume)
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
