# Mechanic AI

A retrieval-augmented diagnostic assistant over a **real** vehicle manual — the
762-page Toyota HILUX owner's manual (India). Ask it a repair question; it
answers only from the manual, cites the page for every claim, and refuses when
the manual does not cover the question.

**Live:** _(deploy pending)_ · **API docs:** `/docs` · **Stack:** FastAPI ·
PostgreSQL + pgvector · sentence-transformers · React

The interesting part of this project is not that RAG works. It is that the
evaluation harness twice contradicted the standard advice, and the code follows
the measurements instead of the advice.

---

## Retrieval results

28 hand-labelled questions (24 answerable, 4 unanswerable), labels assigned by
reading actual chunk text. `k=5`. Reproduce with `python -m eval.run_eval --all`.

| strategy | hit@1 | hit@3 | hit@5 | recall@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| dense (baseline) | 0.542 | 1.000 | 1.000 | 0.896 | 0.757 | 84 | 100 |
| hybrid (dense + BM25, RRF) | 0.583 | 0.875 | 1.000 | 0.896 | 0.733 | 143 | 160 |
| **reranked (hybrid → cross-encoder)** | **0.750** | 0.958 | 1.000 | 0.896 | **0.856** | 609 | 723 |

Reranking lifts hit@1 by 21 points and MRR by 10, and costs 6× the latency.
On the four unanswerable questions it also separates answerable from
unanswerable perfectly (refusal accuracy 1.00, vs 0.96 dense), so the "I don't
know" path is not guesswork.

### MRR broken down by question type

The golden set tags each question, because an aggregate number cannot test the
*hypothesis* that motivated adding BM25.

| strategy | exact-term (n=9) | neutral (n=9) | paraphrase (n=6) |
|---|---|---|---|
| dense | 0.889 | 0.778 | 0.528 |
| hybrid | 0.944 | 0.713 | **0.444** |
| reranked | **1.000** | **0.833** | **0.672** |

- `exact-term` questions use the manual's own jargon — `ISOFIX`, `AdBlue`, `1GD-FTV`.
- `paraphrase` questions deliberately avoid it — *"my truck won't turn over"*
  against a manual that says *"if the engine will not start"*.

BM25 did exactly what it was added to do (+0.055 on jargon) and exactly what it
was feared to do (−0.084 on paraphrases). That is why hybrid is **not** the
final ranker — it is the candidate generator.

---

## Two findings that contradicted the received wisdom

**1. Hybrid retrieval made things worse, and the fix was not more tuning.**
Adding BM25 dropped aggregate MRR from 0.757 to 0.733. The per-probe breakdown
showed why: BM25 matched the manual's *table of contents* as readily as its real
sections, because every rare term in a manual also appears in its own index.
Dotted-leader chunks (`Side turn signal lights . . . . . P. 304`) are lexically
perfect and answer nothing. `app/ingestion/quality.py` flags them by dot density
— 89 of 1110 chunks, and **zero** golden-set answers, verified before shipping.
They are flagged in metadata rather than deleted, because chunk ids are the
primary keys the golden set is labelled against; re-ingesting would renumber
everything and silently invalidate 24 hand-written labels.

**2. Giving the cross-encoder more candidates made it worse, not better.**
The standard recipe is "retrieve 50-100, rerank down to 5". Swept on this corpus
(`eval/sweep.py`), quality fell monotonically as the pool grew:

| candidate_pool | 5 | 8 | 12 | 16 | 20 | 30 |
|---|---|---|---|---|---|---|
| MRR | 0.885 | 0.856 | 0.843 | 0.835 | 0.835 | 0.829 |
| p50 ms | 428 | 576 | 809 | 1027 | 1282 | 1836 |

A deeper pool hands a 6-layer MiniLM more plausible-but-wrong chunks to promote,
and it takes the bait. Shipped at 8 — not the top scorer, because 5–10 are
statistically indistinguishable at n=24 and pool=5 degenerates into pure
reordering, losing the ability to recover anything hybrid ranked below the cut.

---

## Architecture

```
                    ┌─────────────────────────── one container ──────────────────────────┐
   browser ───────► │  FastAPI                                                            │
                    │    /            → React SPA (built by Vite, served as static files) │
                    │    /api/ask     → the pipeline below                                │
                    │    /api/health  → readiness + active configuration                  │
                    └───────────────────────────────┬───────────────────────────────────┘
                                                    │
   ingestion (offline, once)                        │  per request
   ─────────────────────────                        ▼
   manual.pdf                          ┌────────────────────────────┐
     │ pypdf                           │ 1. embed the question      │  bge-small-en-v1.5
     ▼                                 ├────────────────────────────┤
   page text                           │ 2a. dense: pgvector <=>    │──┐
     │ ~800 char chunks, page-tagged    │ 2b. BM25 over the corpus   │──┤ top 10 each
     ▼                                 ├────────────────────────────┤  │
   1110 chunks ──► bge-small ──► 384-d  │ 3. Reciprocal Rank Fusion  │◄─┘ Σ 1/(60+rank)
     │                            │     ├────────────────────────────┤
     │ dot-density noise filter   │     │ 4. cross-encoder rescores  │  ms-marco-MiniLM-L-6
     ▼                            ▼     │    the 8 survivors         │
   89 flagged, 1021 retrievable ─► Postgres ──────────┬─────────────┘
                                   + pgvector          ▼
                                              ┌────────────────────────────┐
                                              │ 5. prompt with excerpts    │  Mistral
                                              │    "cite the page, or say  │  (OpenAI-compatible
                                              │     you don't know"        │   endpoint)
                                              └────────────────────────────┘
                                                            ▼
                                              answer + page citations + scores
```

Retrieval strategy is a single config value (`RETRIEVAL_STRATEGY`), dispatched
through one `search()` function. So `dense` → `hybrid` → `reranked` is a config
change, not a code change, and the eval harness measures all three through the
same interface. A typo fails at startup rather than at first request, because the
field is a `Literal`.

---

## Run it

Needs Docker, and a free API key from [console.mistral.ai](https://console.mistral.ai).

```bash
cp .env.example .env    # then paste your key into LLM_API_KEY
```

```bash
docker compose up --build -d
```

First build takes a few minutes: it installs CPU-only PyTorch and bakes both
model checkpoints into the image so startup needs no network. Then ingest the
manual once:

```bash
docker compose exec app python -m app.ingestion.store
```

```bash
docker compose exec app python -m app.ingestion.flag_noise
```

Open **http://localhost:8000**.

<details>
<summary>Running without Docker (for development)</summary>

```bash
python -m venv .venv && .venv/Scripts/activate    # POSIX: source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d db
python -m app.ingestion.store && python -m app.ingestion.flag_noise
uvicorn app.main:app --reload
```

The frontend can run against it with hot reload; Vite proxies `/api` to :8000, so
there is no CORS involved:

```bash
cd frontend && npm install && npm run dev
```
</details>

---

## Evaluation

```bash
python -m eval.run_eval --all --k 5 --breakdown
```

```bash
python -m eval.sweep --strategy reranked --setting candidate_pool --values 5 8 16 30
```

Retrieval-only, so a full run is free and takes seconds — an eval you avoid
running because it costs money stops being run at all, and then the table above
rots. Metrics (`hit@k`, `recall@k`, MRR, nearest-rank percentiles) are
hand-written in `eval/metrics.py` and unit-tested.

The refusal threshold is **calibrated, not guessed**: `calibrate_threshold()`
sweeps observed top-1 scores for the split that best separates answerable from
unanswerable. A hand-picked 0.5 was useless here — an off-topic question still
scored 0.54 while a good hit scored 0.80 — and the three strategies report scores
on three different scales anyway (cosine, RRF sum, raw logit).

```bash
python -m pytest        # 38 tests, 0.23s
```

No test needs Postgres, torch, or an API key: the pure functions (fusion maths,
metrics, the noise heuristic, rerank ordering) are tested directly, and the
routes are tested with retrieval and generation stubbed at the seam. Failure
modes are asserted for leakage — a provider error becomes a generic 502 and the
test greps the response body to prove the API key never appears in it.

---

## Honest limitations

- **The golden set is 24 answerable questions, hand-labelled by me.** Differences
  under ~0.03 MRR are inside the noise of a set that size.
- **`rrf_k`, `fusion_depth` and `candidate_pool` were tuned on that same set.**
  That is tuning on the test set. The defence is that each is a single integer
  with a monotone effect rather than a free parameter vector — but it is still
  worth saying out loud instead of hiding.
- **hit@5 is saturated at 1.000**, which is why hit@1 and hit@3 are reported. A
  ceiling metric cannot show whether an upgrade helped.
- **Latency is CPU-only** on a laptop, measured natively. The 609 ms p50 for
  reranking is ~72 ms per query-chunk pair; a GPU or an ONNX export would change
  that picture entirely. The same image under Docker Desktop on Windows measured
  0.7–2.8 s for the identical work — the cross-encoder is the only CPU-bound step
  in the request, so it absorbs the whole variance of whatever CPU share the VM
  is given.
- **One worker, both models in memory.** Scaling means more instances, not more
  workers per instance — each worker would load its own copy of the weights.
- **BM25 is an in-process index** rebuilt at startup. Correct for 1110 chunks;
  at millions the keyword side belongs in Postgres as a `tsvector` + GIN index,
  next to the vectors.
- **Free-tier LLM.** Fine for a public Toyota manual. Free tiers generally license
  the provider to train on submitted data, which would disqualify it for
  proprietary content.

---

## Layout

```
app/
  api/          routes + pydantic request/response schemas
  core/         config, db, warm-up, request-id logging
  ingestion/    extract → chunk → embed → store, plus the noise filter
  retrieval/    dense · bm25 · hybrid (RRF) · rerank, behind one dispatcher
  generation/   provider-agnostic OpenAI-compatible LLM client
eval/           golden set, hand-written metrics, harness, setting sweeps
frontend/       React + Vite SPA (build-time Node only)
tests/          38 tests, no infrastructure required
```
