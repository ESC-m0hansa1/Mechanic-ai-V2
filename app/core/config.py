"""Typed, validated application configuration.

Every knob lives here and comes from the environment (.env locally, real env
vars in Docker/production). Type annotations are the validation: if PORT isn't
an int or a required key is missing, the app refuses to start with a clear
error instead of failing deep inside a request.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_file=".env" -> read that file. extra="ignore" -> tolerate unknown keys.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- app ---------------------------------------------------------------
    app_name: str                 # required: missing => startup error
    app_port: int = 8000
    log_level: str = "INFO"

    # --- database ----------------------------------------------------------
    database_url: str             # postgresql://user:pass@host:port/db

    # --- language model (OpenAI-compatible provider) ------------------------
    llm_api_key: str = ""         # empty is allowed so eval/tests can run offline
    llm_base_url: str = "https://api.mistral.ai/v1"
    llm_model: str = "mistral-small-latest"
    llm_max_tokens: int = 600
    llm_timeout_s: float = 60.0

    # --- retrieval ---------------------------------------------------------
    # Literal = pydantic rejects any value outside this set at startup, so a
    # typo in .env fails loudly instead of raising KeyError on first request.
    # Default is the strategy that measured best (eval/results/): reranked.
    retrieval_strategy: Literal["dense", "hybrid", "reranked"] = "reranked"
    embedding_model: str = "BAAI/bge-small-en-v1.5"     # 384-dim, 512-token window
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many hybrid candidates the cross-encoder rescores. Swept 5/6/8/10/12/16/20/30
    # on the golden set (eval/sweep.py) and the result was monotone in the
    # OPPOSITE direction to the usual advice: MRR fell from 0.856 at 8 to 0.829
    # at 30 while p50 latency rose 576 -> 1836 ms. A deeper pool hands a 6-layer
    # MiniLM more plausible-but-wrong chunks to promote, and it takes the bait.
    # 5-10 are statistically indistinguishable here (one question changing rank
    # moves MRR by ~0.02 on n=24), so 8 is chosen on other grounds: it is the
    # shallowest pool that still lets the reranker RECOVER a chunk hybrid ranked
    # 6th-8th, which is the entire point of two stages. Pool=5 scored marginally
    # higher but degenerates into pure reordering of hybrid's top 5.
    candidate_pool: int = 8
    rrf_k: int = 60               # RRF smoothing constant (see hybrid.py)
    # How deep each retriever nominates before RRF fuses the two lists. Measured
    # on the golden set: 10 beat 20 and 30 when hybrid is the FINAL ranker
    # (MRR 0.733 vs 0.719 vs 0.691) because a deep pool lets weak keyword
    # matches accumulate rank credit. It stays 10 under the reranked strategy
    # too: hybrid is asked for `candidate_pool` (8) fused results, and each
    # retriever nominating 10 is what lets a chunk both of them rank mid-list
    # beat one that only a single retriever loves.
    fusion_depth: int = 10


# One shared instance: .env is read once at import, not per request.
settings = Settings()
