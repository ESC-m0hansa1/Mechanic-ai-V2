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
    retrieval_strategy: Literal["dense", "hybrid", "reranked"] = "dense"
    embedding_model: str = "BAAI/bge-small-en-v1.5"     # 384-dim, 512-token window
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many candidates the cheap stage passes to the expensive stage.
    candidate_pool: int = 30
    rrf_k: int = 60               # RRF smoothing constant (see hybrid.py)


# One shared instance: .env is read once at import, not per request.
settings = Settings()
