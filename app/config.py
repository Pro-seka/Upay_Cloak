"""Central settings (env-driven). Part 1 owns this file; Part 2 only READS `get_settings()`.

Copy `.env.example` to `.env`.  Every setting has a safe default, so the app boots with zero config
(LLM disabled -> template narratives).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="UPAY_", extra="ignore",
                                      protected_namespaces=())

    # --- artifacts (produced by Person A's pipeline)
    data_path: Path = ROOT / "data" / "transactions.csv"
    model_path: Path = ROOT / "models" / "risk_engine.joblib"
    reports_dir: Path = ROOT / "reports"
    cache_dir: Path = ROOT / "data" / "cache"
    db_path: Path = ROOT / "data" / "upayshield.db"
    auto_train: bool = False                  # run `python -m ml.train` if artifacts are missing

    # --- http
    cors_origins: list[str] = ["*"]
    serve_frontend: bool = False              # mount repo-root static frontend at "/"

    # --- stream simulator
    stream_default_interval_ms: int = 3500
    stream_demo_alert_every: int = 6

    # --- LLM (Part 2 reads these; never crash when unset)
    llm_provider: Literal["none", "anthropic", "openai"] = "none"
    llm_api_key: str | None = Field(
        None, validation_alias=AliasChoices("UPAY_LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"))
    llm_model: str | None = None              # provider default is chosen in code when None
    llm_timeout_s: float = 8.0
    llm_max_retries: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
