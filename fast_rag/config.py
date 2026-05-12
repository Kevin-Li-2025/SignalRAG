from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "Fast Search RAG"
    cache_path: Path = Path(os.getenv("FAST_RAG_CACHE", ".cache/fast_rag.sqlite3"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto").lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    brave_api_key: str | None = os.getenv("BRAVE_API_KEY")
    user_agent: str = os.getenv(
        "FAST_RAG_USER_AGENT",
        "FastSearchRAG/0.1 (+https://localhost)",
    )


settings = Settings()
