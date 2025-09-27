from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    persist_dir: str = os.getenv("BABYAGI_PERSIST_DIR", ".babyagi_memory")
    max_steps: int = int(os.getenv("BABYAGI_MAX_STEPS", "6"))
    top_k: int = int(os.getenv("BABYAGI_TOP_K", "5"))
    temperature: float = float(os.getenv("BABYAGI_TEMPERATURE", "0.2"))

def load_settings() -> Settings:
    api = os.getenv("OPENAI_API_KEY", "")
    if not api:
        raise RuntimeError("OPENAI_API_KEY 가 .env 또는 환경변수에 설정되어야 합니다.")
    return Settings(openai_api_key=api)
