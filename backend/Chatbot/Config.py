"""Core configuration for FinAI Platform"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"  # or llama3
    OLLAMA_TEMPERATURE: float = 0.3
    OLLAMA_MAX_TOKENS: int = 1024
    OLLAMA_TIMEOUT: int = 60

    # Session memory
    MAX_HISTORY_TURNS: int = 8
    SESSION_TTL_SECONDS: int = 3600

    # Safety
    FINANCIAL_DISCLAIMER: str = (
        "⚠️ This is for educational purposes only and does not constitute financial advice. "
        "Consult a licensed financial advisor before making investment decisions."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()