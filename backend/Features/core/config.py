from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://aifip:aifip_secret@localhost:5432/aifip"
    REDIS_URL: str = "redis://localhost:6379"
    SECRET_KEY: str = "change_me_in_production"
    HUGGINGFACE_MODEL: str = "ProsusAI/finbert"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    SPACY_MODEL: str = "en_core_web_trf"

    # Risk-free rate (will be updated from FRED)
    RISK_FREE_RATE: float = 0.052

    # XP config
    XP_DAILY_CAP: int = 1000
    LEVEL_BASE_XP: int = 500
    LEVEL_EXPONENT: float = 1.6

    # Regime model
    HMM_N_COMPONENTS: int = 5
    HMM_LOOKBACK: int = 60

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
