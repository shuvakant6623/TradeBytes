"""Data models for FinAI chatbot"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class PortfolioContext(BaseModel):
    total_value: Optional[float] = None
    daily_pnl_pct: Optional[float] = None
    top_holdings: Optional[List[str]] = []
    sector_exposure: Optional[dict] = {}
    beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None

class UserProfile(BaseModel):
    risk_profile: RiskProfile = RiskProfile.MODERATE
    investment_horizon: Optional[str] = "medium-term"
    behavioral_bias: Optional[str] = None  # e.g. "loss aversion", "overconfidence"
    last_action: Optional[str] = None

class MarketContext(BaseModel):
    regime: MarketRegime = MarketRegime.UNKNOWN
    vix: Optional[float] = None
    sp500_trend: Optional[str] = None
    news_sentiment: Optional[str] = None  # "positive", "negative", "neutral"
    news_headlines: Optional[List[str]] = []

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=2000)
    portfolio: Optional[PortfolioContext] = None
    user_profile: Optional[UserProfile] = None
    market_context: Optional[MarketContext] = None

class StructuredResponse(BaseModel):
    summary: str
    analysis: str
    risk_note: str
    confidence_level: str  # "High", "Medium", "Low"
    disclaimer: str
    raw_llm_response: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: StructuredResponse
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None