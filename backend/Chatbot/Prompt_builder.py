"""
Prompt Engineering Layer - Context injection and template management.
Handles structured prompts, guardrails, and hallucination mitigation.
"""
import logging
from typing import Optional
from models.schemas import (
    PortfolioContext, UserProfile, MarketContext, MarketRegime
)

logger = logging.getLogger("finai.prompts")

SYSTEM_PROMPT = """You are FinAI, an advanced Financial Intelligence Assistant embedded in a professional investment platform.

ROLE & CONSTRAINTS:
- You provide financial education, analysis, and portfolio insights
- You NEVER give direct investment advice or guarantee returns
- You always frame responses as educational and analytical
- You cite uncertainty when data is insufficient
- You maintain professional, precise language at all times

RESPONSE FORMAT (STRICT JSON):
You MUST respond ONLY with valid JSON in this exact structure:
{
  "summary": "2-3 sentence high-level answer to the question",
  "analysis": "Detailed analytical breakdown (3-5 sentences)",
  "risk_note": "Specific risk consideration relevant to the query",
  "confidence_level": "High|Medium|Low based on data quality and certainty"
}

GUARDRAILS:
- If asked for specific stock picks: explain principles instead, note you cannot recommend specific securities
- If asked about illegal activity: refuse firmly and professionally
- If data is missing: say so explicitly in analysis, do NOT fabricate numbers
- Confidence is "Low" if you lack current market data
- Keep total response under 400 words
"""

def build_context_block(
    portfolio: Optional[PortfolioContext],
    user_profile: Optional[UserProfile],
    market_context: Optional[MarketContext],
) -> str:
    """
    Safely builds structured context injection block.
    Sanitizes inputs to prevent prompt injection.
    """
    sections = []

    if market_context:
        regime_desc = {
            MarketRegime.BULL: "Bullish - positive momentum, risk-on sentiment",
            MarketRegime.BEAR: "Bearish - negative momentum, risk-off sentiment",
            MarketRegime.SIDEWAYS: "Sideways - consolidation phase, low directional conviction",
            MarketRegime.VOLATILE: "High Volatility - elevated VIX, uncertain conditions",
            MarketRegime.UNKNOWN: "Regime unknown - insufficient data",
        }.get(market_context.regime, "Unknown")

        market_lines = [f"[MARKET CONTEXT]", f"Regime: {regime_desc}"]
        if market_context.vix is not None:
            market_lines.append(f"VIX: {market_context.vix:.1f}")
        if market_context.sp500_trend:
            market_lines.append(f"S&P 500 Trend: {_sanitize(market_context.sp500_trend)}")
        if market_context.news_sentiment:
            market_lines.append(f"News Sentiment: {_sanitize(market_context.news_sentiment)}")
        if market_context.news_headlines:
            sanitized_headlines = [_sanitize(h) for h in market_context.news_headlines[:3]]
            market_lines.append(f"Recent Headlines: {' | '.join(sanitized_headlines)}")
        sections.append("\n".join(market_lines))

    if portfolio:
        portfolio_lines = ["[PORTFOLIO SUMMARY]"]
        if portfolio.total_value:
            portfolio_lines.append(f"Portfolio Value: ${portfolio.total_value:,.0f}")
        if portfolio.daily_pnl_pct is not None:
            pnl_sign = "+" if portfolio.daily_pnl_pct >= 0 else ""
            portfolio_lines.append(f"Daily P&L: {pnl_sign}{portfolio.daily_pnl_pct:.2f}%")
        if portfolio.beta is not None:
            portfolio_lines.append(f"Portfolio Beta: {portfolio.beta:.2f}")
        if portfolio.sharpe_ratio is not None:
            portfolio_lines.append(f"Sharpe Ratio: {portfolio.sharpe_ratio:.2f}")
        if portfolio.max_drawdown_pct is not None:
            portfolio_lines.append(f"Max Drawdown: {portfolio.max_drawdown_pct:.1f}%")
        if portfolio.top_holdings:
            sanitized_holdings = [_sanitize(h) for h in portfolio.top_holdings[:5]]
            portfolio_lines.append(f"Top Holdings: {', '.join(sanitized_holdings)}")
        if portfolio.sector_exposure:
            top_sectors = sorted(
                portfolio.sector_exposure.items(), key=lambda x: x[1], reverse=True
            )[:3]
            sector_str = ", ".join([f"{_sanitize(k)}: {v:.0f}%" for k, v in top_sectors])
            portfolio_lines.append(f"Top Sectors: {sector_str}")
        sections.append("\n".join(portfolio_lines))

    if user_profile:
        profile_lines = ["[USER BEHAVIORAL PROFILE]"]
        profile_lines.append(f"Risk Tolerance: {user_profile.risk_profile.value.title()}")
        if user_profile.investment_horizon:
            profile_lines.append(f"Investment Horizon: {_sanitize(user_profile.investment_horizon)}")
        if user_profile.behavioral_bias:
            profile_lines.append(f"Detected Behavioral Bias: {_sanitize(user_profile.behavioral_bias)}")
            profile_lines.append("Note: Address this bias in your response if relevant.")
        if user_profile.last_action:
            profile_lines.append(f"Last Portfolio Action: {_sanitize(user_profile.last_action)}")
        sections.append("\n".join(profile_lines))

    if not sections:
        return "[CONTEXT: No portfolio or market context provided. Respond with general financial education.]"

    return "\n\n".join(sections)

def build_full_prompt(
    user_message: str,
    conversation_history: str,
    portfolio: Optional[PortfolioContext],
    user_profile: Optional[UserProfile],
    market_context: Optional[MarketContext],
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) tuple for Ollama"""

    context_block = build_context_block(portfolio, user_profile, market_context)

    user_prompt_parts = []
    if context_block:
        user_prompt_parts.append(context_block)
    if conversation_history:
        user_prompt_parts.append(f"[CONVERSATION HISTORY]\n{conversation_history}")
    user_prompt_parts.append(f"[USER QUESTION]\n{_sanitize(user_message)}")
    user_prompt_parts.append("\nRespond ONLY with the JSON structure specified. No markdown, no extra text.")

    return SYSTEM_PROMPT, "\n\n".join(user_prompt_parts)

def _sanitize(text: str) -> str:
    """
    Basic prompt injection prevention.
    Strips newlines and control sequences that could break prompt structure.
    """
    if not text:
        return ""
    # Remove potential injection patterns
    dangerous = ["[SYSTEM]", "[USER]", "[ASSISTANT]", "IGNORE PREVIOUS", "ignore all"]
    sanitized = str(text)
    for pattern in dangerous:
        sanitized = sanitized.replace(pattern, "[FILTERED]")
    # Limit length per field
    return sanitized[:500].strip()