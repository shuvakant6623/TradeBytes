import math
from dataclasses import dataclass
from typing import Optional
from core.config import get_settings

settings = get_settings()

REGIME_MULTIPLIERS = {
    "trending": 1.2,
    "high_volatility": 1.5,
    "mean_reverting": 1.1,
    "low_volatility": 0.9,
    "news_driven": 1.3,
    "unknown": 1.0,
}

BASE_XP = {
    "trade_complete": 25,
    "trade_profitable": 50,
    "no_bias_trade": 30,
    "daily_login": 10,
    "portfolio_rebalance": 40,
    "education_module": 75,
    "streak_5day_bonus": 100,
    "leaderboard_top10_weekly": 200,
}

FEATURE_UNLOCKS = {
    5:  ["advanced_correlation_matrix"],
    8:  ["regime_aware_risk_alerts"],
    10: ["behavioral_bias_deep_report"],
    12: ["semantic_news_search"],
    15: ["custom_alert_thresholds"],
    20: ["portfolio_stress_testing"],
}


@dataclass
class XPAwardResult:
    base_xp: int
    quality_multiplier: float
    streak_multiplier: float
    regime_multiplier: float
    final_xp: int
    multipliers: dict
    new_total_xp: int
    new_level: int
    level_up: bool
    new_unlocks: list[str]


def compute_level(total_xp: int) -> int:
    """
    XP_required(level) = base_xp × level^exponent
    Inverted: level = (total_xp / base_xp)^(1/exponent)
    """
    if total_xp <= 0:
        return 1
    level = int((total_xp / settings.LEVEL_BASE_XP) ** (1 / settings.LEVEL_EXPONENT))
    return max(1, level)


def xp_for_level(level: int) -> int:
    """XP threshold to reach a given level."""
    return int(settings.LEVEL_BASE_XP * (level ** settings.LEVEL_EXPONENT))


def compute_quality_multiplier(pnl_z_score: float = 0.0, sharpe_delta: float = 0.0) -> float:
    """
    quality = clamp(pnl_z_score × 0.5 + sharpe_delta × 0.5, 0.5, 3.0)
    Ensures bad trades earn less XP, great risk-adjusted trades earn more.
    """
    raw = pnl_z_score * 0.5 + sharpe_delta * 0.5
    return float(max(0.5, min(3.0, 1.0 + raw * 0.3)))


def compute_streak_multiplier(streak_days: int) -> float:
    """1 + min(streak_days, 30) × 0.02 → max +60% at day 30."""
    return float(1.0 + min(streak_days, 30) * 0.02)


def compute_xp_award(
    action_type: str,
    streak_days: int = 0,
    current_regime: str = "unknown",
    pnl_z_score: float = 0.0,
    sharpe_delta: float = 0.0,
) -> tuple[int, dict]:
    """
    Compute final XP for an action. Returns (final_xp, multipliers_dict).
    """
    base = BASE_XP.get(action_type, 10)
    quality_m = compute_quality_multiplier(pnl_z_score, sharpe_delta)
    streak_m = compute_streak_multiplier(streak_days)
    regime_m = REGIME_MULTIPLIERS.get(current_regime, 1.0)

    final_xp = int(base * quality_m * streak_m * regime_m)

    multipliers = {
        "quality": round(quality_m, 3),
        "streak": round(streak_m, 3),
        "regime": round(regime_m, 3),
    }
    return final_xp, multipliers


def get_new_unlocks(old_level: int, new_level: int) -> list[str]:
    """Return list of features unlocked when progressing from old_level to new_level."""
    unlocks = []
    for level_threshold, features in FEATURE_UNLOCKS.items():
        if old_level < level_threshold <= new_level:
            unlocks.extend(features)
    return unlocks


def check_anti_cheat(
    user_xp_today: int,
    action_type: str,
    is_wash_trade: bool = False,
) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    - Daily XP cap: 1000 XP/day
    - Wash trades: 0 XP
    """
    if is_wash_trade:
        return False, "wash_trade_detected"

    if user_xp_today >= settings.XP_DAILY_CAP:
        return False, "daily_cap_reached"

    return True, "ok"
