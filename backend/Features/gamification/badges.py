import json
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

logger = structlog.get_logger()

BADGE_DEFINITIONS = {
    "first_blood": {
        "name": "First Blood",
        "description": "Completed your first trade",
        "icon": "🩸",
    },
    "discipline_master": {
        "name": "Discipline Master",
        "description": "30-day streak with no overtrading flag",
        "icon": "🧘",
    },
    "bias_buster": {
        "name": "Bias Buster",
        "description": "Zero bias flags for 60 consecutive trading days",
        "icon": "🧠",
    },
    "volatility_hunter": {
        "name": "Volatility Hunter",
        "description": "5 profitable trades during High-Volatility regime",
        "icon": "⚡",
    },
    "diversifier": {
        "name": "Diversifier",
        "description": "Portfolio DI > 0.8 maintained for 30 days",
        "icon": "🌐",
    },
    "sharpe_sniper": {
        "name": "Sharpe Sniper",
        "description": "Sharpe ratio > 2.0 over any rolling 60-day window",
        "icon": "🎯",
    },
    "news_ninja": {
        "name": "News Ninja",
        "description": "10 trades entered within 15 min of high-impact news",
        "icon": "📰",
    },
}


async def check_and_award_badges(user_id: str, db: AsyncSession) -> list[str]:
    """Evaluate badge conditions and award any newly earned badges."""
    row = await db.execute(
        text("SELECT badges FROM user_gamification WHERE user_id = :uid"),
        {"uid": user_id},
    )
    result = row.fetchone()
    existing_badges = {b["badge_id"] for b in (result.badges or [])} if result else set()
    newly_awarded = []

    # First Blood
    if "first_blood" not in existing_badges:
        count_row = await db.execute(
            text("SELECT COUNT(*) FROM trades WHERE user_id = :uid AND status = 'closed'"),
            {"uid": user_id},
        )
        if count_row.scalar() >= 1:
            await _award_badge(user_id, "first_blood", db)
            newly_awarded.append("first_blood")

    # Volatility Hunter — 5 profitable trades during high_volatility regime
    if "volatility_hunter" not in existing_badges:
        hvol_row = await db.execute(
            text("""
                SELECT COUNT(*) FROM trades t
                JOIN regime_states rs ON rs.asset_id = t.asset_id
                  AND rs.time >= t.entry_time
                  AND rs.time < t.exit_time
                WHERE t.user_id = :uid AND t.status = 'closed'
                  AND t.pnl > 0 AND rs.regime = 'high_volatility'
            """),
            {"uid": user_id},
        )
        if hvol_row.scalar() >= 5:
            await _award_badge(user_id, "volatility_hunter", db)
            newly_awarded.append("volatility_hunter")

    # Sharpe Sniper — rolling 60-day Sharpe > 2.0
    if "sharpe_sniper" not in existing_badges:
        sharpe_row = await db.execute(
            text("""
                SELECT MAX(sharpe) FROM portfolio_snapshots
                WHERE user_id = :uid AND window_days = 60
            """),
            {"uid": user_id},
        )
        max_sharpe = sharpe_row.scalar()
        if max_sharpe and max_sharpe >= 2.0:
            await _award_badge(user_id, "sharpe_sniper", db)
            newly_awarded.append("sharpe_sniper")

    # Discipline Master — streak >= 30 and no overtrading flag
    if "discipline_master" not in existing_badges:
        dm_row = await db.execute(
            text("""
                SELECT ug.streak_days, bp.bias_flags
                FROM user_gamification ug
                LEFT JOIN behavioral_profiles bp ON bp.user_id = ug.user_id
                WHERE ug.user_id = :uid
            """),
            {"uid": user_id},
        )
        dm = dm_row.fetchone()
        if dm and dm.streak_days >= 30:
            flags = dm.bias_flags or {}
            if not flags.get("overconfidence", False):
                await _award_badge(user_id, "discipline_master", db)
                newly_awarded.append("discipline_master")

    if newly_awarded:
        await db.commit()
        logger.info("badges_awarded", user_id=user_id, badges=newly_awarded)

    return newly_awarded


async def _award_badge(user_id: str, badge_id: str, db: AsyncSession):
    await db.execute(
        text("""
            UPDATE user_gamification
            SET badges = badges || :new_badge::jsonb
            WHERE user_id = :uid
        """),
        {
            "uid": user_id,
            "new_badge": json.dumps([{
                "badge_id": badge_id,
                "name": BADGE_DEFINITIONS[badge_id]["name"],
                "earned_at": datetime.utcnow().isoformat(),
            }]),
        },
    )
