import uuid
import json
from datetime import datetime
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional

from core.database import get_db
from core.redis_client import RedisCache
from gamification.xp_engine import (
    compute_xp_award, compute_level, xp_for_level,
    get_new_unlocks, check_anti_cheat, FEATURE_UNLOCKS,
)
from gamification.streak import StreakTracker
from gamification.badges import check_and_award_badges
from gamification.leaderboard import update_leaderboard

router = APIRouter(prefix="/api/v1/gamification", tags=["gamification"])
cache = RedisCache("gamification")
streak_tracker = StreakTracker()


class AwardXPRequest(BaseModel):
    user_id: str
    action_type: str
    ref_id: Optional[str] = None
    pnl_z_score: float = 0.0
    sharpe_delta: float = 0.0
    current_regime: str = "unknown"
    is_wash_trade: bool = False


@router.post("/award")
async def award_xp(req: AwardXPRequest, db: AsyncSession = Depends(get_db)):
    """Internal endpoint called by trade service after trade events."""
    # Anti-cheat: check daily cap
    daily_key = f"{req.user_id}:xp_today"
    r = cache
    today_xp = await cache.get(daily_key) or 0

    valid, reason = check_anti_cheat(today_xp, req.action_type, req.is_wash_trade)
    if not valid:
        return {"awarded": 0, "reason": reason}

    # Load streak
    streak = await streak_tracker.get_streak(req.user_id)
    if req.action_type in ("trade_complete", "trade_profitable", "daily_login"):
        await streak_tracker.record_activity(req.user_id, req.ref_id or "", db)
        streak = await streak_tracker.get_streak(req.user_id)

    # Compute XP
    final_xp, multipliers = compute_xp_award(
        req.action_type,
        streak_days=streak,
        current_regime=req.current_regime,
        pnl_z_score=req.pnl_z_score,
        sharpe_delta=req.sharpe_delta,
    )

    # Cap to remaining daily allowance
    remaining = max(0, 1000 - today_xp)
    final_xp = min(final_xp, remaining)

    if final_xp == 0:
        return {"awarded": 0, "reason": "daily_cap_reached"}

    # Persist XP transaction
    tx_id = str(uuid.uuid4())
    await db.execute(
        text("""
            INSERT INTO xp_transactions (id, user_id, ts, action_type, base_xp, multipliers, final_xp, ref_id)
            VALUES (:id, :uid, NOW(), :action, :base_xp, :multipliers, :final_xp, :ref_id)
        """),
        {
            "id": tx_id,
            "uid": req.user_id,
            "action": req.action_type,
            "base_xp": 25,  # Base from action table
            "multipliers": json.dumps(multipliers),
            "final_xp": final_xp,
            "ref_id": req.ref_id,
        },
    )

    # Update user totals
    row = await db.execute(
        text("SELECT total_xp, level, unlocked_features FROM user_gamification WHERE user_id = :uid"),
        {"uid": req.user_id},
    )
    result = row.fetchone()
    if not result:
        await db.execute(
            text("INSERT INTO user_gamification (user_id, total_xp, level) VALUES (:uid, 0, 1) ON CONFLICT DO NOTHING"),
            {"uid": req.user_id},
        )
        result = type("R", (), {"total_xp": 0, "level": 1, "unlocked_features": []})()

    old_level = result.level
    new_total = (result.total_xp or 0) + final_xp
    new_level = compute_level(new_total)
    new_unlocks = get_new_unlocks(old_level, new_level)

    existing_features = list(result.unlocked_features or [])
    all_features = list(set(existing_features + new_unlocks))

    await db.execute(
        text("""
            UPDATE user_gamification
            SET total_xp = :total, level = :level, unlocked_features = :features, updated_at = NOW()
            WHERE user_id = :uid
        """),
        {"total": new_total, "level": new_level, "features": all_features, "uid": req.user_id},
    )
    await db.commit()

    # Update daily XP tracker
    await cache.set(daily_key, today_xp + final_xp, ttl=86400)

    # Check badges asynchronously
    new_badges = await check_and_award_badges(req.user_id, db)

    return {
        "awarded_xp": final_xp,
        "multipliers": multipliers,
        "new_total_xp": new_total,
        "new_level": new_level,
        "level_up": new_level > old_level,
        "new_unlocks": new_unlocks,
        "new_badges": new_badges,
        "next_level_xp": xp_for_level(new_level + 1),
    }


@router.get("/profile/{user_id}")
async def get_gamification_profile(user_id: str, db: AsyncSession = Depends(get_db)):
    cached = await cache.get(f"{user_id}:profile")
    if cached:
        return cached

    row = await db.execute(
        text("SELECT * FROM user_gamification WHERE user_id = :uid"),
        {"uid": user_id},
    )
    result = row.fetchone()
    if not result:
        return {"user_id": user_id, "total_xp": 0, "level": 1, "streak_days": 0}

    data = dict(result._mapping)
    data["next_level_xp"] = xp_for_level(data["level"] + 1)
    data["current_level_xp"] = xp_for_level(data["level"])
    data["xp_progress_pct"] = round(
        (data["total_xp"] - data["current_level_xp"]) /
        max(data["next_level_xp"] - data["current_level_xp"], 1) * 100, 1
    )
    await cache.set(f"{user_id}:profile", data, ttl=60)
    return data


@router.get("/leaderboard")
async def get_leaderboard(
    cohort: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    params: dict = {"limit": limit}
    cohort_clause = "AND cohort = :cohort" if cohort else ""
    if cohort:
        params["cohort"] = cohort

    rows = await db.execute(
        text(f"""
            SELECT ls.user_id, u.username, ls.score, ls.rank, ls.cohort,
                   ug.level, ug.streak_days
            FROM leaderboard_scores ls
            JOIN users u ON u.id = ls.user_id
            JOIN user_gamification ug ON ug.user_id = ls.user_id
            WHERE 1=1 {cohort_clause}
            ORDER BY ls.rank ASC
            LIMIT :limit
        """),
        params,
    )
    return {"leaderboard": [dict(r._mapping) for r in rows.fetchall()]}


@router.get("/xp/history")
async def get_xp_history(
    user_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    rows = await db.execute(
        text("""
            SELECT ts, action_type, base_xp, multipliers, final_xp
            FROM xp_transactions
            WHERE user_id = :uid
            ORDER BY ts DESC
            LIMIT :limit
        """),
        {"uid": user_id, "limit": limit},
    )
    return {"xp_history": [dict(r._mapping) for r in rows.fetchall()]}


@router.get("/unlocks")
async def get_feature_unlocks(user_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.execute(
        text("SELECT level, unlocked_features FROM user_gamification WHERE user_id = :uid"),
        {"uid": user_id},
    )
    result = row.fetchone()
    if not result:
        return {"unlocked": [], "locked": list(FEATURE_UNLOCKS.items())}

    unlocked = list(result.unlocked_features or [])
    locked = [
        {"level_required": lvl, "features": feats}
        for lvl, feats in FEATURE_UNLOCKS.items()
        if lvl > result.level
    ]
    return {"current_level": result.level, "unlocked": unlocked, "locked": locked}
