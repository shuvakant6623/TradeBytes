import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import pandas as pd

from core.database import get_db
from core.redis_client import RedisCache
from behavioral.profiler import BehavioralProfiler

router = APIRouter(prefix="/api/v1/behavioral", tags=["behavioral"])
cache = RedisCache("behavioral")
profiler = BehavioralProfiler()


async def _load_trades(user_id: str, db: AsyncSession) -> pd.DataFrame:
    rows = await db.execute(
        text("""
            SELECT id, asset_id, side, entry_price, exit_price, quantity,
                   entry_time, exit_time, pnl, pnl_pct, status, sector
            FROM trades
            WHERE user_id = :user_id
            ORDER BY entry_time ASC
        """),
        {"user_id": user_id},
    )
    return pd.DataFrame(rows.fetchall(), columns=[
        "id", "asset_id", "side", "entry_price", "exit_price", "quantity",
        "entry_time", "exit_time", "pnl", "pnl_pct", "status", "sector"
    ])


async def _save_profile(user_id: str, metrics, db: AsyncSession):
    await db.execute(
        text("""
            INSERT INTO behavioral_profiles
              (user_id, updated_at, feature_vector, archetype, risk_score, profit_factor,
               win_loss_asymmetry, overtrading_z, diversification_beh, loss_recovery_speed,
               disposition_effect, overconfidence_score, loss_aversion_ratio,
               trade_duration_p50, bias_flags)
            VALUES
              (:user_id, NOW(), :fv, :archetype, :risk, :pf, :wla, :ot_z, :div, :lrs,
               :de, :oc, :la, :dur_p50, :flags)
            ON CONFLICT (user_id) DO UPDATE SET
              updated_at = NOW(),
              feature_vector = EXCLUDED.feature_vector,
              archetype = EXCLUDED.archetype,
              risk_score = EXCLUDED.risk_score,
              profit_factor = EXCLUDED.profit_factor,
              win_loss_asymmetry = EXCLUDED.win_loss_asymmetry,
              overtrading_z = EXCLUDED.overtrading_z,
              diversification_beh = EXCLUDED.diversification_beh,
              loss_recovery_speed = EXCLUDED.loss_recovery_speed,
              disposition_effect = EXCLUDED.disposition_effect,
              overconfidence_score = EXCLUDED.overconfidence_score,
              loss_aversion_ratio = EXCLUDED.loss_aversion_ratio,
              trade_duration_p50 = EXCLUDED.trade_duration_p50,
              bias_flags = EXCLUDED.bias_flags
        """),
        {
            "user_id": user_id,
            "fv": metrics.feature_vector,
            "archetype": metrics.archetype,
            "risk": metrics.risk_score,
            "pf": metrics.profit_factor,
            "wla": metrics.win_loss_asymmetry,
            "ot_z": metrics.overtrading_z,
            "div": metrics.diversification_beh,
            "lrs": metrics.loss_recovery_speed,
            "de": metrics.biases.disposition_score,
            "oc": metrics.biases.overconfidence_score,
            "la": metrics.biases.loss_aversion_ratio,
            "dur_p50": metrics.trade_duration_p50,
            "flags": json.dumps({
                "disposition_effect": metrics.biases.disposition_effect,
                "overconfidence": metrics.biases.overconfidence,
                "loss_aversion": metrics.biases.loss_aversion,
            }),
        },
    )
    await db.commit()


@router.get("/profile/{user_id}")
async def get_profile(user_id: str, db: AsyncSession = Depends(get_db)):
    cached = await cache.get(f"{user_id}:profile")
    if cached:
        return cached

    row = await db.execute(
        text("SELECT * FROM behavioral_profiles WHERE user_id = :uid"),
        {"uid": user_id},
    )
    result = row.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Profile not found. Run recalculate first.")

    data = dict(result._mapping)
    await cache.set(f"{user_id}:profile", data, ttl=600)
    return data


@router.get("/biases/{user_id}")
async def get_biases(user_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.execute(
        text("SELECT bias_flags, disposition_effect, overconfidence_score, loss_aversion_ratio FROM behavioral_profiles WHERE user_id = :uid"),
        {"uid": user_id},
    )
    result = row.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {
        "user_id": user_id,
        "bias_flags": result.bias_flags,
        "scores": {
            "disposition_effect": result.disposition_effect,
            "overconfidence": result.overconfidence_score,
            "loss_aversion_ratio": result.loss_aversion_ratio,
        },
    }


@router.get("/history/{user_id}")
async def get_profile_history(user_id: str, limit: int = 20, db: AsyncSession = Depends(get_db)):
    # In production, store profile snapshots in a separate time-series table
    rows = await db.execute(
        text("""
            SELECT computed_at, health_score, sharpe
            FROM portfolio_snapshots
            WHERE user_id = :uid
            ORDER BY computed_at DESC
            LIMIT :limit
        """),
        {"uid": user_id, "limit": limit},
    )
    return {"history": [dict(r._mapping) for r in rows.fetchall()]}


@router.post("/recalculate")
async def recalculate_profile(
    user_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    async def _recalc():
        trades = await _load_trades(user_id, db)
        if len(trades) < 5:
            return
        metrics = profiler.compute_profile(trades)
        await _save_profile(user_id, metrics, db)
        await cache.delete(f"{user_id}:profile")

    background_tasks.add_task(_recalc)
    return {"status": "recalculation_scheduled", "user_id": user_id}
