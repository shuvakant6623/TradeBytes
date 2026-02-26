import json
import uuid
from datetime import datetime
from typing import Optional
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, UUID4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from core.database import get_db
from core.redis_client import RedisCache
from risk.metrics import RiskEngine, compute_log_returns, compute_max_drawdown

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])
cache = RedisCache("risk")
engine = RiskEngine()


async def _load_user_portfolio(user_id: str, db: AsyncSession) -> tuple[dict, dict]:
    """Returns (prices_dict, weights_dict) for a user's portfolio."""
    rows = await db.execute(
        text("""
            SELECT DISTINCT ON (t.asset_id)
                t.asset_id,
                t.quantity,
                t.entry_price,
                p.close AS current_price
            FROM trades t
            LEFT JOIN LATERAL (
                SELECT close FROM price_history
                WHERE asset_id = t.asset_id
                ORDER BY time DESC LIMIT 1
            ) p ON TRUE
            WHERE t.user_id = :user_id AND t.status = 'open'
        """),
        {"user_id": user_id},
    )
    positions = rows.fetchall()

    if not positions:
        return {}, {}

    total_value = sum((p.current_price or p.entry_price) * p.quantity for p in positions)
    weights = {}
    prices_dict = {}

    for pos in positions:
        asset_id = pos.asset_id
        val = (pos.current_price or pos.entry_price) * pos.quantity
        weights[asset_id] = val / (total_value + 1e-8)

        ph_rows = await db.execute(
            text("""
                SELECT time, close FROM price_history
                WHERE asset_id = :asset_id
                ORDER BY time DESC LIMIT 504
            """),
            {"asset_id": asset_id},
        )
        ph = ph_rows.fetchall()
        if ph:
            s = pd.Series(
                {r.time: r.close for r in reversed(ph)},
                dtype=float,
            )
            prices_dict[asset_id] = s

    return prices_dict, weights


@router.get("/summary/{user_id}")
async def get_risk_summary(
    user_id: str,
    window_days: int = Query(252, ge=20, le=504),
    db: AsyncSession = Depends(get_db),
):
    cached = await cache.get(f"{user_id}:summary:{window_days}")
    if cached:
        return cached

    prices_dict, weights = await _load_user_portfolio(user_id, db)

    if not prices_dict:
        raise HTTPException(status_code=404, detail="No open positions found for user")

    # Load market proxy (SPY)
    spy_rows = await db.execute(
        text("SELECT time, close FROM price_history WHERE asset_id='SPY' ORDER BY time DESC LIMIT 504")
    )
    spy_data = spy_rows.fetchall()
    market_prices = pd.Series({r.time: r.close for r in reversed(spy_data)}) if spy_data else None

    metrics = engine.compute(prices_dict, weights, market_prices, window_days)

    result = {
        "user_id": user_id,
        "computed_at": datetime.utcnow().isoformat(),
        "window_days": window_days,
        "volatility": metrics.volatility,
        "beta": metrics.beta,
        "max_drawdown": metrics.max_drawdown,
        "sharpe": metrics.sharpe,
        "hhi": metrics.hhi,
        "diversification": metrics.diversification,
        "health_score": metrics.health_score,
        "n_observations": metrics.n_observations,
        "warnings": [{"code": w.code, "message": w.message} for w in metrics.warnings],
    }

    # Persist snapshot
    await db.execute(
        text("""
            INSERT INTO portfolio_snapshots
              (id, user_id, computed_at, window_days, volatility, beta, max_drawdown,
               sharpe, hhi, diversification, health_score, correlation_matrix, weights, warnings)
            VALUES
              (:id, :user_id, NOW(), :window_days, :volatility, :beta, :max_drawdown,
               :sharpe, :hhi, :diversification, :health_score, :corr, :weights, :warnings)
        """),
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "window_days": window_days,
            "volatility": metrics.volatility,
            "beta": metrics.beta,
            "max_drawdown": metrics.max_drawdown,
            "sharpe": metrics.sharpe,
            "hhi": metrics.hhi,
            "diversification": metrics.diversification,
            "health_score": metrics.health_score,
            "corr": json.dumps(metrics.correlation_matrix),
            "weights": json.dumps(weights),
            "warnings": json.dumps([{"code": w.code, "message": w.message} for w in metrics.warnings]),
        },
    )
    await db.commit()

    await cache.set(f"{user_id}:summary:{window_days}", result, ttl=300)
    return result


@router.get("/correlation/{user_id}")
async def get_correlation(user_id: str, db: AsyncSession = Depends(get_db)):
    rows = await db.execute(
        text("""
            SELECT correlation_matrix FROM portfolio_snapshots
            WHERE user_id = :user_id
            ORDER BY computed_at DESC LIMIT 1
        """),
        {"user_id": user_id},
    )
    row = rows.fetchone()
    if not row or not row.correlation_matrix:
        raise HTTPException(status_code=404, detail="No snapshot found")
    return {"user_id": user_id, "correlation_matrix": row.correlation_matrix}


@router.get("/drawdown/{user_id}")
async def get_drawdown_series(user_id: str, days: int = 252, db: AsyncSession = Depends(get_db)):
    prices_dict, weights = await _load_user_portfolio(user_id, db)
    if not prices_dict:
        raise HTTPException(status_code=404, detail="No positions found")

    portfolio_prices = sum(
        prices.tail(days) * weights.get(asset, 0)
        for asset, prices in prices_dict.items()
    )

    cummax = portfolio_prices.cummax()
    drawdown_series = ((portfolio_prices - cummax) / cummax).fillna(0)

    return {
        "user_id": user_id,
        "drawdown_series": [
            {"date": str(d), "drawdown": float(v)}
            for d, v in drawdown_series.items()
        ],
        "max_drawdown": float(drawdown_series.min()),
    }


@router.get("/health/{user_id}")
async def get_health_score(user_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.execute(
        text("""
            SELECT health_score, sharpe, max_drawdown, diversification, volatility, beta, computed_at
            FROM portfolio_snapshots
            WHERE user_id = :user_id
            ORDER BY computed_at DESC LIMIT 1
        """),
        {"user_id": user_id},
    )
    result = row.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="No health data found")
    return dict(result._mapping)


class SimulateRequest(BaseModel):
    user_id: str
    hypothetical_positions: dict  # {asset_id: weight_delta}


@router.post("/simulate")
async def simulate_portfolio(req: SimulateRequest, db: AsyncSession = Depends(get_db)):
    prices_dict, weights = await _load_user_portfolio(req.user_id, db)

    # Apply hypothetical changes
    new_weights = {**weights}
    for asset, delta in req.hypothetical_positions.items():
        new_weights[asset] = max(0, new_weights.get(asset, 0) + delta)

    total = sum(new_weights.values())
    new_weights = {k: v / (total + 1e-8) for k, v in new_weights.items()}

    metrics = engine.compute(prices_dict, new_weights)

    return {
        "simulated": True,
        "new_weights": new_weights,
        "volatility": metrics.volatility,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "diversification": metrics.diversification,
        "health_score": metrics.health_score,
    }
