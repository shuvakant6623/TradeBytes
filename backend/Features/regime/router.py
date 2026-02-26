from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, UUID4
from typing import Optional
import pandas as pd

from core.database import get_db
from regime.inference import infer_regime, get_regime_history, load_active_model, _active_model
from regime.trainer import RegimeTrainer

router = APIRouter(prefix="/api/v1/regime", tags=["regime"])


class RegimeResponse(BaseModel):
    asset_id: str
    regime: str
    confidence: dict
    model_version: Optional[str] = None


class RetrainRequest(BaseModel):
    asset_id: str
    train_ratio: float = 0.8


@router.get("/current", response_model=RegimeResponse)
async def get_current_regime(
    asset_id: str = "SPY",
    db: AsyncSession = Depends(get_db),
):
    # Load recent price data
    from sqlalchemy import text
    rows = await db.execute(
        text("""
            SELECT time, open, high, low, close, volume
            FROM price_history
            WHERE asset_id = :asset_id
            ORDER BY time DESC
            LIMIT 300
        """),
        {"asset_id": asset_id},
    )
    data = rows.fetchall()
    if len(data) < 60:
        raise HTTPException(status_code=422, detail="Insufficient price data (need 60+ bars)")

    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
    df = df.sort_values("time").set_index("time")

    output = await infer_regime(asset_id, df, db)
    model = await load_active_model(db)

    return RegimeResponse(
        asset_id=asset_id,
        regime=output.regime,
        confidence=output.confidence,
        model_version=model.version,
    )


@router.get("/history")
async def get_history(
    asset_id: str = "SPY",
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    history = await get_regime_history(asset_id, limit, db)
    return {"asset_id": asset_id, "history": history}


@router.get("/transitions")
async def get_transition_matrix(db: AsyncSession = Depends(get_db)):
    model = await load_active_model(db)
    if model.hmm is None:
        raise HTTPException(status_code=503, detail="Model not trained")

    tm = model.hmm.transmat_.tolist()
    state_map = model.state_map
    return {
        "transition_matrix": tm,
        "state_map": {str(k): v for k, v in state_map.items()},
    }


@router.post("/retrain")
async def retrain(
    req: RetrainRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    async def _retrain():
        from datetime import datetime
        trainer = RegimeTrainer()
        prices = await trainer.load_price_data(db, req.asset_id)
        sentiment = await trainer.load_sentiment_data(db, req.asset_id)
        metrics = trainer.train(prices, sentiment, train_ratio=req.train_ratio)
        version = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        model_path = f"/tmp/regime_model_{version}.pkl"
        await trainer.save_model(db, version, metrics, model_path)
        # Invalidate cached model
        global _active_model
        _active_model = None

    background_tasks.add_task(_retrain)
    return {"status": "retrain_scheduled"}


@router.get("/model/status")
async def model_status(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import text
    row = await db.execute(
        text("SELECT version, trained_at, metrics, is_active FROM regime_model_versions WHERE is_active = TRUE")
    )
    result = row.fetchone()
    if not result:
        return {"status": "no_model_trained"}
    return {
        "version": result.version,
        "trained_at": result.trained_at,
        "metrics": result.metrics,
        "is_active": result.is_active,
    }
