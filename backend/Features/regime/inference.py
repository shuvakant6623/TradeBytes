import os
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

from regime.model import RegimeDetectionModel, RegimeOutput
from regime.features import engineer_features
from core.config import get_settings
from core.redis_client import RedisCache

logger = structlog.get_logger()
settings = get_settings()
cache = RedisCache("regime")

_active_model: Optional[RegimeDetectionModel] = None


async def load_active_model(db: AsyncSession) -> RegimeDetectionModel:
    global _active_model
    if _active_model is not None:
        return _active_model

    row = await db.execute(
        text("SELECT model_path, version FROM regime_model_versions WHERE is_active = TRUE LIMIT 1")
    )
    result = row.fetchone()

    model = RegimeDetectionModel(
        n_components=settings.HMM_N_COMPONENTS,
        lookback=settings.HMM_LOOKBACK,
    )

    if result and os.path.exists(result.model_path):
        model.load(result.model_path)
        logger.info("regime_model_loaded", version=result.version)
    else:
        logger.warning("no_trained_model_found_using_defaults")

    _active_model = model
    return _active_model


async def infer_regime(
    asset_id: str,
    prices: pd.DataFrame,
    db: AsyncSession,
    sentiment: Optional[pd.Series] = None,
) -> RegimeOutput:
    """
    Run online regime inference and persist result.
    """
    cache_key = f"{asset_id}:latest"
    cached = await cache.get(cache_key)
    if cached:
        return RegimeOutput(**cached)

    model = await load_active_model(db)
    features = engineer_features(prices, sentiment)

    if len(features) == 0:
        raise ValueError("Insufficient price data to compute features.")

    output = model.predict(features.values)

    # Persist to TimescaleDB
    import json
    await db.execute(
        text("""
            INSERT INTO regime_states (time, asset_id, regime, confidence, model_version, features)
            VALUES (NOW(), :asset_id, :regime, :confidence, :model_version, :features)
        """),
        {
            "asset_id": asset_id,
            "regime": output.regime,
            "confidence": json.dumps(output.confidence),
            "model_version": model.version,
            "features": json.dumps(features.iloc[-1].to_dict()),
        },
    )
    await db.commit()

    # Cache for 5 minutes
    await cache.set(cache_key, {
        "regime": output.regime,
        "confidence": output.confidence,
        "state_id": output.state_id,
    }, ttl=300)

    # Publish regime change event to Redis
    from core.redis_client import RedisCache
    pub = RedisCache("events")
    await pub.publish("regime_updates", {
        "asset_id": asset_id,
        "regime": output.regime,
        "confidence": output.confidence,
        "ts": datetime.utcnow().isoformat(),
    })

    return output


async def get_regime_history(asset_id: str, limit: int, db: AsyncSession) -> list:
    rows = await db.execute(
        text("""
            SELECT time, regime, confidence
            FROM regime_states
            WHERE asset_id = :asset_id
            ORDER BY time DESC
            LIMIT :limit
        """),
        {"asset_id": asset_id, "limit": limit},
    )
    return [{"time": r.time, "regime": r.regime, "confidence": r.confidence} for r in rows.fetchall()]
