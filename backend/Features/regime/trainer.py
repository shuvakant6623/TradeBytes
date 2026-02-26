import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

from regime.model import RegimeDetectionModel
from regime.features import engineer_features
from core.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class RegimeTrainer:
    def __init__(self):
        self.model = RegimeDetectionModel(
            n_components=settings.HMM_N_COMPONENTS,
            lookback=settings.HMM_LOOKBACK,
        )
        self._reference_features: Optional[np.ndarray] = None

    async def load_price_data(self, db: AsyncSession, asset_id: str) -> pd.DataFrame:
        rows = await db.execute(
            text("""
                SELECT time, open, high, low, close, volume
                FROM price_history
                WHERE asset_id = :asset_id
                ORDER BY time ASC
            """),
            {"asset_id": asset_id},
        )
        df = pd.DataFrame(rows.fetchall(), columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time")

    async def load_sentiment_data(self, db: AsyncSession, asset_id: str) -> Optional[pd.Series]:
        rows = await db.execute(
            text("""
                SELECT bucket, avg_sentiment
                FROM sentiment_rolling_1h
                WHERE ticker_id = :ticker_id
                ORDER BY bucket ASC
            """),
            {"ticker_id": asset_id},
        )
        data = rows.fetchall()
        if not data:
            return None
        df = pd.DataFrame(data, columns=["bucket", "avg_sentiment"])
        df["bucket"] = pd.to_datetime(df["bucket"])
        return df.set_index("bucket")["avg_sentiment"]

    def train(
        self,
        prices: pd.DataFrame,
        sentiment: Optional[pd.Series] = None,
        train_ratio: float = 0.8,
    ) -> dict:
        """
        Walk-forward training with strict chronological split.
        Returns evaluation metrics on hold-out set.
        """
        features = engineer_features(prices, sentiment)
        X = features.values
        n = len(X)

        if n < 200:
            raise ValueError(f"Insufficient data: {n} samples, need at least 200.")

        # Strict chronological split — NO shuffling
        split = int(n * train_ratio)
        X_train = X[:split]
        X_test = X[split:]

        logger.info("training_regime_model", train_n=split, test_n=len(X_test))

        # Store reference features for PSI drift detection
        self._reference_features = X_train.copy()

        self.model.fit(X_train)

        # Evaluate on hold-out (walk-forward accuracy proxy)
        train_states = self.model.hmm.predict(self.model.scaler.transform(X_train))
        regime_distribution = {
            label: int((np.array(list(self.model.state_map.values())) == label).sum())
            for label in ["trending", "mean_reverting", "high_volatility", "low_volatility", "news_driven"]
        }

        # Log-likelihood on test set
        X_test_scaled = self.model.scaler.transform(X_test)
        test_ll = float(self.model.hmm.score(X_test_scaled))

        metrics = {
            "train_samples": split,
            "test_samples": len(X_test),
            "test_log_likelihood": test_ll,
            "state_distribution": regime_distribution,
            "state_map": self.model.state_map,
        }
        logger.info("training_complete", **metrics)
        return metrics

    async def save_model(self, db: AsyncSession, version: str, metrics: dict, model_path: str) -> None:
        import json
        self.model.version = version
        self.model.save(model_path)

        await db.execute(
            text("UPDATE regime_model_versions SET is_active = FALSE WHERE is_active = TRUE")
        )
        await db.execute(
            text("""
                INSERT INTO regime_model_versions (version, trained_at, train_end, metrics, is_active, model_path)
                VALUES (:version, NOW(), :train_end, :metrics, TRUE, :model_path)
                ON CONFLICT (version) DO UPDATE
                  SET trained_at = NOW(), metrics = :metrics, is_active = TRUE
            """),
            {
                "version": version,
                "train_end": date.today().isoformat(),
                "metrics": json.dumps(metrics),
                "model_path": model_path,
            },
        )
        await db.commit()

    def check_drift(self, X_current: np.ndarray) -> dict:
        """Returns PSI score and retrain recommendation."""
        if self._reference_features is None:
            return {"psi": 0.0, "needs_retrain": False}
        psi = self.model.compute_psi(self._reference_features, X_current)
        return {"psi": psi, "needs_retrain": psi > 0.25}
