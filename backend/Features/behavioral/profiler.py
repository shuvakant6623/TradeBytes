import json
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import structlog

from behavioral.metrics import (
    compute_risk_tolerance_score, compute_profit_factor, compute_win_loss_asymmetry,
    compute_overtrading_z, compute_diversification_behaviour, compute_loss_recovery_speed,
    compute_trade_duration_stats, compute_disposition_effect, compute_overconfidence,
    compute_loss_aversion, build_feature_vector, BiasFlags, BehavioralMetrics,
)

logger = structlog.get_logger()

ARCHETYPE_NAMES = [
    "Disciplined Systematic",
    "Aggressive Momentum Trader",
    "Risk-Averse Accumulator",
    "Overtrader",
    "Inconsistent Speculator",
    "Balanced Opportunist",
]

_kmeans: Optional[KMeans] = None
_scaler: Optional[StandardScaler] = None


def get_or_train_kmeans(feature_vectors: Optional[list[list[float]]] = None) -> tuple[KMeans, StandardScaler]:
    global _kmeans, _scaler
    if _kmeans is not None:
        return _kmeans, _scaler

    # If no vectors provided, create a synthetic representative set
    if feature_vectors is None or len(feature_vectors) < 10:
        rng = np.random.RandomState(42)
        feature_vectors = _generate_synthetic_vectors(rng)

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(feature_vectors))

    km = KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=500)
    km.fit(X)

    _kmeans = km
    _scaler = scaler
    return _kmeans, _scaler


def _generate_synthetic_vectors(rng: np.random.RandomState) -> list[list[float]]:
    """Synthetic archetypes for cold-start K-Means."""
    prototypes = [
        [40, 1.9, 1.4, -0.5, 0.72, 0.80, 0.02, 0.5, 1.2, 1440],   # Disciplined
        [80, 0.9, 0.8,  2.5, 0.30, 0.40, 0.15, 2.5, 1.5,  60],    # Aggressive
        [20, 1.3, 1.1, -1.0, 0.50, 0.60, 0.05, 0.3, 2.8, 7200],   # Risk-averse
        [55, 0.8, 0.7,  3.5, 0.40, 0.35, 0.25, 2.0, 1.8,  30],    # Overtrader
        [60, 0.7, 0.6,  1.0, 0.25, 0.25, 0.20, 1.5, 2.0, 240],    # Inconsistent
        [50, 1.5, 1.2,  0.0, 0.60, 0.65, 0.08, 1.0, 1.5, 480],    # Balanced
    ]
    vectors = []
    for proto in prototypes:
        for _ in range(50):
            noise = rng.normal(0, 0.1, len(proto))
            vectors.append([max(0, p + p * n) for p, n in zip(proto, noise)])
    return vectors


def classify_archetype(feature_vector: list[float]) -> str:
    km, scaler = get_or_train_kmeans()
    X = scaler.transform([feature_vector])
    cluster_id = int(km.predict(X)[0])
    return ARCHETYPE_NAMES[cluster_id % len(ARCHETYPE_NAMES)]


class BehavioralProfiler:
    """Full profiling pipeline. Called after each qualifying event (10 new trades)."""

    def compute_profile(self, trades: pd.DataFrame) -> BehavioralMetrics:
        # Ensure required columns
        for col in ["pnl", "pnl_pct", "status", "entry_time", "entry_price", "quantity"]:
            if col not in trades.columns:
                trades[col] = 0

        risk_score = compute_risk_tolerance_score(trades)
        pf = compute_profit_factor(trades)
        wla = compute_win_loss_asymmetry(trades)
        ot_z = compute_overtrading_z(trades)
        div_beh = compute_diversification_behaviour(trades)
        lrs = compute_loss_recovery_speed(trades)
        dur = compute_trade_duration_stats(trades)

        # Bias detection
        de_score, de_flag = compute_disposition_effect(trades)
        oc_score, oc_flag = compute_overconfidence(ot_z, risk_score)
        la_ratio, la_flag = compute_loss_aversion(trades)

        evidence = []
        if de_flag:
            evidence.append("Sells winners early (PGR significantly > PLR)")
        if oc_flag:
            evidence.append(f"Trades {oc_score:.1f}σ above expected frequency with outsized positions")
        if la_flag:
            evidence.append(f"Losses held {la_ratio:.1f}x longer than wins on average")

        biases = BiasFlags(
            disposition_effect=de_flag,
            overconfidence=oc_flag,
            loss_aversion=la_flag,
            disposition_score=de_score,
            overconfidence_score=oc_score,
            loss_aversion_ratio=la_ratio,
            evidence=evidence,
        )

        metrics = BehavioralMetrics(
            risk_score=risk_score,
            profit_factor=pf,
            win_loss_asymmetry=wla,
            overtrading_z=ot_z,
            diversification_beh=div_beh,
            loss_recovery_speed=lrs,
            trade_duration_p50=dur["p50"],
            trade_duration_p25=dur["p25"],
            trade_duration_p75=dur["p75"],
            biases=biases,
            feature_vector=[],
        )

        metrics.feature_vector = build_feature_vector(metrics)
        metrics.archetype = classify_archetype(metrics.feature_vector)

        return metrics
