import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats


@dataclass
class BiasFlags:
    disposition_effect: bool = False
    overconfidence: bool = False
    loss_aversion: bool = False
    disposition_score: float = 0.0
    overconfidence_score: float = 0.0
    loss_aversion_ratio: float = 1.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class BehavioralMetrics:
    risk_score: float
    profit_factor: float
    win_loss_asymmetry: float
    overtrading_z: float
    diversification_beh: float
    loss_recovery_speed: float
    trade_duration_p50: float
    trade_duration_p25: float
    trade_duration_p75: float
    biases: BiasFlags
    feature_vector: list[float]
    archetype: Optional[str] = None


def compute_risk_tolerance_score(trades: pd.DataFrame) -> float:
    """
    RTS = α×avg_position_size_pct + β×max_loss_pct + γ×leverage_proxy
    Normalised to [0, 100]. Higher = more risk-tolerant.
    """
    if trades.empty:
        return 50.0

    avg_pos_pct = trades["quantity"].mean() * trades["entry_price"].mean()
    max_loss = trades[trades["pnl"] < 0]["pnl"].min() if len(trades[trades["pnl"] < 0]) else 0
    max_loss_pct = abs(max_loss / (avg_pos_pct + 1e-8))

    raw = (0.4 * min(avg_pos_pct / 10000, 1.0) +
           0.4 * min(max_loss_pct * 10, 1.0) +
           0.2 * 0.5)  # leverage placeholder

    return float(np.clip(raw * 100, 0, 100))


def compute_profit_factor(trades: pd.DataFrame) -> float:
    """PF = gross_profit / gross_loss. Industry threshold > 1.5."""
    closed = trades[trades["status"] == "closed"].copy()
    if closed.empty:
        return 0.0
    gross_profit = closed[closed["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(closed[closed["pnl"] < 0]["pnl"].sum())
    if gross_loss < 1e-8:
        return float(min(gross_profit / 1e-8, 5.0))
    return float(np.clip(gross_profit / gross_loss, 0, 5.0))


def compute_win_loss_asymmetry(trades: pd.DataFrame) -> float:
    """WLA = avg_win_pct / avg_loss_pct. > 1 = favourable R:R."""
    closed = trades[trades["status"] == "closed"].copy()
    wins = closed[closed["pnl"] > 0]["pnl_pct"]
    losses = closed[closed["pnl"] < 0]["pnl_pct"].abs()
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 1
    return float(np.clip(avg_win / (avg_loss + 1e-8), 0, 3.0))


def compute_overtrading_z(trades: pd.DataFrame, expected_weekly_trades: float = 5.0) -> float:
    """Z-score of weekly trade frequency vs expected. > 2.0 = overtrading."""
    if len(trades) < 7:
        return 0.0
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    weekly = trades.groupby(trades["entry_time"].dt.isocalendar().week).size()
    z = (weekly.mean() - expected_weekly_trades) / (weekly.std() + 1e-8)
    return float(np.clip(z, -3, 5))


def compute_diversification_behaviour(trades: pd.DataFrame) -> float:
    """DivB = 1 - HHI(trade_count_by_sector)."""
    if trades.empty or "sector" not in trades.columns:
        return 0.5
    sector_counts = trades["sector"].value_counts(normalize=True)
    hhi = float((sector_counts ** 2).sum())
    return float(1.0 - hhi)


def compute_loss_recovery_speed(trades: pd.DataFrame) -> float:
    """LRS: normalised speed of portfolio recovery from drawdown events."""
    closed = trades[trades["status"] == "closed"].sort_values("exit_time").copy()
    if len(closed) < 10:
        return 0.5

    pnl = closed["pnl"].values
    cumulative = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumulative)
    underwater = cumulative < peak

    recovery_periods = []
    i = 0
    while i < len(underwater):
        if underwater[i]:
            start = i
            while i < len(underwater) and underwater[i]:
                i += 1
            recovery_periods.append(i - start)
        i += 1

    if not recovery_periods:
        return 1.0

    avg_recovery = float(np.mean(recovery_periods))
    return float(1.0 / (1.0 + avg_recovery / 10))


def compute_trade_duration_stats(trades: pd.DataFrame) -> dict:
    """Duration in minutes."""
    closed = trades[
        (trades["status"] == "closed") &
        trades["exit_time"].notna()
    ].copy()

    if closed.empty:
        return {"p25": 0, "p50": 0, "p75": 0}

    closed["duration"] = (
        pd.to_datetime(closed["exit_time"]) - pd.to_datetime(closed["entry_time"])
    ).dt.total_seconds() / 60

    durations = closed["duration"].dropna()
    return {
        "p25": float(durations.quantile(0.25)),
        "p50": float(durations.quantile(0.50)),
        "p75": float(durations.quantile(0.75)),
    }


def compute_disposition_effect(trades: pd.DataFrame, market_prices: Optional[dict] = None) -> tuple[float, bool]:
    """
    Odean (1998) methodology.
    PGR = realised_gains / (realised_gains + paper_gains)
    PLR = realised_losses / (realised_losses + paper_losses)
    DE = PGR - PLR. Positive = disposition effect present.
    """
    closed = trades[trades["status"] == "closed"].copy()
    if len(closed) < 20:
        return 0.0, False

    gains_realised = len(closed[closed["pnl"] > 0])
    losses_realised = len(closed[closed["pnl"] < 0])

    # Estimate paper positions from open trades
    open_trades = trades[trades["status"] == "open"].copy()
    gains_paper = len(open_trades[open_trades.get("unrealised_pnl", pd.Series(dtype=float)) > 0]) if "unrealised_pnl" in open_trades else len(open_trades) // 2
    losses_paper = len(open_trades) - gains_paper

    pgr = gains_realised / max(gains_realised + gains_paper, 1)
    plr = losses_realised / max(losses_realised + losses_paper, 1)
    de_score = pgr - plr

    # Binomial test: is PGR significantly > PLR?
    n_total = gains_realised + losses_realised
    p_val = stats.binom_test(gains_realised, n=n_total, p=0.5, alternative="greater") if n_total > 0 else 1.0
    is_significant = bool(de_score > 0.05 and p_val < 0.05)

    return float(de_score), is_significant


def compute_overconfidence(overtrading_z: float, risk_score: float) -> tuple[float, bool]:
    """OC = (trade_freq_z + position_size_z) / 2. > 1.5 = overconfident."""
    position_z = (risk_score - 50) / 20  # Normalise risk score to z-score
    oc_score = (overtrading_z + position_z) / 2
    return float(oc_score), bool(oc_score > 1.5)


def compute_loss_aversion(trades: pd.DataFrame) -> tuple[float, bool]:
    """LA_ratio = avg_loss_held_duration / avg_win_held_duration. > 2.0 = loss averse."""
    closed = trades[
        (trades["status"] == "closed") & trades["exit_time"].notna()
    ].copy()

    if len(closed) < 10:
        return 1.0, False

    closed["duration"] = (
        pd.to_datetime(closed["exit_time"]) - pd.to_datetime(closed["entry_time"])
    ).dt.total_seconds()

    wins = closed[closed["pnl"] > 0]["duration"]
    losses = closed[closed["pnl"] < 0]["duration"]

    avg_win_dur = wins.mean() if len(wins) > 0 else 1
    avg_loss_dur = losses.mean() if len(losses) > 0 else 1

    ratio = float(avg_loss_dur / (avg_win_dur + 1e-8))
    return float(np.clip(ratio, 0, 5)), bool(ratio > 2.0)


def build_feature_vector(metrics: "BehavioralMetrics") -> list[float]:
    return [
        metrics.risk_score,
        metrics.profit_factor,
        metrics.win_loss_asymmetry,
        metrics.overtrading_z,
        metrics.diversification_beh,
        metrics.loss_recovery_speed,
        metrics.biases.disposition_score,
        metrics.biases.overconfidence_score,
        metrics.biases.loss_aversion_ratio,
        metrics.trade_duration_p50,
    ]
