import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from core.config import get_settings

settings = get_settings()


@dataclass
class RiskWarning:
    code: str
    message: str


@dataclass
class RiskMetrics:
    volatility: Optional[float]
    beta: Optional[float]
    max_drawdown: Optional[float]
    sharpe: Optional[float]
    hhi: Optional[float]
    diversification: Optional[float]
    health_score: Optional[float]
    correlation_matrix: Optional[dict]
    warnings: list[RiskWarning] = field(default_factory=list)
    n_observations: int = 0


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Log returns prevent compounding bias."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_volatility(returns: pd.Series) -> Optional[float]:
    """Annualised volatility from log returns."""
    if len(returns) < 10:
        return None
    return float(returns.std() * np.sqrt(252))


def compute_beta(asset_returns: pd.Series, market_returns: pd.Series) -> Optional[float]:
    """OLS beta vs market proxy. Requires aligned series."""
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(aligned) < 30:
        return None
    a, m = aligned.iloc[:, 0].values, aligned.iloc[:, 1].values
    cov = np.cov(a, m)[0, 1]
    var_m = np.var(m, ddof=1)
    if var_m < 1e-12:
        return None
    return float(cov / var_m)


def compute_max_drawdown(prices: pd.Series) -> Optional[float]:
    """O(n) single-pass MDD calculation."""
    if len(prices) < 2:
        return None
    arr = prices.values.astype(float)
    peak = arr[0]
    mdd = 0.0
    for p in arr[1:]:
        if p > peak:
            peak = p
        dd = (p - peak) / peak
        if dd < mdd:
            mdd = dd
    return float(mdd)


def compute_sharpe(returns: pd.Series, risk_free_rate: float = None) -> Optional[float]:
    """Annualised Sharpe ratio."""
    if len(returns) < 30:
        return None
    rf = risk_free_rate if risk_free_rate is not None else settings.RISK_FREE_RATE
    rf_daily = rf / 252
    excess = returns - rf_daily
    std = float(excess.std())
    if std < 1e-12:
        return None
    return float(excess.mean() / std * np.sqrt(252))


def compute_correlation_matrix(returns_df: pd.DataFrame) -> Optional[dict]:
    """
    Pearson correlation matrix for portfolio assets.
    For n > 50 assets: uses incremental Welford (not implemented here, shown as comment).
    """
    clean = returns_df.dropna(how="all").fillna(0)
    if clean.shape[1] < 2 or len(clean) < 10:
        return None
    corr = clean.corr()
    return corr.to_dict()


def compute_hhi(weights: dict[str, float]) -> float:
    """Herfindahl-Hirschman Index. Returns DI = 1 - HHI."""
    w = np.array(list(weights.values()), dtype=float)
    w = w / (w.sum() + 1e-8)
    hhi = float(np.sum(w ** 2))
    di = 1.0 - hhi
    return di


def compute_health_score(
    sharpe: Optional[float],
    mdd: Optional[float],
    diversification: Optional[float],
    volatility: Optional[float],
    beta: Optional[float],
) -> Optional[float]:
    """
    Composite health score [0, 100].
    PHS = 0.25×norm(Sharpe) + 0.25×(1−norm(MDD)) + 0.20×norm(DI)
        + 0.15×norm(1/σ) + 0.15×(1−|β−1|/2)
    """
    components = []

    # Sharpe component: clamp to [-1, 4] then normalise to [0, 1]
    if sharpe is not None:
        s_norm = np.clip((sharpe + 1) / 5, 0, 1)
        components.append(("sharpe", float(s_norm), 0.25))

    # Drawdown component: MDD ∈ [-1, 0], invert
    if mdd is not None:
        dd_norm = 1.0 - min(abs(mdd), 1.0)
        components.append(("mdd", float(dd_norm), 0.25))

    # Diversification component: already ∈ [0, 1]
    if diversification is not None:
        components.append(("di", float(np.clip(diversification, 0, 1)), 0.20))

    # Volatility component: lower vol = higher score. Normalise 1/σ, clamp
    if volatility is not None and volatility > 0:
        vol_score = np.clip(1.0 / (1.0 + volatility * 5), 0, 1)
        components.append(("vol", float(vol_score), 0.15))

    # Beta component: ideal beta = 1, penalise deviation
    if beta is not None:
        beta_score = max(0.0, 1.0 - abs(beta - 1.0) / 2.0)
        components.append(("beta", float(beta_score), 0.15))

    if not components:
        return None

    total_weight = sum(w for _, _, w in components)
    score = sum(v * w for _, v, w in components) / total_weight
    return round(float(score * 100), 2)


class RiskEngine:
    """Main risk computation orchestrator."""

    def compute(
        self,
        prices_dict: dict[str, pd.Series],   # {asset_id: price_series}
        weights: dict[str, float],
        market_prices: Optional[pd.Series] = None,
        window_days: int = 252,
    ) -> RiskMetrics:
        warnings: list[RiskWarning] = []

        # Build returns matrix
        returns_dict = {
            asset: compute_log_returns(prices.iloc[-window_days:])
            for asset, prices in prices_dict.items()
            if len(prices) > 1
        }

        if not returns_dict:
            return RiskMetrics(
                volatility=None, beta=None, max_drawdown=None,
                sharpe=None, hhi=None, diversification=None,
                health_score=None, correlation_matrix=None,
                warnings=[RiskWarning("NO_DATA", "No price data available")],
            )

        returns_df = pd.DataFrame(returns_dict)
        n_obs = len(returns_df.dropna(how="all"))

        if n_obs < 30:
            warnings.append(RiskWarning("SMALL_SAMPLE", f"Only {n_obs} observations; metrics may be unreliable"))

        # Portfolio returns (weighted)
        aligned_weights = {k: weights.get(k, 0) for k in returns_dict}
        total_w = sum(aligned_weights.values())
        norm_weights = {k: v / (total_w + 1e-8) for k, v in aligned_weights.items()}

        portfolio_returns = sum(
            returns_dict[a] * w for a, w in norm_weights.items()
        ).fillna(0)

        # Compute all metrics
        volatility = compute_volatility(portfolio_returns)
        sharpe = compute_sharpe(portfolio_returns)
        mdd = compute_max_drawdown(
            pd.concat([prices_dict[a].iloc[-window_days:] for a in returns_dict], axis=1)
            .mul(pd.Series(norm_weights)).sum(axis=1)
        )

        # Beta vs market
        beta = None
        if market_prices is not None:
            market_ret = compute_log_returns(market_prices.iloc[-window_days:])
            beta = compute_beta(portfolio_returns, market_ret)

        # Correlation matrix — check for high correlation warning
        corr_matrix = compute_correlation_matrix(returns_df)
        if corr_matrix and len(returns_dict) > 1:
            corr_vals = [
                corr_matrix[a][b]
                for a in corr_matrix
                for b in corr_matrix[a]
                if a != b
            ]
            if corr_vals and np.mean([abs(v) for v in corr_vals]) > 0.85:
                warnings.append(RiskWarning("HIGH_CORRELATION", "Portfolio assets are highly correlated; diversification benefit is limited"))

        diversification = compute_hhi(norm_weights)
        health = compute_health_score(sharpe, mdd, diversification, volatility, beta)

        return RiskMetrics(
            volatility=volatility,
            beta=beta,
            max_drawdown=mdd,
            sharpe=sharpe,
            hhi=1.0 - diversification if diversification is not None else None,
            diversification=diversification,
            health_score=health,
            correlation_matrix=corr_matrix,
            warnings=warnings,
            n_observations=n_obs,
        )
