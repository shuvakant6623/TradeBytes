import numpy as np
import pandas as pd
from typing import Optional


def compute_hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """R/S analysis for Hurst exponent. H>0.5 trending, H<0.5 mean-reverting."""
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        chunks = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
        if not chunks:
            continue
        rs_vals = []
        for chunk in chunks:
            mean = np.mean(chunk)
            deviation = np.cumsum(chunk - mean)
            r = np.max(deviation) - np.min(deviation)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            tau.append(np.mean(rs_vals))

    if len(tau) < 2:
        return 0.5

    lags_arr = np.log(list(lags[:len(tau)]))
    tau_arr = np.log(tau)
    poly = np.polyfit(lags_arr, tau_arr, 1)
    return float(np.clip(poly[0], 0.0, 1.0))


def engineer_features(prices: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Compute all regime features using strict historical windows.

    Args:
        prices: DataFrame with columns [open, high, low, close, volume], DatetimeIndex
        sentiment_series: Optional hourly sentiment from news module (lagged 1 period)

    Returns:
        DataFrame of features, aligned to prices index, no future data
    """
    df = prices.copy()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # 1. Realised volatility (20-day rolling std of log returns)
    df["realised_vol"] = df["log_ret"].rolling(20, min_periods=10).std() * np.sqrt(252)

    # 2. Volatility-of-volatility (5-day std of rolling 5-day vol)
    df["vol_5d"] = df["log_ret"].rolling(5, min_periods=3).std()
    df["vol_of_vol"] = df["vol_5d"].rolling(10, min_periods=5).std()

    # 3. Rolling return z-score (20-day normalised cumulative return)
    roll_mean = df["log_ret"].rolling(20, min_periods=10).mean()
    roll_std = df["log_ret"].rolling(20, min_periods=10).std()
    df["ret_zscore"] = (df["log_ret"] - roll_mean) / (roll_std + 1e-8)

    # 4. Volume spike indicator (volume / 20-day avg volume)
    df["vol_spike"] = df["volume"] / (df["volume"].rolling(20, min_periods=10).mean() + 1)

    # 5. ATR ratio (Average True Range / close)
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - df["close"].shift(1)).abs()
    low_prev = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["atr_ratio"] = tr.rolling(14, min_periods=7).mean() / (df["close"] + 1e-8)

    # 6. Hurst exponent (100-day rolling window, computed every day)
    def rolling_hurst(series: pd.Series, window: int = 100) -> pd.Series:
        result = pd.Series(index=series.index, dtype=float)
        arr = series.values
        for i in range(window, len(arr) + 1):
            result.iloc[i - 1] = compute_hurst_exponent(arr[i - window:i])
        return result

    df["hurst"] = rolling_hurst(df["log_ret"].fillna(0))

    # 7. Bid-ask spread proxy (high-low / close)
    df["spread_proxy"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

    # 8. Sentiment variance (lagged 1 period to prevent leakage)
    if sentiment_series is not None:
        # Resample to daily, lag by 1 day
        daily_sentiment = sentiment_series.resample("1D").agg(["mean", "std"]).fillna(0)
        df["sentiment_var"] = daily_sentiment["std"].reindex(df.index, method="ffill").shift(1)
    else:
        df["sentiment_var"] = 0.0

    feature_cols = [
        "realised_vol", "vol_of_vol", "ret_zscore",
        "vol_spike", "atr_ratio", "hurst", "spread_proxy", "sentiment_var"
    ]

    return df[feature_cols].dropna()
