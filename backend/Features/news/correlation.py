import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


async def compute_news_price_correlation(
    ticker_id: str,
    db: AsyncSession,
    lag_bars: int = 1,
    window_days: int = 30,
) -> dict:
    """
    Measure statistical correlation between hourly sentiment and next-bar return.
    lag_bars: how many bars after publication to measure price impact.
    Returns Pearson-r, Spearman-r, p-value, and signal quality score.
    """
    # Load rolling sentiment
    sentiment_rows = await db.execute(
        text("""
            SELECT bucket, avg_sentiment
            FROM sentiment_rolling_1h
            WHERE ticker_id = :ticker_id
              AND bucket >= NOW() - INTERVAL ':window days'
            ORDER BY bucket ASC
        """.replace(":window", str(window_days))),
        {"ticker_id": ticker_id},
    )
    sentiment_df = pd.DataFrame(sentiment_rows.fetchall(), columns=["bucket", "avg_sentiment"])
    sentiment_df["bucket"] = pd.to_datetime(sentiment_df["bucket"])

    # Load hourly price data
    price_rows = await db.execute(
        text("""
            SELECT time_bucket('1 hour', time) AS hour, last(close, time) AS close
            FROM price_history
            WHERE asset_id = :asset_id
              AND time >= NOW() - INTERVAL ':window days'
            GROUP BY hour
            ORDER BY hour ASC
        """.replace(":window", str(window_days))),
        {"asset_id": ticker_id},
    )
    price_df = pd.DataFrame(price_rows.fetchall(), columns=["hour", "close"])
    price_df["hour"] = pd.to_datetime(price_df["hour"])
    price_df["log_ret"] = np.log(price_df["close"] / price_df["close"].shift(1))

    # Align on time index
    merged = pd.merge(
        sentiment_df.rename(columns={"bucket": "hour"}),
        price_df[["hour", "log_ret"]],
        on="hour",
        how="inner",
    )

    # Apply lag — shift returns forward by lag_bars
    merged["future_ret"] = merged["log_ret"].shift(-lag_bars)
    merged = merged.dropna()

    if len(merged) < 20:
        return {"error": "insufficient_data", "n": len(merged)}

    x = merged["avg_sentiment"].values
    y = merged["future_ret"].values

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    # Signal quality: combination of abs correlation and statistical significance
    signal_quality = abs(pearson_r) * (1 - pearson_p) if pearson_p < 0.05 else 0.0

    return {
        "ticker_id": ticker_id,
        "lag_bars": lag_bars,
        "window_days": window_days,
        "n_observations": len(merged),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 4),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 4),
        "signal_quality": round(float(signal_quality), 4),
        "is_significant": bool(pearson_p < 0.05),
    }


async def semantic_search(
    query_embedding: list[float],
    ticker_filter: str | None,
    limit: int,
    db: AsyncSession,
) -> list[dict]:
    """HNSW approximate nearest-neighbour search via pgvector."""
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    ticker_clause = "AND ntm.ticker_id = :ticker_id" if ticker_filter else ""
    params = {"embedding": embedding_str, "limit": limit}
    if ticker_filter:
        params["ticker_id"] = ticker_filter

    rows = await db.execute(
        text(f"""
            SELECT
                na.id, na.headline, na.published_at, na.source,
                na.sentiment_score, na.sentiment_label,
                1 - (na.embedding <=> :embedding::vector) AS similarity
            FROM news_articles na
            JOIN news_ticker_map ntm ON ntm.article_id = na.id
            {ticker_clause}
            ORDER BY na.embedding <=> :embedding::vector
            LIMIT :limit
        """),
        params,
    )
    return [dict(r._mapping) for r in rows.fetchall()]
