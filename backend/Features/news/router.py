from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta

from core.database import get_db
from news.correlation import compute_news_price_correlation, semantic_search
from news.nlp.pipeline import get_nlp_pipeline

router = APIRouter(prefix="/api/v1/news", tags=["news"])


@router.get("/feed")
async def get_news_feed(
    ticker: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    filters = []
    params: dict = {"limit": limit, "offset": offset}

    base_query = """
        SELECT na.id, na.headline, na.published_at, na.source,
               na.sentiment_score, na.sentiment_label, na.url,
               array_agg(ntm.ticker_id) AS tickers
        FROM news_articles na
        LEFT JOIN news_ticker_map ntm ON ntm.article_id = na.id
    """

    if ticker:
        filters.append("ntm.ticker_id = :ticker")
        params["ticker"] = ticker
    if sentiment:
        filters.append("na.sentiment_label = :sentiment")
        params["sentiment"] = sentiment

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    query = f"{base_query} {where} GROUP BY na.id ORDER BY na.published_at DESC LIMIT :limit OFFSET :offset"

    rows = await db.execute(text(query), params)
    articles = [dict(r._mapping) for r in rows.fetchall()]
    return {"articles": articles, "count": len(articles)}


@router.get("/sentiment/{ticker}")
async def get_rolling_sentiment(
    ticker: str,
    hours: int = Query(168, description="Look-back in hours (default 7 days)"),
    db: AsyncSession = Depends(get_db),
):
    rows = await db.execute(
        text("""
            SELECT bucket, avg_sentiment, sentiment_variance, article_count,
                   bullish_count, bearish_count
            FROM sentiment_rolling_1h
            WHERE ticker_id = :ticker
              AND bucket >= NOW() - (INTERVAL '1 hour' * :hours)
            ORDER BY bucket ASC
        """),
        {"ticker": ticker, "hours": hours},
    )
    data = [dict(r._mapping) for r in rows.fetchall()]
    return {"ticker": ticker, "sentiment_series": data}


@router.get("/correlation/{ticker}")
async def get_news_price_correlation(
    ticker: str,
    lag_bars: int = Query(1, ge=1, le=8),
    window_days: int = Query(30, ge=7, le=90),
    db: AsyncSession = Depends(get_db),
):
    result = await compute_news_price_correlation(ticker, db, lag_bars, window_days)
    return result


class SemanticSearchRequest(BaseModel):
    query: str
    ticker: Optional[str] = None
    limit: int = 10


@router.post("/search/semantic")
async def semantic_news_search(
    req: SemanticSearchRequest,
    db: AsyncSession = Depends(get_db),
):
    pipeline = get_nlp_pipeline()
    query_embedding = pipeline.embedder.encode(req.query)
    results = await semantic_search(query_embedding, req.ticker, req.limit, db)
    return {"results": results}


@router.get("/impact/{ticker}")
async def get_news_impact(
    ticker: str,
    days: int = 30,
    db: AsyncSession = Depends(get_db),
):
    """Returns articles with their associated next-bar price move."""
    rows = await db.execute(
        text("""
            SELECT
                na.headline, na.published_at, na.sentiment_score, na.sentiment_label,
                ph.close AS close_at_publish,
                LEAD(ph.close) OVER (ORDER BY ph.time) AS close_next_bar,
                (LEAD(ph.close) OVER (ORDER BY ph.time) - ph.close) / ph.close AS pct_move
            FROM news_articles na
            JOIN news_ticker_map ntm ON ntm.article_id = na.id
            JOIN price_history ph ON ph.asset_id = :ticker
              AND ph.time >= na.published_at
              AND ph.time < na.published_at + INTERVAL '1 hour'
            WHERE ntm.ticker_id = :ticker
              AND na.published_at >= NOW() - INTERVAL ':days days'
            ORDER BY na.published_at DESC
        """.replace(":days", str(days))),
        {"ticker": ticker},
    )
    return {"ticker": ticker, "impact_data": [dict(r._mapping) for r in rows.fetchall()]}
