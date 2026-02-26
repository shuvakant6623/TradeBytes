import asyncio
import json
import uuid
from datetime import datetime
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from news.nlp.pipeline import get_nlp_pipeline
from core.database import AsyncSessionLocal
from core.redis_client import get_redis

logger = structlog.get_logger()
STREAM_KEY = "news:raw_stream"
CONSUMER_GROUP = "news_processors"
CONSUMER_NAME = "processor_1"


class NewsProcessor:
    """
    Consumes from Redis Stream, runs NLP pipeline, stores results.
    Timestamp alignment: published_at is respected as-is (UTC).
    Sentiment is attributed to the bar AFTER publication (1-period lag in aggregation).
    """

    async def ensure_consumer_group(self):
        r = await get_redis()
        try:
            await r.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
        except Exception:
            pass  # Group already exists

    async def run_forever(self):
        await self.ensure_consumer_group()
        logger.info("news_processor_started")
        while True:
            try:
                await self._process_batch()
            except Exception as e:
                logger.error("processor_error", error=str(e))
                await asyncio.sleep(5)

    async def _process_batch(self):
        r = await get_redis()
        messages = await r.xreadgroup(
            CONSUMER_GROUP, CONSUMER_NAME,
            {STREAM_KEY: ">"},
            count=32,
            block=5000,
        )
        if not messages:
            return

        pipeline = get_nlp_pipeline()
        stream_messages = messages[0][1]  # [(msg_id, fields), ...]

        texts = [msg[1]["raw_text"] for msg in stream_messages]
        nlp_results = pipeline.process_batch(texts)

        async with AsyncSessionLocal() as db:
            for (msg_id, fields), nlp in zip(stream_messages, nlp_results):
                try:
                    article_id = await self._insert_article(db, fields, nlp)
                    await self._insert_ticker_maps(db, article_id, fields["headline"], nlp.tickers)
                except Exception as e:
                    logger.error("article_insert_failed", error=str(e), url=fields.get("url"))

            await db.commit()

        # Acknowledge processed messages
        ids = [msg[0] for msg in stream_messages]
        await r.xack(STREAM_KEY, CONSUMER_GROUP, *ids)
        logger.info("batch_processed", count=len(stream_messages))

    async def _insert_article(self, db: AsyncSession, fields: dict, nlp) -> str:
        article_id = str(uuid.uuid4())
        embedding_str = "[" + ",".join(str(x) for x in nlp.embedding) + "]"

        await db.execute(
            text("""
                INSERT INTO news_articles
                  (id, published_at, source, headline, body_excerpt, url, fingerprint,
                   sentiment_score, sentiment_label, embedding, model_version)
                VALUES
                  (:id, :published_at, :source, :headline, :body_excerpt, :url, :fingerprint,
                   :sentiment_score, :sentiment_label, :embedding::vector, 'finbert_v1')
                ON CONFLICT (url) DO NOTHING
            """),
            {
                "id": article_id,
                "published_at": fields["published_at"],
                "source": fields["source"],
                "headline": fields["headline"],
                "body_excerpt": fields["body_excerpt"],
                "url": fields.get("url", ""),
                "fingerprint": fields["fingerprint"],
                "sentiment_score": nlp.sentiment.score,
                "sentiment_label": nlp.sentiment.label,
                "embedding": embedding_str,
            },
        )
        return article_id

    async def _insert_ticker_maps(self, db: AsyncSession, article_id: str, headline: str, tickers: list):
        for ticker in tickers:
            await db.execute(
                text("""
                    INSERT INTO news_ticker_map (article_id, ticker_id, confidence)
                    VALUES (:article_id, :ticker_id, 1.0)
                    ON CONFLICT DO NOTHING
                """),
                {"article_id": article_id, "ticker_id": ticker},
            )
