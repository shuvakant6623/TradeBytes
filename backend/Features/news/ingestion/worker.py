import asyncio
import hashlib
import json
import aiohttp
import structlog
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from simhash import Simhash

from core.redis_client import get_redis, RedisCache

logger = structlog.get_logger()
cache = RedisCache("news_dedup")

NEWS_SOURCES = [
    {"name": "benzinga", "url": "https://api.benzinga.com/api/v2/news?token={key}&pageSize=50"},
    {"name": "newsapi", "url": "https://newsapi.org/v2/top-headlines?category=business&apiKey={key}"},
]


@dataclass
class RawNewsEvent:
    headline: str
    body_excerpt: str
    url: str
    source: str
    published_at: datetime
    raw_text: str


class NewsDeduplicator:
    """Three-stage deduplication pipeline."""

    async def is_duplicate(self, event: RawNewsEvent, embedding: Optional[list] = None) -> bool:
        # Stage 1: Exact hash (SHA-256 of normalised headline)
        fingerprint = hashlib.sha256(event.headline.lower().strip().encode()).hexdigest()
        r = await get_redis()

        if await r.sismember("news:fingerprints", fingerprint):
            return True
        await r.sadd("news:fingerprints", fingerprint)
        await r.expire("news:fingerprints", 259200)  # 72h

        # Stage 2: SimHash near-duplicate (Hamming distance < 4)
        simhash_val = Simhash(event.headline).value
        recent_hashes = await r.lrange("news:simhashes", 0, 999)
        for existing in recent_hashes:
            existing_val = int(existing)
            # Hamming distance
            xor = simhash_val ^ existing_val
            distance = bin(xor).count("1")
            if distance < 4:
                return True

        await r.lpush("news:simhashes", str(simhash_val))
        await r.ltrim("news:simhashes", 0, 999)

        return False

    def compute_fingerprint(self, headline: str) -> str:
        return hashlib.sha256(headline.lower().strip().encode()).hexdigest()


class NewsIngestionWorker:
    """Async worker that fetches news and publishes to Redis Stream."""

    def __init__(self, source_config: dict, api_key: str):
        self.source_config = source_config
        self.api_key = api_key
        self.deduplicator = NewsDeduplicator()

    async def run(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            try:
                url = self.source_config["url"].format(key=self.api_key)
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.error("fetch_failed", source=self.source_config["name"], status=resp.status)
                        return

                    data = await resp.json()
                    articles = self._parse_response(data, self.source_config["name"])

                    published = 0
                    for article in articles:
                        if await self.deduplicator.is_duplicate(article):
                            continue
                        await self._publish(article)
                        published += 1

                    logger.info("ingestion_complete", source=self.source_config["name"], published=published)

            except Exception as e:
                logger.error("ingestion_error", source=self.source_config["name"], error=str(e))

    def _parse_response(self, data: dict, source: str) -> list[RawNewsEvent]:
        articles = []
        items = data.get("articles", data.get("data", []))
        for item in items:
            headline = item.get("title", item.get("headline", ""))
            if not headline:
                continue
            try:
                pub_at = datetime.fromisoformat(
                    item.get("publishedAt", item.get("created", datetime.utcnow().isoformat()))
                    .replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pub_at = datetime.utcnow()

            articles.append(RawNewsEvent(
                headline=headline,
                body_excerpt=item.get("description", item.get("body", ""))[:500],
                url=item.get("url", ""),
                source=source,
                published_at=pub_at,
                raw_text=f"{headline}. {item.get('description', '')}",
            ))
        return articles

    async def _publish(self, event: RawNewsEvent):
        r = await get_redis()
        await r.xadd("news:raw_stream", {
            "headline": event.headline,
            "body_excerpt": event.body_excerpt,
            "url": event.url,
            "source": event.source,
            "published_at": event.published_at.isoformat(),
            "raw_text": event.raw_text,
            "fingerprint": self.deduplicator.compute_fingerprint(event.headline),
        })


class IngestionScheduler:
    """Runs all ingestion workers on a schedule."""

    def __init__(self):
        self.workers: list[NewsIngestionWorker] = []

    def register(self, source_config: dict, api_key: str):
        self.workers.append(NewsIngestionWorker(source_config, api_key))

    async def run_forever(self, interval_seconds: int = 300):
        while True:
            tasks = [w.run() for w in self.workers]
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("ingestion_cycle_done", next_in=interval_seconds)
            await asyncio.sleep(interval_seconds)
