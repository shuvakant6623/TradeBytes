import json
from typing import Any, Optional
import redis.asyncio as aioredis
from core.config import get_settings

settings = get_settings()

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis


class RedisCache:
    def __init__(self, prefix: str):
        self.prefix = prefix

    async def get(self, key: str) -> Optional[Any]:
        r = await get_redis()
        val = await r.get(f"{self.prefix}:{key}")
        return json.loads(val) if val else None

    async def set(self, key: str, value: Any, ttl: int = 60) -> None:
        r = await get_redis()
        await r.setex(f"{self.prefix}:{key}", ttl, json.dumps(value))

    async def delete(self, key: str) -> None:
        r = await get_redis()
        await r.delete(f"{self.prefix}:{key}")

    async def publish(self, channel: str, message: dict) -> None:
        r = await get_redis()
        await r.publish(channel, json.dumps(message))

    async def incr(self, key: str, amount: int = 1, ttl: int = 86400) -> int:
        r = await get_redis()
        pipe = r.pipeline()
        full_key = f"{self.prefix}:{key}"
        await pipe.incrby(full_key, amount)
        await pipe.expire(full_key, ttl)
        results = await pipe.execute()
        return results[0]
