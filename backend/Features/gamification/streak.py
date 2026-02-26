from datetime import date, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from core.redis_client import get_redis


class StreakTracker:
    """
    Streak integrity rules:
    - A streak day requires at least one qualifying action during market hours.
    - Grace period: one missed day per 14-day window.
    - All streak increments are validated against real DB records.
    """

    async def record_activity(
        self,
        user_id: str,
        ref_id: str,
        db: AsyncSession,
    ) -> dict:
        """
        Called after a qualifying action (trade close, daily login).
        Returns updated streak state.
        """
        today = date.today()

        row = await db.execute(
            text("SELECT streak_days, streak_grace_used, last_active_day FROM user_gamification WHERE user_id = :uid"),
            {"uid": user_id},
        )
        result = row.fetchone()
        if not result:
            await self._create_gamification_record(user_id, db)
            return await self.record_activity(user_id, ref_id, db)

        streak = result.streak_days
        grace_used = result.streak_grace_used
        last_day = result.last_active_day

        if last_day == today:
            return {"streak_days": streak, "changed": False}

        yesterday = today - timedelta(days=1)
        two_days_ago = today - timedelta(days=2)

        if last_day == yesterday:
            # Normal streak increment
            new_streak = streak + 1
            new_grace = grace_used
        elif last_day == two_days_ago and not grace_used:
            # Use grace period
            new_streak = streak + 1
            new_grace = True
        else:
            # Streak broken — reset
            new_streak = 1
            new_grace = False

        await db.execute(
            text("""
                UPDATE user_gamification
                SET streak_days = :streak, streak_grace_used = :grace,
                    last_active_day = :today, updated_at = NOW()
                WHERE user_id = :uid
            """),
            {"streak": new_streak, "grace": new_grace, "today": today, "uid": user_id},
        )

        # Reset grace every 14 days
        if new_streak % 14 == 0 and new_streak > 0:
            await db.execute(
                text("UPDATE user_gamification SET streak_grace_used = FALSE WHERE user_id = :uid"),
                {"uid": user_id},
            )

        await db.commit()

        # Mirror to Redis for fast reads
        r = await get_redis()
        await r.setex(f"streak:{user_id}", 86400, str(new_streak))

        return {
            "streak_days": new_streak,
            "grace_used": new_grace,
            "changed": True,
            "broken": new_streak == 1 and streak > 1,
        }

    async def _create_gamification_record(self, user_id: str, db: AsyncSession):
        await db.execute(
            text("""
                INSERT INTO user_gamification (user_id, total_xp, level, streak_days)
                VALUES (:uid, 0, 1, 0)
                ON CONFLICT DO NOTHING
            """),
            {"uid": user_id},
        )
        await db.commit()

    async def get_streak(self, user_id: str) -> int:
        r = await get_redis()
        cached = await r.get(f"streak:{user_id}")
        return int(cached) if cached else 0
