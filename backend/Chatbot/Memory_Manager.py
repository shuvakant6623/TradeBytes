"""
Session-based conversation memory manager.
Uses in-memory dict (Redis-ready interface).
"""
import time
import logging
from typing import List, Dict, Optional
from collections import defaultdict
from core.config import settings

logger = logging.getLogger("finai.memory")

class ConversationTurn:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.timestamp = time.time()

class MemoryManager:
    """
    In-memory session store with TTL-based expiry.
    Drop-in replaceable with Redis for production.
    """
    def __init__(self):
        self._store: Dict[str, List[ConversationTurn]] = defaultdict(list)
        self._last_access: Dict[str, float] = {}
        logger.info("MemoryManager initialized (in-memory mode)")

    def _evict_expired(self):
        """Remove sessions older than TTL"""
        now = time.time()
        expired = [
            sid for sid, last in self._last_access.items()
            if now - last > settings.SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._store[sid]
            del self._last_access[sid]
            logger.info(f"Evicted expired session: {sid}")

    def add_turn(self, session_id: str, role: str, content: str):
        self._evict_expired()
        self._store[session_id].append(ConversationTurn(role, content))
        self._last_access[session_id] = time.time()

        # Keep only last N turns (sliding window)
        if len(self._store[session_id]) > settings.MAX_HISTORY_TURNS * 2:
            self._store[session_id] = self._store[session_id][-settings.MAX_HISTORY_TURNS * 2:]

    def get_history(self, session_id: str) -> List[ConversationTurn]:
        self._last_access[session_id] = time.time()
        return self._store.get(session_id, [])

    def get_history_as_text(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return ""
        lines = []
        for turn in history[-settings.MAX_HISTORY_TURNS * 2:]:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def clear_session(self, session_id: str):
        self._store.pop(session_id, None)
        self._last_access.pop(session_id, None)

    def session_count(self) -> int:
        return len(self._store)

# Singleton instance
memory_manager = MemoryManager()