from __future__ import annotations

import logging
import time

from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ChatMessage:
    role: str  # "user" | "assistant" | "system"
    content: str

class SessionService:
    """
    Simple in-memory session store.
    Only use this for local development/testing.
    Swap with Redis or a real DB for production.

    This service also tracks last access time so sessions can "expire" after inactivity.
    """

    def __init__(self):
        self._sessions: Dict[str, List[ChatMessage]] = {}
        self._last_access: Dict[str, float] = {}

    # Function to update last access time in session:
    def touch(self, session_id: str) -> None:
        self._last_access[session_id] = time.time()

    # Grab the last access time given a session:
    def get_last_access(self, session_id: str) -> Optional[float]:
        return self._last_access.get(session_id)

    # Check whether or not a session has expired:
    def is_expired(self, session_id: str, ttl_seconds: int) -> bool:
        last = self._last_access.get(session_id)
        if last is None:
            return False
        return (time.time() - last) > ttl_seconds

    # Delete the session from our dictionary:
    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._last_access.pop(session_id, None)
        logger.info("Deleted in-memory session '%s'", session_id)

    # An agent needs to be able to retrieve a chat session's history:
    def get_history(self, session_id: str) -> List[dict]:
        self.touch(session_id)
        messages = self._sessions.get(session_id, [])

        logger.debug("Retrieved history for session '%s' with %d messages.", session_id, len(messages))

        return [{"role": m.role, "content": m.content} for m in messages]

    # Allow the agent to append chat history to the session:
    def append(self, session_id: str, role: str, content: str) -> None:
        self.touch(session_id)
        self._sessions.setdefault(session_id, []).append(ChatMessage(role=role, content=content))
        logger.debug("Appended message to session '%s' (role=%s).", session_id, role)
