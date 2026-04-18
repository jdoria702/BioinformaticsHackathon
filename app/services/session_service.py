import logging

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ChatMessage:
    role: str # "user" | "assistant" | "system"
    content: str

class SessionService:
    """
    Simple in-memory session store.
    Only use this for local development/testing.
    Swap with Redis or a real DB for production.
    """
    # We'll store chat history as a list of strings, and we'll store the session key as a string:
    def __init__(self):
        self._sessions: Dict[str, List[ChatMessage]] = {}
    
    # An agent needs to be able to retrieve a chat session's history:
    def get_history(self, session_id: str) -> List[dict]:
        messages = self._sessions.get(session_id, [])

        # Log if messages is empty or not:
        if messages is None: 
            logger.debug(f"Session '{session_id}' not found. Returning empty chat history.")
            return []

        logger.debug(f"Retrieved history for session '{session_id}' with {len(messages)} messages.")

        # Return a list of dicts to keep prompt building easy & JSON-friendly:
        chat_history = []

        for message in messages:
            chat_history.append({
                "role": message.role,
                "content": message.content
            })
        
        return chat_history
