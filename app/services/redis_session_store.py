from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import List, Optional

import redis

logger = logging.getLogger(__name__)

class RedisSessionStore:
    """Redis-backed session metadata + uploaded file list.

    Keys:
      - session:{id}:meta -> hash {created_at,last_access}
      - session:{id}:files -> set of stored filenames

    TTL is applied to both keys.

    TLDR; for a session ID, two keys are created, one for metadata, the other for associated files.

    Example:
        Given session_id = "abc123"

        1. Metadata (hash) -> session:abc123:meta -> 
        {
           created_at: 1713500000,
           last_access: 1713500300
        }

        2. Files (set) -> session:abc123:files -> ["file1.pdf", "file2.txt"]
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        prefix: str = "session",
    ):
        # Connect to Redis:
        self._r = redis.Redis.from_url(redis_url, decode_responses=True)
        self._ttl = int(ttl_seconds)
        self._prefix = prefix

    # Build Redis keys:
    def _meta_key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}:meta"

    def _files_key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}:files"

    # Create metadata, set TTL, clears old file lists:
    def create(self, session_id: str) -> None:
        now = int(time.time())
        meta_key = self._meta_key(session_id)
        pipe = self._r.pipeline()
        pipe.hset(meta_key, mapping={"created_at": now, "last_access": now})
        pipe.expire(meta_key, self._ttl)
        pipe.delete(self._files_key(session_id))
        pipe.execute()

    # Check if a session exists:
    def exists(self, session_id: str) -> bool:
        return bool(self._r.exists(self._meta_key(session_id)))

    # Keep session alive by updating last access time:
    def touch(self, session_id: str) -> None:
        now = int(time.time())
        meta_key = self._meta_key(session_id)
        pipe = self._r.pipeline()
        pipe.hset(meta_key, "last_access", now)
        pipe.expire(meta_key, self._ttl)
        pipe.expire(self._files_key(session_id), self._ttl)
        pipe.execute()

    # Check last access time:
    def get_last_access(self, session_id: str) -> Optional[int]:
        v = self._r.hget(self._meta_key(session_id), "last_access")
        return int(v) if v is not None else None

    # Add a file to session:
    def add_file(self, session_id: str, stored_filename: str) -> None:
        files_key = self._files_key(session_id)
        pipe = self._r.pipeline()
        pipe.sadd(files_key, stored_filename)
        pipe.expire(files_key, self._ttl)
        pipe.execute()

    # Get all files for a session:
    def list_files(self, session_id: str) -> List[str]:
        return sorted(self._r.smembers(self._files_key(session_id)))

    # Hard delete session metadata and files:
    def delete(self, session_id: str) -> None:
        pipe = self._r.pipeline()
        pipe.delete(self._meta_key(session_id))
        pipe.delete(self._files_key(session_id))
        pipe.execute()

    # Remove a specific file from Redis store:
    def remove_file(self, session_id: str, stored_filename: str) -> None:
        files_key = self._files_key(session_id)
        pipe = self._r.pipeline()
        pipe.srem(files_key, stored_filename)
        pipe.expire(files_key, self._ttl)
        pipe.execute()
