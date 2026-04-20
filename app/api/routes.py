import hashlib
import logging
import os
import threading
import time
import uuid
import time

from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.agent.tutor_agent import BioTutorAgent
from app.vectorDB.ingest import extract_blocks, chunk_blocks, build_block_metadata
from app.vectorDB.retriever import ChromaRetriever
from app.services.redis_session_store import RedisSessionStore

# Create logger:
logger = logging.getLogger(__name__)

# Create a Flask blueprint that we can later register with our main app (app.py):
api_bp = Blueprint("api", __name__)

# cache_key: -> { "answer": ..., "timestamp": ...}
response_cache = {}

# Expire cache after 1 hour
CACHE_TTL = 60 * 60

# Normalize queries so that strings are case insensitive
def normalize_query(text: str) -> str:
    return " ".join(text.lower().strip().split())

# Initialize the agent for chat endpoints (this needs to fail-fast because it's a core component of our app):
try:
    logger.info("Initializing BioTutorAgent...")
    agent = BioTutorAgent()
    logger.info("BioTutorAgent initialized successfully.")
except Exception:
    logger.exception("BioTutorAgent failed to initialize (fail-fast).")
    raise

# Initialize the retriever (also needs to fail-fast because it provides access to our specialized knowledge base):
try:
    logger.info("Initializing retriever...")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "bio_tutor_docs")
    retriever = ChromaRetriever(persist_dir=persist_dir, collection_name=collection_name)
    logger.info("Retriever initialized successfully.")
except Exception:
    logger.exception("Retriever failed to initialize (fail-fast).")
    raise

# We want file hashes to be deterministic to prevent duplicate uploads:
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Simple health check to see if the application is running and functioning properly:
@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# Session time-to-live:
# Support session-cleanup (user-uploaded files should be deleted from DB when session ends):
# Session end is indicated by server-side session expiry/timeout because the session ID is browser supplied (prevents spoofing and is reliable).
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))  # 1 hour default!

REQUIRE_SESSION_ID = os.getenv("REQUIRE_SESSION_ID", "1") == "1"

# Daemon sweeper/background cleanup job thread interval:
CLEANUP_SWEEP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_SWEEP_INTERVAL_SECONDS", "300"))  # 5 minutes.

# Our server requires the session from the browser/client in order for a request to be accepted:
def _require_session_id(session_id: str | None):
    if not session_id and REQUIRE_SESSION_ID:
        return jsonify({"error": "session_id is required"}), 400
    return None

# Minimal wiring for Redis session lifecyle:
REDIS_URL = os.getenv("REDIS_URL")
USE_REDIS_SESSIONS = bool(REDIS_URL)

if USE_REDIS_SESSIONS:
    redis_sessions = RedisSessionStore(redis_url=REDIS_URL, ttl_seconds=SESSION_TTL_SECONDS)
    logger.info("Redis session store enabled (REDIS_URL set)")
else:
    redis_sessions = None
    logger.info("Redis session store disabled (REDIS_URL not set)")

# Do our best to cleanup all expired-session data:
def _cleanup_session_data(session_id: str) -> None:
    """Best-effort cleanup of all state associated with a session."""

    # 0) Load file list from Redis if available (preferred):
    uploads_dir = os.getenv("UPLOADS_DIR", "uploads")
    stored_files = []
    if redis_sessions is not None:
        try:
            stored_files = redis_sessions.list_files(session_id)
        except Exception:
            logger.exception("Failed to read Redis file list for session_id=%s", session_id)

    # 1) Delete uploaded files first:
    try:
        # Use Chroma as a fallback if Redis returns an empty list of filess or if Redis fails:
        if not stored_files:
            stored_files = retriever.list_session_uploaded_files(session_id)
        deleted = 0
        for name in stored_files:
            path = os.path.join(uploads_dir, name)
            if os.path.exists(path):
                os.remove(path)
                deleted += 1
        logger.info("Deleted %d uploaded files for session_id=%s", deleted, session_id)
    except Exception:
        logger.exception("Failed to delete uploaded files for session_id=%s", session_id)

    # 2) Delete vectors:
    try:
        retriever.delete_session(session_id)
    except Exception:
        logger.exception("Failed to delete Chroma data for session_id=%s", session_id)

    # 3) Delete in-memory chat history:
    try:
        agent.session_service.delete_session(session_id)
    except Exception:
        logger.exception("Failed to delete in-memory session history for session_id=%s", session_id)

    # 4) Delete Redis session metadata
    if redis_sessions is not None:
        try:
            redis_sessions.delete(session_id)
        except Exception:
            logger.exception("Failed to delete Redis session for session_id=%s", session_id)

def _is_expired(session_id: str) -> bool:
    if redis_sessions is not None:
        last = redis_sessions.get_last_access(session_id)
        if last is None:
            return False
        return (time.time() - float(last)) > SESSION_TTL_SECONDS
    return agent.session_service.is_expired(session_id, SESSION_TTL_SECONDS)

# Touch/update a session's activity timestamp (in-memory + Redis if enabled):
def _touch_session(session_id: str) -> None:
    # Always keep in-memory activity for chat history coherence.
    agent.session_service.touch(session_id)

    if redis_sessions is not None:
        if not redis_sessions.exists(session_id):
            redis_sessions.create(session_id)
        redis_sessions.touch(session_id)

# Background sweeper function:
def _sweeper_loop():
    """Background sweeper to cleanup expired sessions even if no further requests arrive."""
    logger.info(
        "Starting background session sweeper (ttl=%s interval=%s)",
        SESSION_TTL_SECONDS,
        CLEANUP_SWEEP_INTERVAL_SECONDS,
    )
    while True:
        try:
            time.sleep(CLEANUP_SWEEP_INTERVAL_SECONDS)
            now = time.time()

            # Prefer Redis for session enumeration if enabled; otherwise fall back to in-memory:
            if redis_sessions is not None:
                # Scan keys: session:*:meta
                for meta_key in redis_sessions._r.scan_iter(match=f"{redis_sessions._prefix}:*:meta"):
                    # meta_key format: session:{id}:meta
                    parts = meta_key.split(":")
                    if len(parts) < 3:
                        continue
                    session_id = parts[1]
                    last = redis_sessions.get_last_access(session_id)
                    if last is not None and (now - float(last)) > SESSION_TTL_SECONDS:
                        logger.info("Sweeper expiring session_id=%s", session_id)
                        _cleanup_session_data(session_id)
            else:
                last_access = getattr(agent.session_service, "_last_access", {}).copy()
                for session_id, ts in last_access.items():
                    if (now - ts) > SESSION_TTL_SECONDS:
                        logger.info("Sweeper expiring session_id=%s", session_id)
                        _cleanup_session_data(session_id)
        except Exception:
            logger.exception("Session sweeper loop error")

# Start sweeper thread once at import time:
_sweeper_thread = threading.Thread(target=_sweeper_loop, name="session-sweeper", daemon=True)
_sweeper_thread.start()

# Automatic cleanup (this usually happens when a browser stays open but the last request was sent over an hour ago):
@api_bp.before_request
def expire_sessions_before_request():
    """Expire the current session_id if it has been idle past TTL."""
    if request.endpoint not in {"api.chat", "api.upload"}:
        return

    data = request.get_json(silent=True) or {}
    session_id = (
        request.form.get("session_id")
        or request.args.get("session_id")
        or data.get("session_id")
    )

    if not session_id:
        return

    if _is_expired(session_id):
        logger.info("Session expired (session_id=%s, ttl=%s). Cleaning up.", session_id, SESSION_TTL_SECONDS)
        _cleanup_session_data(session_id)

# Switching over to server-generated session IDs for greater control over cleanup:
@api_bp.route("/session/new", methods=["POST"])
def new_session():
    """Create a server-generated session id (avoids trusting browser-supplied ids)."""
    session_id = str(uuid.uuid4())
    _touch_session(session_id)
    return jsonify({"session_id": session_id, "ttl_seconds": SESSION_TTL_SECONDS})

# Explicitly end the session:
@api_bp.route("/session/end", methods=["POST"])
def end_session():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    _cleanup_session_data(session_id)
    return jsonify({"status": "ok"})

# Chat endpoint for all user messages:
@api_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id")
    topic = data.get("topic", "sequence_alignment")

    normalized_message = normalize_query(message)
    cache_key = f"{session_id}:{topic}:{normalized_message}"

    if not message:
        return jsonify({"error": "Message is required."}), 400

    # Require session ID to be provided:
    required = _require_session_id(session_id)
    if required:
        return required

    _touch_session(session_id)

    try:
        cached = response_cache.get(cache_key)

        # Cache hit and not expired
        if cached and time.time() - cached["timestamp"] < CACHE_TTL:
            return jsonify({
                "answer": cached["answer"],
                "cached": True
            })
        
        # Cache hit but time expired
        if cached:
            del response_cache[cache_key]


        # New request, call agent and store result in cache
        result = agent.respond(
            user_message=message,
            session_id=session_id,
            topic=topic
        )

        # Cache answer and timestamp
        answer = result.get("answer", "")
        if answer:
            response_cache[cache_key] = {
                "answer": answer,
                "timestamp": time.time()
            }

        return jsonify(result)
    except Exception:
        logger.exception("Error while handling /chat (session_id=%s, topic=%s)", session_id, topic)
        return jsonify({"error": "Internal server error."}), 500

# File upload endpoint for user-specific documents:
@api_bp.route("/upload", methods=["POST"])
def upload():
    """Upload a file and ingest it into Chroma scoped to a session_id."""

    session_id = request.form.get("session_id") or request.args.get("session_id")

    required = _require_session_id(session_id)
    if required:
        return required

    _touch_session(session_id)

    # Make sure a file is sent:
    if "file" not in request.files:
        return jsonify({"error": "Missing file field 'file'."}), 400
    
    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "No file selected."}), 400

    # Set the upload directory:
    uploads_dir = os.getenv("UPLOADS_DIR", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    # Store file name (original + UUID version):
    original_name = secure_filename(f.filename)
    stored_name = f"{uuid.uuid4()}_{original_name}"
    stored_path = os.path.join(uploads_dir, stored_name)

    # Extract blocks from file upload:
    try:
        f.save(stored_path)

        file_hash = _sha256_file(stored_path)

        # Make sure that we handle empty extraction:
        blocks, kind = extract_blocks(stored_path)
        if not blocks:
            return jsonify({"error": "No extractable content found in file."}), 400

        # Make sure we handle empty chunks:
        chunks = chunk_blocks(blocks, max_chars=1200, overlap=150)
        if not chunks:
            return jsonify({"error": "No chunks could be generated from the file"}), 400
        
        # Build metadata per chunk and scope it to this session:
        metadatas = build_block_metadata(
            path=stored_path,
            kind=kind,
            block_count=len(chunks),
            ingestion_method="upload",
        )
        ids = []
        for i, md in enumerate(metadatas):
            md["session_id"] = session_id
            md["original_filename"] = original_name
            md["stored_filename"] = stored_name
            md["file_hash"] = file_hash
            md["chunk_index"] = i

            # Deterministic IDs => prevents duplicate storage within session for same file:
            ids.append(f"{session_id}:{file_hash}:{i}")

        # Prefer upsert if available (avoids duplicate-id errors):
        try:
            retriever._collection.upsert(
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
                embeddings=retriever._embedder.encode(chunks, normalize_embeddings=True).tolist(),
            )
            indexed = len(chunks)
        except Exception:
            # Fallback to add_chunks (may raise if ids already exist)
            indexed = retriever.add_chunks(chunks, metadatas=metadatas, ids=ids)

        logger.info(
            "Ingested upload session_id=%s original=%s stored=%s kind=%s chunks=%d file_hash=%s",
            session_id,
            original_name,
            stored_name,
            kind,
            indexed,
            file_hash,
        )

        # After stored_name is created, record it in Redis for cleanup:
        if redis_sessions is not None:
            try:
                redis_sessions.add_file(session_id, stored_name)
            except Exception:
                logger.exception("Failed to record stored file in Redis (session_id=%s file=%s)", session_id, stored_name)

        return jsonify({
            "status": "ok",
            "session_id": session_id,
            "file_type": kind,
            "original_filename": original_name,
            "stored_filename": stored_name,
            "chunks_indexed": indexed,
            "file_hash": file_hash,
        })

    # Reject upload (likely to happen if extension is unsupported):
    except ValueError as e:
        logger.warning(
            "Rejected upload session_id=%s filename=%s error=%s",
            session_id,
            original_name,
            e,
        )
        return jsonify({"error": str(e)}), 400

    # Ingestion failure:
    except Exception:
        logger.exception("Failed to ingest uploaded file (session_id=%s, filename=%s)", session_id, original_name)

        # Cleanup (don't want file to stay on disk if ingestion fails):
        if os.path.exists(stored_path):
            os.remove(stored_path)

        return jsonify({"error": "Failed to ingest file."}), 500
