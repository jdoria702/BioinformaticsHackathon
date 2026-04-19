import logging
import os
import uuid

from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.agent.tutor_agent import BioTutorAgent
from app.vectorDB.ingest import extract_blocks, chunk_blocks, build_block_metadata
from app.vectorDB.retriever import ChromaRetriever

# Create logger:
logger = logging.getLogger(__name__)

# Create a Flask blueprint that we can later register with our main app (app.py):
api_bp = Blueprint("api", __name__)

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

# Homepage:
@api_bp.route("/")
def index():
    return render_template("index.html")

# Simple health check to see if the application is running and functioning properly:
@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# Chat endpoint for all user messages:
@api_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id", "default")
    topic = data.get("topic", "sequence_alignment")

    if not message:
        return jsonify({"error": "Message is required."}), 400

    try:
        result = agent.respond(
            user_message=message,
            session_id=session_id,
            topic=topic
        )
        return jsonify(result)
    except Exception:
        logger.exception("Error while handling /chat (session_id=%s, topic=%s)", session_id, topic)
        return jsonify({"error": "Internal server error."}), 500

# File upload endpoint for user-specific documents:
@api_bp.route("/upload", methods=["POST"])
def upload():
    """Upload a file and ingest it into Chroma scoped to a session_id."""

    # Accept session_id from form field (preferred for multipart) or query param:
    session_id = request.form.get("session_id") or request.args.get("session_id") or "default"

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
        for md in metadatas:
            md["session_id"] = session_id
            md["original_filename"] = original_name

        retriever.add_chunks(chunks, metadatas=metadatas)

        logger.info(
            "Ingested upload session_id=%s original=%s stored=%s kind=%s chunks=%d",
            session_id,
            original_name,
            stored_name,
            kind,
            len(chunks),
        )

        return jsonify({
            "status": "ok",
            "session_id": session_id,
            "file_type": kind,
            "original_filename": original_name,
            "stored_filename": stored_name,
            "chunks_indexed": len(chunks),
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
