import logging

from flask import Blueprint, render_template, request, jsonify
from app.agent.tutor_agent import BioTutorAgent

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
    data = request.get_json() or {}
    message = data.get("message", "")
    session_id = data.get("session_id", "default")
    topic = data.get("topic", "sequence_alignment")

    if not message.strip():
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
    return jsonify({"error": "/upload not implemented."}), 501
    