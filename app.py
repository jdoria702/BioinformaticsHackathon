import logging
import os

from flask import Flask, render_template, redirect, request
from flask_scss import Scss
from app.api.routes import api_bp

# Create a logger for later debugging:
logger = logging.getLogger(__name__)

def _configure_logging():
    """
    Use this function to configure app-wide logging at startup.
    - INFO by default.
    - DEBUG when FLASK_DEBUG=1 or app.debug is enabled (usually reserved for development).
    """
    # Don't double-config if handlers already exist:
    root = logging.getLogger()
    
    if root.handlers:
        return
    
    level = logging.DEBUG if os.getenv("FLASK_DEBUG") == "1" else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

# Function that creates + starts our application:
def create_app():
    # Start the logger:
    _configure_logging()
    logger.info("Starting Flask app creation...")

    app = Flask(__name__)
    logger.info("Flask app instantiated.")

    # Register the API blueprint:
    app.register_blueprint(api_bp, url_prefix="/api")
    logger.info("Registered blueprint: api_bp at url_prefix=/api")

    logger.info("App creation completed.")

    return app

if __name__ == "__main__":
    app = create_app() 
    logger.info("Running development server (debug=%s)", True)
    app.run(debug=True)
