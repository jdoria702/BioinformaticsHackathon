import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class LLMService:
    """
    Minimal LLM Service for now.
    - Default: hardcoded response for local development.
    - Need to find an LLM provider to set LLM_PROVIDER!
    """
    # Grab the LLM provider, or use the hardcoded response if unavailable:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "hardcoded_answer").lower()
        logger.info(f"LLMService initialized with provider='{self.provider}'.")
    
    def generate(self, prompt: str) -> Dict[str, str]:
        logger.debug(f"Generating response using provider='{self.provider}' (prompt_length={len(prompt)}).")

        if self.provider == "hardcoded_answer":
            logger.debug("Using hardcoded response (no LLM provider configured).")

            # Deterministic hardcoded answer:
            return {
                "answer": (
                    "Hardcoded tutor response (LLM_PROVIDER not set).\n\n"
                    "I received your prompt and would respond with a helpful explanation here.\n"
                    "If you want real LLM output, set LLM_Provider and configure credentials."
                )
            }
        
        logger.error(f"Unsupported or unconfigured LLM provider: '{self.provider}'")
        
        # Placeholder for real provider wiring:
        raise NotImplementedError(f"Unknown/unconfigured LLM_PROVIDER={self.provider!r}")