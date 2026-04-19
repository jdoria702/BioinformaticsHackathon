import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class LLMService:
    """
    LLM Service.
    - Default: hardcoded response for local development.
    - Provider: Google Gemini when LLM_PROVIDER=google|gemini.

    Env vars:
      - LLM_PROVIDER=hardcoded_answer|google|gemini
      - GEMINI_API_KEY=...
      - GEMINI_MODEL=gemini-1.5-flash (default)
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "hardcoded_answer").lower()
        logger.info("LLMService initialized with provider=%r", self.provider)
    
    # Function that uses LLM to generate answer:
    def generate(self, prompt: str) -> Dict[str, str]:
        logger.debug("Generating response using provider=%r (prompt_length=%d)", self.provider, len(prompt))

        if self.provider == "hardcoded_answer":
            logger.debug("Using hardcoded response (no LLM provider configured).")
            return {
                "answer": (
                    "Hardcoded tutor response (LLM_PROVIDER not set).\n\n"
                    "I received your prompt and would respond with a helpful explanation here.\n"
                    "If you want real LLM output, set LLM_PROVIDER and configure credentials."
                )
            }

        if self.provider in {"google", "gemini"}:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")

            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

            # New SDK (google-genai). Lazy import so hardcoded mode still works without dependency.
            from google import genai

            client = genai.Client(api_key=api_key)

            # Keep it simple: we already embed all system instructions + context into the prompt.
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise RuntimeError("Gemini returned an empty response")

            return {"answer": text}

        logger.error("Unsupported or unconfigured LLM provider: %r", self.provider)
        raise NotImplementedError(f"Unknown/unconfigured LLM_PROVIDER={self.provider!r}")
