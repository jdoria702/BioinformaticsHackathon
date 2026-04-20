from __future__ import annotations

import os
from typing import Optional

from app.vectorDB.retriever import ChromaRetriever

class LessonService:
    """
    Provides topic context for the tutor.
    - Base context is hardcoded.
    - Augmented context is retrieved per-session from Chroma (uploaded user documents).
    """

    _TOPICS = {
        "sequence_alignment": (
            "Sequence alignment is the process of arranging DNA/RNA/protein sequences "
            "to identify regions of similarity. Common types: global (Needleman-Wunsch) "
            " and local (Smith-Waterman). Key concepts include scoring matrices, gap "
            "penalties, and dynamic programming."
        ),
        "blast": (
            "BLAST is a heuristic algorithm to find local similarities between sequences. "
            "It uses seed-and-extend, reports hits with E-values, bit scores, and alignments."
        ),
    }

    def __init__(self):
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
        collection_name = os.getenv("CHROMA_COLLECTION", "bio_tutor_docs")
        self._retriever = ChromaRetriever(persist_dir=persist_dir, collection_name=collection_name)

    # Function that allows agent to retrieve topics:
    def get_topic_context(self, topic: str) -> str:
        return self._TOPICS.get(
            topic,
            "General bioinformatics tutoring context. Ask the user clarifying questions and explain step-by-step."
        )

    # Function that queries Chroma using session ID + query and then formats the answer:
    def get_retrieved_context(self, query_text: str, session_id: Optional[str], k: int = 4) -> str:
        """Return a formatted retrieval context string for prompts."""
        where = {"session_id": session_id} if session_id else None
        chunks = self._retriever.query(query_text=query_text, k=k, where=where)
        if not chunks:
            return ""

        formatted = []
        for c in chunks:
            formatted.append(f"[source={c.source} score={c.score:.3f}]\n{c.text}")

        return "\n\n".join(formatted)
    
    # Create a second retriever
    ## change the collection name

    