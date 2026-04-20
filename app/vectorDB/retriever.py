from __future__ import annotations

import logging
import os
import uuid

from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "bio_tutor_docs" # Storage location of embeddings.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Lightweight embedding model from HuggingFace (can change later).

@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    score: float
    metadata: Dict

class ChromaRetriever:
    """
    Chroma-backed retriever. This handles:
    - add_chunks(texts, metadatas)
    - query(text)
    """
    def __init__(
        self, 
        persist_dir: str = "chroma_db", # Uploads will persist in memory.
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBED_MODEL,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection: Collection = self._client.get_or_create_collection(name=collection_name)

        self._embedder = SentenceTransformer(embedding_model)

        logger.info(
            "ChromaRetriever ready (persist_dir=%s, collection=%s, embed_model=%s)",
            self.persist_dir,
            collection_name,
            embedding_model,
        )

    # Function called whenever new content is being added to Chroma to add chunks to the DB (text -> chunk):
    def add_chunks(self, texts: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None) -> int:
        # Check if text is passed in:
        if not texts:
            logger.debug("No text passed to add_chunks().")
            return 0

        # Ensure ids exist (Chroma requires ids)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError("ids length must match texts length")

        # We should update the metadatas whenever we add chunks:
        def normalize_metadatas(texts, metadatas=None):

            # If the metadata list isn't provided by user, we can create a new one and initialize it using the text intead:
            if metadatas is None:
                metadatas = []

            # Create a list to store our normalized list of metadatas:
            normalized = []

            # Each text should have a corresponding metadata entry:
            for i in range(len(texts)):
                if i < len(metadatas) and metadatas[i] is not None:
                    normalized.append(metadatas[i])
                else:
                    normalized.append({})
            
            return normalized
        
        normalized_metadatas = normalize_metadatas(texts, metadatas)

        if len(normalized_metadatas) != len(texts):
            logger.error("Metadata/text length mismatch after normalization.")
            raise ValueError("Metadatas length must match texts length.")
        
        logger.debug("Metadata normalized successfully.")

        # Generate embeddings:
        embeddings = self._embedder.encode(texts, normalize_embeddings=True).tolist()

        logger.debug("Generated embeddings for %d chunks", len(embeddings))

        # Add embeddings to Chroma:
        self._collection.add(documents=texts, metadatas=normalized_metadatas, embeddings=embeddings, ids=ids)

        logger.info("Successfully added %d chunks to Chroma", len(texts))

        return len(texts)
    
    # We need to be able to query Chroma for relevant chunks:
    def query(self, query_text: str, k: int = 4, where: Optional[Dict] = None) -> List[RetrievedChunk]:
        # Check if query text exists:
        if not query_text.strip():
            logger.debug("No query text passed to query().")
            return []
        
        # Embed the query:
        logger.info("Querying Chroma (k=%d, where=%s)", k, where)
        query_embedding = self._embedder.encode([query_text], normalize_embeddings=True).tolist()

        logger.debug("Generated embedding for query")

        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=k, # Only return the top k results.
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns a lists-of-lists:
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        logger.debug(
            "Raw query results: docs=%d metas=%d dists=%d",
            len(documents),
            len(metadatas),
            len(distances),
        )

        out: List[RetrievedChunk] = []

        for document, metadata, distance in zip(documents, metadatas, distances):
            # Using cosine distance, smaller means closer. We'll convert this to a score for display purposes:
            if distance is not None:
                score = 1.0 - float(distance)
            else:
                score = 0.0
            
            if metadata is None:
                source = "unknown"
            else:
                source = metadata.get("source", "unknown")
            
            # Add retrieved chunk to output list:
            out.append(RetrievedChunk(text=document, source=source, score=score, metadata=metadata or {}))
        
        logger.info("Query returned %d chunks", len(out))
        
        return out
    
    # Delete documents from a specific location in our document database:
    def delete_where(self, where: Dict) -> None:
        """Delete documents matching a Chroma 'where' filter."""
        logger.info("Deleting from Chroma where=%s", where)
        self._collection.delete(where=where)

    # Delete all documents for a particular session ID:
    def delete_session(self, session_id: str) -> None:
        """Delete all documents associated with a session_id."""
        self.delete_where({"session_id": session_id})

    # List all files uploaded for a particular session:
    def list_session_uploaded_files(self, session_id: str) -> List[str]:
        """Return stored filenames referenced by Chroma metadata for a session."""
        # Chroma doesn't support a pure metadata-only scan efficiently; we fetch ids with where-filter.
        # Use include=['metadatas'] and no embeddings.
        result = self._collection.get(where={"session_id": session_id}, include=["metadatas"])
        metadatas = result.get("metadatas") or []

        out: List[str] = []
        for md in metadatas:
            if not md:
                continue
            name = md.get("stored_filename")
            if name:
                out.append(name)

        # de-dup while preserving order:
        seen = set()
        unique = []
        for n in out:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        return unique

    # List all files belonging to a session including metadata:
    def list_session_files_detailed(self, session_id: str) -> List[Dict]:
        """List unique uploaded files for a session with basic metadata."""
        result = self._collection.get(where={"session_id": session_id}, include=["metadatas"])
        metadatas = result.get("metadatas") or []

        files: Dict[str, Dict] = {}
        for md in metadatas:
            if not md:
                continue
            stored = md.get("stored_filename")
            if not stored:
                continue
            if stored not in files:
                files[stored] = {
                    "stored_filename": stored,
                    "original_filename": md.get("original_filename"),
                    "file_hash": md.get("file_hash"),
                    "file_type": md.get("file_type"),
                }
        return list(files.values())
    
    # Delete file for a session within the collection:
    def delete_session_file(self, session_id: str, stored_filename: str) -> None:
        """Delete all vectors belonging to a specific uploaded file within a session."""
        self.delete_where({"session_id": session_id, "stored_filename": stored_filename})
