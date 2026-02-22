"""Module 3: Experience Library.

Captures structured experience summaries on score-change events
and retrieves relevant past experiences via RAG (ChromaDB + embeddings).
"""

import os
import logging
import hashlib
from . import config

logger = logging.getLogger(__name__)


class ExperienceLib:
    """Experience Library with RAG for reflective learning.
    
    When score changes (gain or loss/death), the system summarizes 
    the interaction history into structured experience and stores it 
    in ChromaDB. During gameplay, relevant experiences are retrieved 
    using embedding similarity.
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.experiences = []     # raw list of experience texts
        self._collection = None   # ChromaDB collection (lazy init)
        self._chroma_client = None

    def reset(self):
        """Reset experience library for a fresh start."""
        self.experiences = []
        # Don't reset ChromaDB - experiences persist across epochs
        # (this is key to learning across epochs)

    def _init_chroma(self):
        """Lazily initialize ChromaDB."""
        if self._collection is not None:
            return

        os.makedirs(self.persist_dir, exist_ok=True)

        import chromadb
        self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB initialized at {self.persist_dir} "
                     f"with {self._collection.count()} existing experiences")

    def store_experience(self, experience_text: str, metadata: dict = None):
        """Store an experience summary in the library.
        
        Args:
            experience_text: The structured experience summary from LLM
            metadata: Optional metadata (location, score, epoch, etc.)
        """
        if not experience_text or experience_text.strip() == "":
            return

        self.experiences.append(experience_text)

        # Store in ChromaDB for retrieval
        try:
            self._init_chroma()
            doc_id = hashlib.md5(experience_text.encode()).hexdigest()
            
            self._collection.add(
                documents=[experience_text],
                ids=[doc_id],
                metadatas=[metadata or {}],
            )
            logger.info(f"Stored experience (total: {len(self.experiences)})")
        except Exception as e:
            logger.warning(f"Failed to store experience in ChromaDB: {e}")
            # Still keep in memory list

    def retrieve_relevant(self, query: str, top_k: int = None) -> str:
        """Retrieve relevant past experiences using RAG.

        Args:
            query: Current game context to search against
            top_k: Number of experiences to retrieve

        Returns:
            Formatted string of relevant experiences
        """
        top_k = top_k or config.EXPERIENCE_TOP_K

        # Guard against empty DB — check ChromaDB, not the in-memory list.
        # self.experiences is cleared on reset() but ChromaDB persists across
        # epochs (that cross-epoch persistence is the paper's core learning mechanism).
        try:
            self._init_chroma()
            if self._collection.count() == 0:
                return "No relevant experiences found yet."

            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count()),
            )

            if results and results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                output = []
                for i, doc in enumerate(docs, 1):
                    output.append(f"Experience {i}:\n{doc}")
                return "\n\n".join(output)

        except Exception as e:
            logger.warning(f"ChromaDB retrieval failed: {e}")

        # Fallback: return most recent in-memory experiences (current session only)
        if not self.experiences:
            return "No relevant experiences found yet."
        recent = self.experiences[-top_k:]
        output = []
        for i, exp in enumerate(recent, 1):
            output.append(f"Experience {i} (recent):\n{exp}")
        return "\n\n".join(output)

    def clear_collection(self):
        """Clear all stored experiences (for completely fresh start)."""
        self.experiences = []
        try:
            self._init_chroma()
            self._chroma_client.delete_collection(config.CHROMA_COLLECTION)
            self._collection = None
            self._chroma_client = None
            logger.info("Experience library cleared")
        except Exception as e:
            logger.warning(f"Failed to clear ChromaDB: {e}")

    def num_experiences(self) -> int:
        """Number of stored experiences (total across all epochs in ChromaDB)."""
        try:
            self._init_chroma()
            return self._collection.count()
        except Exception:
            return len(self.experiences)
