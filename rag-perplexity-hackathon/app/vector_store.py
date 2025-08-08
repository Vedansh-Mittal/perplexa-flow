import os
import uuid
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

from .config import get_chroma_dir

# Resolve a stable on-disk path for Chroma persistence
DB_DIR = get_chroma_dir()
os.makedirs(DB_DIR, exist_ok=True)


class VectorStore:
    """Wrapper around a persistent ChromaDB collection.

    - Uses SentenceTransformer "all-MiniLM-L6-v2" for embeddings
    - Persists to configured ./db folder
    """

    def __init__(self, collection_name: str = "documents") -> None:
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] | None = None,
    ) -> List[str]:
        if not texts:
            return []
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length")

        ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        return ids

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        result = self.collection.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"],
        )
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        out: List[Dict[str, Any]] = []
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            try:
                similarity = 1.0 - float(dist) if dist is not None else None
            except Exception:
                similarity = None
            out.append(
                {
                    "id": _id,
                    "text": doc,
                    "metadata": meta or {},
                    "distance": float(dist) if dist is not None else None,
                    "similarity": similarity,
                }
            )
        return out

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all vectors that belong to a specific document id."""
        self.collection.delete(where={"doc_id": doc_id})

    def clear(self) -> None:
        """Delete and recreate the collection (clears all vectors)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # If it doesn't exist or other benign errors, ignore
            pass
        self._ensure_collection()
