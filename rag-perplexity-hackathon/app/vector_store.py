import os
import uuid
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

# Resolve a stable on-disk path for Chroma persistence (./db next to this folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DB_DIR = os.path.join(BASE_DIR, "db")
os.makedirs(DB_DIR, exist_ok=True)


class VectorStore:
    """Wrapper around a persistent ChromaDB collection.

    - Uses SentenceTransformer "all-MiniLM-L6-v2" for embeddings
    - Persists to ./db folder
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

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] | None = None) -> int:
        if not texts:
            return 0
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length")

        ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        return len(texts)

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        result = self.collection.query(query_texts=[text], n_results=top_k)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        out: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"text": doc, "metadata": meta, "distance": float(dist)})
        return out

    def clear(self) -> None:
        """Delete and recreate the collection (clears all vectors)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # If it doesn't exist or other benign errors, ignore
            pass
        self._ensure_collection()
