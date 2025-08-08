from typing import List
import requests

from .config import get_perplexity_api_key
from .document_loader import load_text
from .utils import chunk_text
from .vector_store import VectorStore


class RAGPipeline:
    """Encapsulates the RAG flow: ingest -> embed/store -> retrieve -> generate."""

    def __init__(self) -> None:
        self.vs = VectorStore()

    def ingest_file(self, file_bytes: bytes, filename: str) -> int:
        """Extract text, split into chunks, and store in the vector DB.

        Returns number of chunks stored.
        """
        text = load_text(file_bytes, filename)
        chunks = chunk_text(text, max_len=500, overlap=50)
        metadatas = [
            {"source": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
        return self.vs.add_texts(chunks, metadatas)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        results = self.vs.query(query, top_k=top_k)
        return [r["text"] for r in results]

    @staticmethod
    def build_prompt(retrieved_chunks: List[str], user_query: str) -> str:
        context = "\n---\n".join(retrieved_chunks)
        return (
            "You are an AI assistant. Use ONLY the following context to answer:\n"
            f"{context}\n\n"
            f"Question: {user_query}"
        )

    @staticmethod
    def call_perplexity(final_prompt: str) -> str:
        api_key = get_perplexity_api_key()
        if not api_key:
            raise RuntimeError(
                "PERPLEXITY_API_KEY is not set. Please configure it in your .env file."
            )

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.1-sonar-large-32k-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": final_prompt},
            ],
            "max_tokens": 500,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Perplexity API error {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            # Fallback if the schema differs
            return str(data)

    def query(self, user_query: str) -> str:
        chunks = self.retrieve(user_query, top_k=3)
        final_prompt = self.build_prompt(chunks, user_query)
        return self.call_perplexity(final_prompt)
