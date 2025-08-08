import uuid
from typing import List, Tuple
import requests

from .config import (
    get_perplexity_api_key,
    get_model_name,
    get_qa_confidence_threshold,
    get_system_prompt,
)
from .document_loader import load_text
from .utils import chunk_text
from .vector_store import VectorStore
from .registry import DocumentRegistry
from .qa_parser import parse_qa_pairs


class RAGPipeline:
    """Encapsulates the RAG flow: ingest -> embed/store -> retrieve -> generate."""

    def __init__(self, registry: DocumentRegistry | None = None) -> None:
        self.vs = VectorStore()
        self.registry = registry or DocumentRegistry()

    def ingest_file(self, file_bytes: bytes, filename: str) -> tuple[str, int]:
        """Extract text, split into chunks, and store in the vector DB.

        Returns (doc_id, number_of_chunks).
        """
        text = load_text(file_bytes, filename)
        chunks = chunk_text(text, max_len=500, overlap=50)
        doc_id = str(uuid.uuid4())
        metadatas = [
            {"source": filename, "chunk_index": i, "type": "doc", "doc_id": doc_id}
            for i in range(len(chunks))
        ]
        ids = self.vs.add_texts(chunks, metadatas)
        self.registry.register(doc_id, "doc", filename, len(ids))
        return doc_id, len(ids)

    def ingest_qa_text(self, file_bytes: bytes, filename: str) -> tuple[str, int]:
        """Parse Q&A pairs and store them with rich metadata.

        Each vector embeds the QUESTION text only; metadata contains the answer.
        Returns (doc_id, number_of_pairs).
        """
        text = load_text(file_bytes, filename)
        pairs = parse_qa_pairs(text)
        if not pairs:
            raise ValueError("No Q&A pairs found in uploaded document.")
        doc_id = str(uuid.uuid4())
        questions = [p["question"].strip() for p in pairs]
        metadatas = [
            {
                "type": "qa",
                "doc_id": doc_id,
                "source": filename,
                "pair_index": i,
                "question": p["question"].strip(),
                "answer": p["answer"].strip(),
            }
            for i, p in enumerate(pairs)
        ]
        ids = self.vs.add_texts(questions, metadatas)
        self.registry.register(doc_id, "qa", filename, len(ids))
        return doc_id, len(ids)

    def retrieve(self, query: str, top_k: int = 8):
        return self.vs.query(query, top_k=top_k)

    @staticmethod
    def build_prompt(doc_chunks: List[str], qa_pairs: List[Tuple[str, str]], user_query: str) -> str:
        doc_context = "\n---\n".join(doc_chunks) if doc_chunks else ""
        qa_context = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs]) if qa_pairs else ""
        parts = []
        if qa_context:
            parts.append("Approved Q&A:\n" + qa_context)
        if doc_context:
            parts.append("Policy excerpts:\n" + doc_context)
        context = "\n\n".join(parts)
        return (
            "Use ONLY the following context to answer. If not fully answered by the context, "
            "reply exactly with: Not in policy\n\n"
            f"{context}\n\nQuestion: {user_query}"
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
            "model": get_model_name(),
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": final_prompt},
            ],
            "max_tokens": 500,
            "temperature": 0.1,
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
        results = self.retrieve(user_query, top_k=8)
        qa_hits = [r for r in results if (r.get("metadata") or {}).get("type") == "qa"]
        qa_hits.sort(key=lambda r: (r.get("similarity") or 0.0), reverse=True)
        doc_hits = [r for r in results if (r.get("metadata") or {}).get("type") == "doc"]

        threshold = get_qa_confidence_threshold()
        if qa_hits and (qa_hits[0].get("similarity") or 0.0) >= threshold:
            meta = qa_hits[0].get("metadata") or {}
            answer = meta.get("answer")
            if answer:
                return answer

        doc_contexts = [r.get("text", "") for r in doc_hits[:3] if r.get("text")]
        qa_contexts = []
        for r in qa_hits[:3]:
            m = r.get("metadata") or {}
            q = m.get("question", "")
            a = m.get("answer", "")
            if q and a:
                qa_contexts.append((q, a))

        final_prompt = self.build_prompt(doc_contexts, qa_contexts, user_query)
        return self.call_perplexity(final_prompt)
