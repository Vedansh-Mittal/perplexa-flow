import json
import os
from datetime import datetime
from typing import Dict, Any, List

from .config import get_chroma_dir


class DocumentRegistry:
    """Simple JSON-backed registry for uploaded items (docs and QA sets).

    Each record:
    - doc_id: str
    - type: 'doc' | 'qa'
    - filename: str
    - count: int  (number of vectors stored)
    - created_at: ISO timestamp
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.path.join(get_chroma_dir(), "registry.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._save({})

    def _load(self) -> Dict[str, Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: Dict[str, Dict[str, Any]]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register(self, doc_id: str, doc_type: str, filename: str, count: int) -> None:
        data = self._load()
        data[doc_id] = {
            "doc_id": doc_id,
            "type": doc_type,
            "filename": filename,
            "count": int(count),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        self._save(data)

    def delete(self, doc_id: str) -> None:
        data = self._load()
        if doc_id in data:
            del data[doc_id]
            self._save(data)

    def list(self) -> List[Dict[str, Any]]:
        data = self._load()
        # Return newest first
        return sorted(data.values(), key=lambda r: r.get("created_at", ""), reverse=True)
