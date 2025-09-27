from __future__ import annotations
from typing import Any, Dict, List
from chromadb.config import Settings
import chromadb

class VectorMemory:
    def __init__(self, persist_dir: str, embed_fn):
        self._embed = embed_fn
        self.client = chromadb.Client(Settings(is_persistent=True, persist_directory=persist_dir))
        self.col = self.client.get_or_create_collection(name="babyagi_memory")

    def add(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        v = self._embed(text)
        self.col.add(documents=[text], embeddings=[v], ids=[doc_id], metadatas=[metadata])

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        v = self._embed(query)
        res = self.col.query(query_embeddings=[v], n_results=k)
        out: List[Dict[str, Any]] = []
        if not res["ids"]:
            return out
        for i in range(len(res["ids"][0])):
            out.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                "dist": res.get("distances", [[None]])[0][i],
            })
        return out
