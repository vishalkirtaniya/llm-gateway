import asyncio
import hashlib
import json
from typing import List
import numpy as np

from ..config import settings
from .llm_client import LLMError, embed


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


# scale vectors to length 1 (works for one vector or a matrix)
def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)  # length of each vector
    norms = np.where(norms == 0, 1.0, norms)  # avoid dividing by zero
    return matrix / norms


def _cache_key(text: str) -> str:
    # model name is part of the key
    return hashlib.sha256(
        f"{settings.EMBEDDING_MODEL} | {text}".encode("utf=8")
    ).hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(settings.EMBED_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # if file is missing then start with empty cache
        return {}


def _save_cache(cache: dict) -> None:
    settings.EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.EMBED_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


class KnowledgeBase:
    def __init__(self):
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.matrix = None
        self._ready = False
        self._lock = asyncio.Lock()

    # convert text into chunks and then into matrix, using the cache
    async def _embed_documents(self, chunks: List[str]) -> np.ndarray:
        cache = _load_cache()
        missing = list(dict.fromkeys(c for c in chunks if _cache_key(c) not in cache))
        for i in range(0, len(missing), settings.EMBED_BATCH_SIZE):
            batch = missing[i : 1 + settings.EMBED_BATCH_SIZE]
            vectors = await embed([settings.EMBED_DOC_PREFIX + c for c in batch])
            if len(vectors) != len(batch):
                raise LLMError(
                    502, f"Expected {len(batch)} embeddings, got {len(vectors)}"
                )
            for text, vector in zip(batch, vectors):
                cache[_cache_key(text)] = vector
        if missing:
            _save_cache(cache)
        return np.array([cache[_cache_key(c)] for c in chunks], dtype=np.float32)

    async def _build(self) -> None:
        self.chunks, self.sources = [], []
        for path in sorted(settings.KNOWLEDGE_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            for chunk in chunk_text(text, settings.CHUNK_WORDS, settings.CHUNK_OVERLAP):
                if chunk.strip():
                    self.chunks.append(chunk)
                    self.sources.append(path.name)
        if self.chunks:
            self.matrix = normalize(await self._embed_documents(self.chunks))
        self._ready = True

    async def search(self, query: str, top_k: int = None) -> List[dict]:
        top_k = top_k or settings.TOP_K
        async with self._lock:
            if not self._ready:
                await self._build()
        if self.matrix is None:
            return []
        vector = await embed([settings.EMBED_QUERY_PREFIX + query])
        query_vec = normalize(np.array(vector[0], dtype=np.float32))
        scores = (
            self.matrix @ query_vec
        )  # cosine similarity against every chunk at once
        best = np.argsort(scores)[::-1][
            :top_k
        ]  # indices of the highest scores, best first
        return [
            {
                "text": self.chunks[i],
                "source": self.sources[i],
                "score": round(float(scores[i]), 4),
            }
            for i in best
        ]
