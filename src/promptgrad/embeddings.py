"""
Embedding computation for prompt sensitivity analysis.

Supports:
  - Local sentence-transformers (default, no API key needed)
  - OpenAI embeddings (optional, requires `openai` extra)
  - A fast TF-IDF fallback that needs zero ML dependencies
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class EmbeddingEngine(ABC):
    """Abstract embedding engine."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (N, D) float32 array of embeddings."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def pairwise_cosine(self, embeddings: np.ndarray) -> np.ndarray:
        """Return an (N, N) cosine similarity matrix."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = embeddings / norms
        return (normed @ normed.T).astype(np.float32)

    def embedding_shift(self, original: np.ndarray, variants: np.ndarray) -> np.ndarray:
        """
        Measure how far each variant embedding is from the original.

        Returns a 1-D array of L2 distances (one per variant).
        """
        return np.linalg.norm(variants - original[np.newaxis, :], axis=1)


class TFIDFEmbeddingEngine(EmbeddingEngine):
    """
    Pure-Python TF-IDF embedding — zero ML dependencies.

    Good for fast smoke-testing.  Dimensions = vocabulary size (capped).
    """

    def __init__(self, max_features: int = 512):
        self._max_features = max_features
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "tfidf"

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_vocab(self, corpus: list[str]) -> None:
        from collections import Counter
        doc_freq: Counter = Counter()
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            doc_freq.update(tokens)
        # keep top-N by document frequency
        top = [t for t, _ in doc_freq.most_common(self._max_features)]
        self._vocab = {t: i for i, t in enumerate(top)}
        n = len(corpus)
        self._idf = np.array(
            [math.log((n + 1) / (doc_freq.get(t, 0) + 1)) + 1.0 for t in top],
            dtype=np.float32,
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        self._build_vocab(texts)
        V = len(self._vocab)
        out = np.zeros((len(texts), V), dtype=np.float32)
        for i, text in enumerate(texts):
            from collections import Counter
            tf = Counter(self._tokenize(text))
            for token, count in tf.items():
                if token in self._vocab:
                    j = self._vocab[token]
                    out[i, j] = count * self._idf[j]  # type: ignore[index]
            # L2-normalise
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class SentenceTransformerEngine(EmbeddingEngine):
    """
    Local sentence-transformers engine (requires `sentence-transformers`).

    Install via:  pip install promptgrad[local]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformerEngine.\n"
                    "Install it with:  pip install promptgrad[local]"
                ) from e
            self._model = SentenceTransformer(self._model_name)

    @property
    def name(self) -> str:
        return f"sentence_transformers:{self._model_name}"

    def embed(self, texts: List[str]) -> np.ndarray:
        self._load()
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class OpenAIEmbeddingEngine(EmbeddingEngine):
    """
    OpenAI embedding engine (requires `openai` package).

    Install via:  pip install promptgrad[openai]
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self._model = model
        self._api_key = api_key

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def embed(self, texts: List[str]) -> np.ndarray:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIEmbeddingEngine.\n"
                "Install it with:  pip install promptgrad[openai]"
            ) from e
        client = OpenAI(api_key=self._api_key)
        response = client.embeddings.create(input=texts, model=self._model)
        return np.array(
            [item.embedding for item in response.data], dtype=np.float32
        )


def get_engine(backend: str = "auto", **kwargs) -> EmbeddingEngine:
    """
    Factory helper.

    backend: "auto" | "tfidf" | "sentence_transformers" | "openai"

    "auto" tries sentence_transformers, falls back to tfidf.
    """
    if backend == "tfidf":
        return TFIDFEmbeddingEngine(**kwargs)
    if backend == "sentence_transformers":
        return SentenceTransformerEngine(**kwargs)
    if backend == "openai":
        return OpenAIEmbeddingEngine(**kwargs)
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401
            return SentenceTransformerEngine(**kwargs)
        except ImportError:
            return TFIDFEmbeddingEngine(**kwargs)
    raise ValueError(f"Unknown embedding backend: {backend!r}")
