import math
import re
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
from exceptions import VectorSearchError
from logger import get_logger

logger = get_logger(__name__)

_ID_MAP = None
_DOC_TOKENS: List[List[str]] = []
_IDF: Dict[str, float] = {}
_DOC_TFIDF: List[Dict[str, float]] = []
_DOC_NORMS: List[float] = []


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _build_tfidf_index(id_map: List[Any]) -> None:
    global _DOC_TOKENS, _IDF, _DOC_TFIDF, _DOC_NORMS
    if _DOC_TFIDF:
        return

    docs_tokens: List[List[str]] = []
    df: Counter = Counter()

    for entry in id_map:
        if isinstance(entry, dict):
            text = f"{entry.get('source', '')} {entry.get('summary', '')}"
        else:
            text = str(entry)
        tokens = _tokenize(text)
        docs_tokens.append(tokens)
        for term in set(tokens):
            df[term] += 1

    num_docs = max(len(docs_tokens), 1)
    idf: Dict[str, float] = {
        term: math.log((1 + num_docs) / (1 + count)) + 1.0
        for term, count in df.items()
    }

    doc_tfidf: List[Dict[str, float]] = []
    doc_norms: List[float] = []

    for tokens in docs_tokens:
        if not tokens:
            doc_tfidf.append({})
            doc_norms.append(0.0)
            continue
        tf = Counter(tokens)
        total = len(tokens)
        weights = {term: (cnt / total) * idf[term] for term, cnt in tf.items()}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        doc_tfidf.append(weights)
        doc_norms.append(norm)

    _DOC_TOKENS = docs_tokens
    _IDF = idf
    _DOC_TFIDF = doc_tfidf
    _DOC_NORMS = doc_norms


def get_vector_index(vectors_path: str, id_map_path: str):
    global _ID_MAP
    if _ID_MAP is None:
        logger.info(f"Lazy loading id_map from {id_map_path}")
        _ID_MAP = np.load(id_map_path, allow_pickle=True).tolist()
        _build_tfidf_index(_ID_MAP)
    return _ID_MAP


class VectorStore:
    """Lightweight local retrieval over id_map without external embedding APIs."""

    def __init__(
        self,
        vectors_path: str = "data/argo_vectors.npy",
        id_map_path: str = "data/id_map.npy",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.vectors_path = vectors_path
        self.id_map_path = id_map_path
        self.model_name = model_name

    def retrieve_relevant_context(
        self, user_query: str, top_k: int = 20
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Retrieves the top-k most relevant context entries for a given query."""
        try:
            id_map = get_vector_index(self.vectors_path, self.id_map_path)

            q_tokens = _tokenize(user_query)
            q_counts = Counter(q_tokens)
            q_total = max(len(q_tokens), 1)
            q_weights = {
                term: (cnt / q_total) * _IDF[term]
                for term, cnt in q_counts.items()
                if term in _IDF
            }
            q_norm = math.sqrt(sum(w * w for w in q_weights.values()))

            scores = np.zeros(len(id_map), dtype=np.float32)
            if q_norm > 0:
                for idx, doc_weights in enumerate(_DOC_TFIDF):
                    denom = q_norm * _DOC_NORMS[idx]
                    if denom == 0 or not doc_weights:
                        continue
                    dot = sum(
                        q_val * doc_weights[term]
                        for term, q_val in q_weights.items()
                        if term in doc_weights
                    )
                    scores[idx] = dot / denom

            k = min(top_k, len(id_map))
            top_indices = np.argsort(-scores)[:k]

            retrieved: List[Dict[str, Any]] = []
            ctx_lines: List[str] = []

            for idx in top_indices:
                entry = id_map[idx]
                if isinstance(entry, dict):
                    retrieved.append(entry)
                    s = entry.get("summary", "")
                    src = entry.get("source", "")
                else:
                    retrieved.append({"summary": str(entry), "source": ""})
                    s = str(entry)
                    src = ""
                ctx_lines.append(f"[{src}] {s}")

            context_text = "\n".join(ctx_lines)
            logger.info(f"Retrieved {len(retrieved)} relevant contexts via TF-IDF.")
            return context_text, retrieved
        except Exception as e:
            logger.error(f"Context retrieval failed for query '{user_query}': {e}")
            raise VectorSearchError(f"Failed to retrieve context: {e}")