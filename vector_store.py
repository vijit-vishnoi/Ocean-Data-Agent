import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
from exceptions import VectorSearchError
from logger import get_logger

logger = get_logger(__name__)

_EMBED_MODEL = None
_FAISS_INDEX = None
_ID_MAP = None

def get_embedder(model_name: str):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Lazy loading SentenceTransformer model: {model_name}")
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL

def get_vector_index(index_path: str, id_map_path: str):
    global _FAISS_INDEX, _ID_MAP
    if _FAISS_INDEX is None or _ID_MAP is None:
        logger.info(f"Lazy loading FAISS index from {index_path}")
        _FAISS_INDEX = faiss.read_index(index_path)
        _ID_MAP = np.load(id_map_path, allow_pickle=True).tolist()
    return _FAISS_INDEX, _ID_MAP

class VectorStore:
    """Manages the FAISS vector index and embedding model for context retrieval."""

    def __init__(self, index_path: str = "data/argo_faiss.index", id_map_path: str = "data/id_map.npy", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the VectorStore with paths, but defers loading until the first query.
        """
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.model_name = model_name

    def retrieve_relevant_context(self, user_query: str, top_k: int = 20) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieves the top-k most relevant contexts for a given query.
        """
        try:
            embed_model = get_embedder(self.model_name)
            index, id_map = get_vector_index(self.index_path, self.id_map_path)
            
            q_emb = embed_model.encode([user_query], convert_to_numpy=True)
            D, I = index.search(q_emb, top_k)
            
            retrieved = []
            ctx_lines = []
            
            for idx in I[0]:
                if idx < 0 or idx >= len(id_map):
                    continue
                entry = id_map[idx]
                retrieved.append(entry)
                s = entry.get("summary") if isinstance(entry, dict) else str(entry)
                src = entry.get("source", "") if isinstance(entry, dict) else ""
                ctx_lines.append(f"[{src}] {s}")
                
            context_text = "\n".join(ctx_lines)
            logger.info(f"Retrieved {len(retrieved)} relevant contexts for query.")
            return context_text, retrieved
        except Exception as e:
            logger.error(f"Vector search failed for query '{user_query}': {e}")
            raise VectorSearchError(f"Failed to retrieve context: {e}")
