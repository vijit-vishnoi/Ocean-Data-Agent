import numpy as np
import requests
import os
from typing import List, Tuple, Dict, Any
from exceptions import VectorSearchError
from logger import get_logger

logger = get_logger(__name__)

_VECTORS = None
_ID_MAP = None

def get_huggingface_embedding(query: str, model_name: str) -> np.ndarray:
    """Gets an embedding from the Hugging Face Serverless Inference API."""
    api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("Missing HUGGINGFACE_API_KEY or HF_TOKEN environment variable.")
    
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.post(url, headers=headers, json={"inputs": [query]})
    if response.status_code != 200:
        raise VectorSearchError(f"Hugging Face API error: {response.text}")
        
    embeddings = response.json()
    return np.array(embeddings[0], dtype=np.float32)

def get_vector_index(vectors_path: str, id_map_path: str):
    global _VECTORS, _ID_MAP
    if _VECTORS is None or _ID_MAP is None:
        logger.info(f"Lazy loading vectors from {vectors_path}")
        _VECTORS = np.load(vectors_path)
        _ID_MAP = np.load(id_map_path, allow_pickle=True).tolist()
    return _VECTORS, _ID_MAP

class VectorStore:
    """Manages the numpy vector store and HF inference API for context retrieval."""

    def __init__(self, vectors_path: str = "data/argo_vectors.npy", id_map_path: str = "data/id_map.npy", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the VectorStore with paths, but defers loading until the first query.
        """
        self.vectors_path = vectors_path
        self.id_map_path = id_map_path
        self.model_name = model_name

    def retrieve_relevant_context(self, user_query: str, top_k: int = 20) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieves the top-k most relevant contexts for a given query.
        """
        try:
            vectors, id_map = get_vector_index(self.vectors_path, self.id_map_path)
            
            # Fetch embedding from Hugging Face Inference API
            q_emb = get_huggingface_embedding(user_query, self.model_name)
            
            # Calculate L2 distances using numpy
            distances = np.linalg.norm(vectors - q_emb, axis=1)
            top_indices = np.argsort(distances)[:top_k]
            
            retrieved = []
            ctx_lines = []
            
            for idx in top_indices:
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
