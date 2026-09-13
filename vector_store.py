import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any
from exceptions import VectorSearchError
from logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    """Manages the FAISS vector index and embedding model for context retrieval."""

    def __init__(self, index_path: str = "data/argo_faiss.index", id_map_path: str = "data/id_map.npy", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the VectorStore by loading the FAISS index, ID mapping, and embedding model.

        Args:
            index_path (str): Path to the FAISS index file.
            id_map_path (str): Path to the numpy array containing ID mapping.
            model_name (str): The name of the sentence-transformers model.
        """
        try:
            self.index = faiss.read_index(index_path)
            self.id_map = np.load(id_map_path, allow_pickle=True).tolist()
            self.embed_model = SentenceTransformer(model_name)
            logger.info(f"Loaded FAISS index from {index_path} and model {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            raise VectorSearchError(f"VectorStore initialization failed: {e}")

    def retrieve_relevant_context(self, user_query: str, top_k: int = 20) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieves the top-k most relevant contexts for a given query.

        Args:
            user_query (str): The user's input query.
            top_k (int): The number of relevant documents to retrieve.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: A formatted string of context lines, and the raw list of retrieved dictionaries.
        
        Raises:
            VectorSearchError: If the FAISS search fails.
        """
        try:
            q_emb = self.embed_model.encode([user_query], convert_to_numpy=True)
            D, I = self.index.search(q_emb, top_k)
            
            retrieved = []
            ctx_lines = []
            
            for idx in I[0]:
                if idx < 0 or idx >= len(self.id_map):
                    continue
                entry = self.id_map[idx]
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
