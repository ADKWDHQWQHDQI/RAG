"""
Embedding module using sentence-transformers
"""
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
from config import Config


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = None
        print(f"[LOADING] Loading embedding model: {self.model_name}")
    
    def load(self):
        """Load the embedding model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"[SUCCESS] Embedding model loaded: {self.model_name}")
            print(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading embedding model: {e}")
            return False
    
    def encode(self, text: Union[str, List[str]], show_progress: bool = False) -> Optional[Union[List[float], List[List[float]]]]:
        """
        Encode text into embeddings
        
        Args:
            text: Single text string or list of texts
            show_progress: Show progress bar for batch encoding
            
        Returns:
            Embedding vector(s) or None if error
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            embeddings = self.model.encode(
                text,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Convert to list for database storage
            if isinstance(text, str):
                # Single text - embeddings is a numpy array
                return embeddings.tolist()  # type: ignore
            else:
                # Multiple texts - embeddings is array of arrays
                return [emb.tolist() for emb in embeddings]  # type: ignore
        except Exception as e:
            print(f"[ERROR] Error encoding text: {e}")
            return None
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> Optional[List[List[float]]]:
        """
        Encode multiple texts in batches
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to encode at once
            
        Returns:
            List of embedding vectors or None if error
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return [emb.tolist() for emb in embeddings]  # type: ignore
        except Exception as e:
            print(f"[ERROR] Error encoding batch: {e}")
            return None
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("Failed to retrieve embedding dimension.")
        return int(dim)  # type: ignore
    
    def __enter__(self):
        """Context manager entry"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Clean up if needed
        pass
