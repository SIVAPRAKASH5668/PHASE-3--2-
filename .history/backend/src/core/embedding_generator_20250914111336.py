from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Union, Dict, Optional, Any
import torch
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Advanced multilingual embedding generation with caching and batch processing"""
    
    def __init__(self, model_name: str ="paraphrase-multilingual-MiniLM-L12-v2"):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        """
        Initialize embedding generator with multilingual support
        
        Args:
            model_name: Name of the sentence transformer model
        """
        try:
            # Initialize model with optimization
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("🚀 Using GPU for embeddings")
            else:
                logger.info("💻 Using CPU for embeddings")
            
            # Performance optimizations
            self.model.eval()  # Set to evaluation mode
            
            # Cache for frequently used embeddings
            self._embedding_cache = {}
            self._cache_size_limit = 1000
            
            logger.info(f"✅ Loaded embedding model: {model_name}")
            logger.info(f"📏 Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    @lru_cache(maxsize=500)
    def _cached_encode(self, text: str) -> np.ndarray:
        """Cache frequently used embeddings"""
        return self.model.encode(text, convert_to_numpy=True)
    
    async def generate_embedding_async(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Async wrapper for embedding generation
        
        Args:
            text: Single text string or list of strings
            
        Returns:
            numpy array or list of numpy arrays
        """
        return await asyncio.to_thread(self.generate_embedding, text)
    
    def generate_embedding(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s) with enhanced error handling
        """
        try:
            if isinstance(text, str):
                if not text.strip():
                    return np.zeros(self.embedding_dimension, dtype=np.float32)
                
                # Check cache first
                text_key = text.strip()[:200]  # Limit key size
                if text_key in self._embedding_cache:
                    return self._embedding_cache[text_key]
                
                # Generate embedding
                embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                
                # ✅ ENSURE RESULT IS VALID
                if embedding is None or embedding.size == 0:
                    logger.warning("⚠️ Model returned empty embedding, using zero vector")
                    embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                
                # Cache if within limits
                if len(self._embedding_cache) < self._cache_size_limit:
                    self._embedding_cache[text_key] = embedding
                
                return embedding.astype(np.float32)
            
            elif isinstance(text, list):
                if not text:
                    return []
                
                # Filter out empty strings and prepare valid texts
                valid_texts = []
                text_mapping = []  # Track original positions
                
                for i, t in enumerate(text):
                    if isinstance(t, str) and t.strip():
                        valid_texts.append(t.strip())
                        text_mapping.append(i)
                
                if not valid_texts:
                    return [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in text]
                
                # Generate embeddings for valid texts
                embeddings = self.model.encode(valid_texts, convert_to_numpy=True, 
                                            normalize_embeddings=True, batch_size=32)
                
                # ✅ HANDLE SINGLE EMBEDDING RESULT
                if embeddings.ndim == 1:
                    # Single result case
                    embeddings = embeddings.reshape(1, -1)
                
                # Map back to original positions
                result = []
                valid_idx = 0
                for i, t in enumerate(text):
                    if isinstance(t, str) and t.strip():
                        result.append(embeddings[valid_idx].astype(np.float32))
                        valid_idx += 1
                    else:
                        result.append(np.zeros(self.embedding_dimension, dtype=np.float32))
                
                return result
            
            else:
                raise ValueError("Text must be string or list of strings")
                
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            if isinstance(text, str):
                return np.zeros(self.embedding_dimension, dtype=np.float32)
            else:
                return [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in text]
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        try:
            # Handle different input types
            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)
            
            # Check for zero vectors
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity efficiently
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            
            # Ensure result is in valid range
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         embeddings: List[np.ndarray], 
                         threshold: float = 0.5,
                         top_k: int = None) -> List[Dict]:
        """
        Find most similar embeddings to query with enhanced performance
        
        Args:
            query_embedding: Query embedding
            embeddings: List of embeddings to compare against
            threshold: Minimum similarity threshold
            top_k: Return top K results (None for all above threshold)
            
        Returns:
            List of similarity results sorted by score
        """
        try:
            if not embeddings:
                return []
            
            # Convert to numpy array for efficient computation
            query_emb = np.array(query_embedding, dtype=np.float32)
            embedding_matrix = np.array(embeddings, dtype=np.float32)
            
            # Batch similarity calculation
            similarities = np.dot(embedding_matrix, query_emb) / (
                np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(query_emb)
            )
            
            # Find indices above threshold
            above_threshold = similarities >= threshold
            valid_indices = np.where(above_threshold)[0]
            valid_similarities = similarities[above_threshold]
            
            # Create results
            results = [
                {
                    'index': int(idx),
                    'similarity': float(sim)
                }
                for idx, sim in zip(valid_indices, valid_similarities)
            ]
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Apply top_k limit if specified
            if top_k:
                results = results[:top_k]
            
            logger.info(f"✅ Found {len(results)} similar embeddings above threshold {threshold}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []
    
    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings in batches for optimal performance
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings
        """
        try:
            if not texts:
                return []
            
            # Filter valid texts
            valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
            if not valid_texts:
                return [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in texts]
            
            # Process in batches
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts, 
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=min(batch_size, len(batch_texts))
                    )
                    
                    # Handle single vs batch results
                    if batch_embeddings.ndim == 1:
                        all_embeddings.append(batch_embeddings.astype(np.float32))
                    else:
                        all_embeddings.extend([emb.astype(np.float32) for emb in batch_embeddings])
                        
                except Exception as e:
                    logger.error(f"❌ Batch {i//batch_size + 1} failed: {e}")
                    # Add zero embeddings for failed batch
                    for _ in batch_texts:
                        all_embeddings.append(np.zeros(self.embedding_dimension, dtype=np.float32))
            
            logger.info(f"✅ Generated {len(all_embeddings)} embeddings in {len(texts)//batch_size + 1} batches")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ Batch embedding generation failed: {e}")
            return [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in texts]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generator statistics"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "cache_size": len(self._embedding_cache),
            "cache_limit": self._cache_size_limit,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        self._embedding_cache.clear()
        logger.info("🧹 Embedding cache cleared")
