from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Union, Dict, Optional, Any
import torch
import asyncio
from functools import lru_cache
import traceback
import json

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Advanced multilingual embedding generation with comprehensive debugging"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Initialize embedding generator with debug logging"""
        logger.info(f"🚀 Initializing EmbeddingGenerator with model: {model_name}")
        
        try:
            # Initialize model with optimization
            logger.debug("📥 Loading SentenceTransformer model...")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"📏 Model loaded - Embedding dimension: {self.embedding_dimension}")
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 Using GPU: {gpu_name}")
            else:
                logger.info("💻 Using CPU for embeddings")
            
            # Performance optimizations
            self.model.eval()  # Set to evaluation mode
            
            # Cache for frequently used embeddings
            self._embedding_cache = {}
            self._cache_size_limit = 1000
            
            # Statistics tracking
            self.stats = {
                "embeddings_generated": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "failed_embeddings": 0,
                "zero_vectors_returned": 0
            }
            
            logger.info(f"✅ EmbeddingGenerator initialized successfully")
            logger.info(f"🔧 Cache limit: {self._cache_size_limit}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize EmbeddingGenerator: {e}")
            logger.error(f"📜 Full traceback: {traceback.format_exc()}")
            raise

    async def generate_embedding_async(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Async wrapper for embedding generation with debug"""
        logger.debug(f"🔄 Async embedding generation for: {type(text)}")
        return await asyncio.to_thread(self.generate_embedding, text)

    def generate_embedding(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings with comprehensive debugging"""
        try:
            logger.debug(f"🎯 Starting embedding generation for input type: {type(text)}")
            
            if isinstance(text, str):
                return self._generate_single_embedding(text)
            elif isinstance(text, list):
                return self._generate_batch_embeddings(text)
            else:
                raise ValueError(f"Invalid input type: {type(text)}. Expected str or List[str]")
                
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            logger.error(f"📜 Full traceback: {traceback.format_exc()}")
            self.stats["failed_embeddings"] += 1
            
            if isinstance(text, str):
                return np.zeros(self.embedding_dimension, dtype=np.float32)
            else:
                return [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in text]

    def _generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate single embedding with detailed logging"""
        logger.debug(f"📝 Processing single text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        if not text.strip():
            logger.warning("⚠️ Empty or whitespace-only text provided - returning zero vector")
            self.stats["zero_vectors_returned"] += 1
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        # Check cache first
        text_key = text.strip()[:500]  # Limit key size for memory efficiency
        if text_key in self._embedding_cache:
            logger.debug("💾 Using cached embedding")
            self.stats["cache_hits"] += 1
            return self._embedding_cache[text_key]
        
        self.stats["cache_misses"] += 1
        
        try:
            # Generate embedding
            logger.debug("🧠 Generating new embedding...")
            embedding = self.model.encode(
                text, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Validate embedding
            if embedding is None or embedding.size == 0:
                logger.error("❌ Model returned empty embedding - using zero vector")
                self.stats["zero_vectors_returned"] += 1
                embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
            else:
                # Check embedding properties
                norm = np.linalg.norm(embedding)
                has_nan = np.any(np.isnan(embedding))
                has_inf = np.any(np.isinf(embedding))
                
                logger.debug(f"📊 Embedding stats - Shape: {embedding.shape}, Norm: {norm:.6f}, Has NaN: {has_nan}, Has Inf: {has_inf}")
                
                if has_nan or has_inf:
                    logger.error("❌ Embedding contains NaN or Inf values - using zero vector")
                    self.stats["zero_vectors_returned"] += 1
                    embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                elif norm < 1e-8:
                    logger.warning("⚠️ Embedding has very small norm - might be problematic")
            
            # Convert to float32 for consistency
            embedding = embedding.astype(np.float32)
            
            # Cache if within limits
            if len(self._embedding_cache) < self._cache_size_limit:
                self._embedding_cache[text_key] = embedding
                logger.debug(f"💾 Cached embedding (cache size: {len(self._embedding_cache)})")
            
            self.stats["embeddings_generated"] += 1
            logger.debug(f"✅ Successfully generated embedding with shape: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding for text: {e}")
            logger.error(f"📜 Embedding generation traceback: {traceback.format_exc()}")
            self.stats["failed_embeddings"] += 1
            self.stats["zero_vectors_returned"] += 1
            return np.zeros(self.embedding_dimension, dtype=np.float32)

    def _generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate batch embeddings with detailed logging"""
        logger.debug(f"📚 Processing batch of {len(texts)} texts")
        
        if not texts:
            logger.warning("⚠️ Empty text list provided")
            return []
        
        # Filter and prepare valid texts
        valid_texts = []
        text_mapping = []  # Track original positions
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                text_mapping.append(i)
            else:
                logger.debug(f"⚠️ Invalid text at index {i}: {type(text)} - '{text}'")
        
        logger.debug(f"📊 Batch stats - Total: {len(texts)}, Valid: {len(valid_texts)}")
        
        if not valid_texts:
            logger.warning("⚠️ No valid texts in batch - returning zero vectors")
            self.stats["zero_vectors_returned"] += len(texts)
            return [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in texts]
        
        try:
            # Generate embeddings for valid texts
            logger.debug("🧠 Generating batch embeddings...")
            embeddings = self.model.encode(
                valid_texts, 
                convert_to_numpy=True,
                normalize_embeddings=True, 
                batch_size=32,
                show_progress_bar=False
            )
            
            # Handle single result case
            if embeddings.ndim == 1:
                logger.debug("🔄 Single embedding result - reshaping")
                embeddings = embeddings.reshape(1, -1)
            
            logger.debug(f"📊 Generated embeddings shape: {embeddings.shape}")
            
            # Map back to original positions
            result = []
            valid_idx = 0
            
            for i, text in enumerate(texts):
                if isinstance(text, str) and text.strip():
                    embedding = embeddings[valid_idx].astype(np.float32)
                    
                    # Validate individual embedding
                    norm = np.linalg.norm(embedding)
                    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)) or norm < 1e-8:
                        logger.warning(f"⚠️ Invalid embedding at batch index {i} - using zero vector")
                        embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                        self.stats["zero_vectors_returned"] += 1
                    
                    result.append(embedding)
                    valid_idx += 1
                    self.stats["embeddings_generated"] += 1
                else:
                    logger.debug(f"⚠️ Adding zero vector for invalid text at index {i}")
                    result.append(np.zeros(self.embedding_dimension, dtype=np.float32))
                    self.stats["zero_vectors_returned"] += 1
            
            logger.debug(f"✅ Successfully generated {len(result)} batch embeddings")
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch embedding generation failed: {e}")
            logger.error(f"📜 Batch embedding traceback: {traceback.format_exc()}")
            self.stats["failed_embeddings"] += len(texts)
            self.stats["zero_vectors_returned"] += len(texts)
            return [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in texts]

    def get_tidb_compatible_embedding(self, text: str) -> Optional[str]:
        """Generate embedding in TiDB-compatible format with debugging"""
        logger.debug(f"🔧 Generating TiDB-compatible embedding for: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        try:
            embedding = self.generate_embedding(text)
            
            # Validate embedding
            if embedding is None or embedding.size == 0:
                logger.error("❌ No embedding generated for TiDB format")
                return None
            
            if np.all(embedding == 0):
                logger.warning("⚠️ Zero vector generated - TiDB format might not be useful")
                return None
            
            # Convert to TiDB vector format: '[0.1,0.2,0.3]'
            vector_str = '[' + ','.join(f'{float(x):.6f}' for x in embedding) + ']'
            
            # Validate format
            if len(vector_str) < 10:  # Minimum reasonable length
                logger.error(f"❌ Generated vector string too short: {len(vector_str)} characters")
                return None
            
            logger.debug(f"✅ Generated TiDB vector string - Length: {len(vector_str)}, Preview: {vector_str[:100]}...")
            logger.debug(f"📊 Vector stats - Min: {embedding.min():.6f}, Max: {embedding.max():.6f}, Mean: {embedding.mean():.6f}")
            
            return vector_str
            
        except Exception as e:
            logger.error(f"❌ Failed to generate TiDB-compatible embedding: {e}")
            logger.error(f"📜 TiDB embedding traceback: {traceback.format_exc()}")
            return None

    def validate_embedding(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Comprehensive embedding validation with detailed results"""
        validation = {
            'is_valid': False,
            'issues': [],
            'stats': {},
            'recommendations': []
        }
        
        try:
            if embedding is None:
                validation['issues'].append('Embedding is None')
                return validation
            
            if embedding.size == 0:
                validation['issues'].append('Embedding is empty')
                return validation
            
            # Check dimensions
            if len(embedding.shape) != 1:
                validation['issues'].append(f'Invalid shape: {embedding.shape} (expected 1D)')
            
            if embedding.shape[0] != self.embedding_dimension:
                validation['issues'].append(f'Dimension mismatch: {embedding.shape[0]} vs expected {self.embedding_dimension}')
            
            # Check for problematic values
            has_nan = np.any(np.isnan(embedding))
            has_inf = np.any(np.isinf(embedding))
            is_all_zeros = np.all(embedding == 0)
            
            if has_nan:
                validation['issues'].append('Contains NaN values')
            if has_inf:
                validation['issues'].append('Contains infinite values')
            if is_all_zeros:
                validation['issues'].append('All values are zero')
            
            # Calculate statistics
            norm = np.linalg.norm(embedding)
            validation['stats'] = {
                'norm': float(norm),
                'min': float(embedding.min()),
                'max': float(embedding.max()),
                'mean': float(embedding.mean()),
                'std': float(embedding.std()),
                'has_nan': has_nan,
                'has_inf': has_inf,
                'is_all_zeros': is_all_zeros
            }
            
            # Generate recommendations
            if norm < 1e-8:
                validation['recommendations'].append('Norm is very small - check input text quality')
            elif norm > 10:
                validation['recommendations'].append('Norm is very large - check normalization')
            
            if validation['stats']['std'] < 1e-6:
                validation['recommendations'].append('Very low variance - check embedding diversity')
            
            # Determine if valid
            validation['is_valid'] = len(validation['issues']) == 0
            
            logger.debug(f"🔍 Embedding validation - Valid: {validation['is_valid']}, Issues: {len(validation['issues'])}")
            
        except Exception as e:
            validation['issues'].append(f'Validation failed: {str(e)}')
            logger.error(f"❌ Embedding validation error: {e}")
        
        return validation

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity with debugging"""
        try:
            logger.debug("🔢 Calculating cosine similarity")
            
            # Convert to numpy arrays
            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)
            
            logger.debug(f"📊 Embedding shapes - 1: {emb1.shape}, 2: {emb2.shape}")
            
            # Check for zero vectors
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            logger.debug(f"📊 Embedding norms - 1: {norm1:.6f}, 2: {norm2:.6f}")
            
            if norm1 == 0 or norm2 == 0:
                logger.warning("⚠️ One or both embeddings have zero norm - similarity set to 0")
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            similarity = np.clip(similarity, -1.0, 1.0)  # Ensure valid range
            
            logger.debug(f"✅ Calculated similarity: {similarity:.6f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {e}")
            logger.error(f"📜 Similarity calculation traceback: {traceback.format_exc()}")
            return 0.0

    def debug_embedding_pipeline(self, sample_texts: List[str]) -> Dict[str, Any]:
        """Comprehensive debug of embedding pipeline"""
        logger.info("🔍 Starting embedding pipeline debug")
        
        debug_results = {
            'model_info': {
                'name': self.model_name,
                'dimension': self.embedding_dimension,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'cache_info': {
                'size': len(self._embedding_cache),
                'limit': self._cache_size_limit
            },
            'statistics': self.stats.copy(),
            'sample_results': []
        }
        
        try:
            for i, text in enumerate(sample_texts[:5]):  # Test first 5 samples
                logger.info(f"🧪 Testing sample {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                sample_result = {
                    'text': text[:100],
                    'text_length': len(text)
                }
                
                try:
                    # Generate embedding
                    embedding = self.generate_embedding(text)
                    sample_result['embedding_generated'] = True
                    sample_result['embedding_shape'] = embedding.shape
                    
                    # Validate embedding
                    validation = self.validate_embedding(embedding)
                    sample_result['validation'] = validation
                    
                    # Generate TiDB format
                    tidb_format = self.get_tidb_compatible_embedding(text)
                    sample_result['tidb_format_generated'] = tidb_format is not None
                    if tidb_format:
                        sample_result['tidb_format_length'] = len(tidb_format)
                        sample_result['tidb_preview'] = tidb_format[:100]
                    
                    logger.info(f"✅ Sample {i+1} processed successfully")
                    
                except Exception as e:
                    sample_result['error'] = str(e)
                    logger.error(f"❌ Sample {i+1} failed: {e}")
                
                debug_results['sample_results'].append(sample_result)
            
        except Exception as e:
            debug_results['debug_error'] = str(e)
            logger.error(f"❌ Debug pipeline failed: {e}")
        
        logger.info("🔍 Embedding pipeline debug completed")
        return debug_results

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get comprehensive embedding generator statistics"""
        return {
            "model_info": {
                "name": self.model_name,
                "dimension": self.embedding_dimension,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "cache_info": {
                "size": len(self._embedding_cache),
                "limit": self._cache_size_limit,
                "hit_rate": self.stats["cache_hits"] / max(self.stats["cache_hits"] + self.stats["cache_misses"], 1)
            },
            "performance_stats": self.stats,
            "health_indicators": {
                "zero_vector_rate": self.stats["zero_vectors_returned"] / max(self.stats["embeddings_generated"], 1),
                "failure_rate": self.stats["failed_embeddings"] / max(self.stats["embeddings_generated"], 1)
            }
        }

    def clear_cache(self):
        """Clear embedding cache with logging"""
        old_size = len(self._embedding_cache)
        self._embedding_cache.clear()
        logger.info(f"🧹 Cleared embedding cache (removed {old_size} entries)")

    def __del__(self):
        """Cleanup with final statistics"""
        try:
            if hasattr(self, 'stats'):
                logger.info(f"🏁 EmbeddingGenerator cleanup - Final stats: {self.stats}")
        except:
            pass
