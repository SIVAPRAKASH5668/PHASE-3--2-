import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from sqlalchemy import text

logger = logging.getLogger(__name__)

class VectorOperations:
    """
    ✅ FIXED: Advanced vector operations with TiDB vector functions and similarity search
    Resolves "List argument must consist only of tuples or dictionaries" error
    """
    
    def __init__(self, db_client):
        self.db_client = db_client
        self.vector_dimension = 384  # Default for multilingual models
    
    @staticmethod
    def format_vector_for_tidb(vector) -> str:
        """
        ✅ FIXED: Convert numpy vector to TiDB-compatible vector format
        """
        try:
            if isinstance(vector, np.ndarray):
                vector_list = vector.astype(np.float32).tolist()
            elif isinstance(vector, list):
                vector_list = [float(x) for x in vector]
            elif hasattr(vector, 'tolist'):  # PyTorch tensor or similar
                vector_list = vector.tolist()
            else:
                raise ValueError(f"Unsupported vector type: {type(vector)}")
            
            # TiDB expects vector in format: '[1.0,2.0,3.0]'
            vector_str = '[' + ','.join(f'{x:.6f}' for x in vector_list) + ']'
            return vector_str
            
        except Exception as e:
            logger.error(f"❌ Vector formatting failed: {e}")
            return f"[{','.join(['0.0'] * 384)}]"  # Fallback zero vector
    
    @staticmethod
    def parse_vector_from_tidb(vector_str: str) -> np.ndarray:
        """Parse TiDB vector string back to numpy array"""
        try:
            if not vector_str or vector_str == '[]':
                return np.zeros(384, dtype=np.float32)
            
            # Remove brackets and split
            clean_str = vector_str.strip('[]')
            if not clean_str:
                return np.zeros(384, dtype=np.float32)
            
            values = [float(x.strip()) for x in clean_str.split(',')]
            return np.array(values, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"❌ Vector parsing failed: {e}")
            return np.zeros(384, dtype=np.float32)
    
    async def search_similar(self, query_vector, limit: int = 10, 
                       similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        ✅ FIXED: Basic vector similarity search with proper SQLAlchemy parameters
        """
        try:
            # ✅ FIXED: Ensure vector is properly formatted
            if isinstance(query_vector, (list, tuple)):
                formatted_vector = self.format_vector_for_tidb(np.array(query_vector))
            else:
                formatted_vector = self.format_vector_for_tidb(query_vector)
            
            session = self.db_client.SessionLocal()
            
            try:
                # ✅ FIXED: Use SQLAlchemy parameter style (:param) instead of %s
                query = text("""
                    SELECT id, title, abstract, authors, language, source,
                        paper_url, pdf_url, published_date, research_domain,
                        context_quality_score,
                        (1 - VEC_COSINE_DISTANCE(
                            CAST(embedding AS VECTOR(:dim)), 
                            CAST(:formatted_vector AS VECTOR(:dim))
                        )) as similarity_score
                    FROM research_papers
                    WHERE processing_status = 'completed'
                    AND LENGTH(embedding) > 10
                    AND (1 - VEC_COSINE_DISTANCE(
                        CAST(embedding AS VECTOR(:dim)), 
                        CAST(:formatted_vector AS VECTOR(:dim))
                    )) >= :similarity_threshold
                    ORDER BY similarity_score DESC 
                    LIMIT :limit
                """)
                
                # ✅ FIXED: Use proper parameter dict
                result = session.execute(query, {
                    'dim': self.vector_dimension,
                    'formatted_vector': formatted_vector,
                    'similarity_threshold': similarity_threshold,
                    'limit': limit
                })
                
                rows = result.fetchall()
                
                papers = []
                for row in rows:
                    try:
                        paper_dict = {
                            'id': row[0],
                            'title': row[1] or '',
                            'abstract': row[2] or '',
                            'authors': row[3] or '',
                            'language': row[4] or 'en',
                            'source': row[5] or 'unknown',
                            'paper_url': row[6] or '',
                            'pdf_url': row[7] or '',
                            'published_date': str(row[8]) if row[8] else None,
                            'research_domain': row[9] or 'General Research',
                            'context_quality_score': float(row[10]) if row[10] else 0.5,
                            'similarity_score': float(row[11]) if row[11] else 0.0
                        }
                        papers.append(paper_dict)
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing row: {e}")
                        continue
                
                logger.info(f"✅ Found {len(papers)} similar papers above threshold {similarity_threshold}")
                return papers
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"❌ Vector similarity search failed: {e}")
            return []


    
    async def search_similar_multilingual(self, query_embeddings: Dict[str, Any],
                                    limit: int = 20,
                                    similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
    
        try:
            all_results = []
            seen_paper_ids = set()
            
            # ✅ FIXED: Handle different input formats
            if isinstance(query_embeddings, dict):
                if 'embeddings' in query_embeddings:
                    # Extract embeddings from nested structure
                    embeddings = query_embeddings['embeddings']
                else:
                    # Assume direct language -> vector mapping
                    embeddings = query_embeddings
            else:
                logger.error("❌ Invalid query_embeddings format")
                return []
            
            if not embeddings:
                logger.warning("⚠️ No embeddings provided for multilingual search")
                return []
            
            # Search with each language vector
            for lang_code, vector_data in embeddings.items():
                try:
                    logger.info(f"🔍 Searching with {lang_code} vector")
                    
                    # ✅ FIXED: Handle different vector formats
                    if isinstance(vector_data, (list, tuple)):
                        query_vector = np.array(vector_data)
                    elif isinstance(vector_data, np.ndarray):
                        query_vector = vector_data
                    elif hasattr(vector_data, 'tolist'):
                        query_vector = np.array(vector_data.tolist())
                    else:
                        logger.warning(f"⚠️ Unsupported vector format for {lang_code}: {type(vector_data)}")
                        continue
                    
                    # Perform similarity search
                    lang_results = await self.search_similar(
                        query_vector, 
                        limit=limit//len(embeddings) + 3,  # Extra for deduplication
                        similarity_threshold=similarity_threshold
                    )
                    
                    # Add language metadata and deduplicate
                    for paper in lang_results:
                        paper_id = paper.get('id')
                        if paper_id and paper_id not in seen_paper_ids:
                            paper['matched_language'] = lang_code
                            paper['query_language'] = lang_code
                            paper['vector_search'] = True
                            all_results.append(paper)
                            seen_paper_ids.add(paper_id)
                
                except Exception as e:
                    logger.error(f"❌ Vector similarity search failed for {lang_code}: {e}")
                    continue
            
            # Sort by similarity score and limit
            all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            final_results = all_results[:limit]
            
            logger.info(f"✅ Multilingual search found {len(final_results)} unique papers")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Multilingual vector search failed: {e}")
            return []

    
    def _deduplicate_vector_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate papers from vector search results"""
        try:
            seen_ids = set()
            seen_titles = set()
            unique_results = []
            
            for paper in results:
                paper_id = paper.get('id')
                title = paper.get('title', '').strip().lower()
                
                # Check ID-based deduplication first
                if paper_id and paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    unique_results.append(paper)
                elif title and title not in seen_titles:
                    # Fallback to title-based deduplication
                    seen_titles.add(title)
                    unique_results.append(paper)
            
            return unique_results
            
        except Exception as e:
            logger.error(f"❌ Vector result deduplication failed: {e}")
            return results
    
    
    
    async def batch_update_embeddings(self, paper_embeddings: List[Tuple[int, np.ndarray]]) -> int:
        """✅ FIXED: Batch update multiple paper embeddings with proper SQL parameters"""
        try:
            if not paper_embeddings:
                return 0
            
            session = self.db_client.SessionLocal()
            successful_updates = 0
            
            try:
                # ✅ FIXED: Use executemany with list of dictionaries for batch operations
                batch_params = []
                
                for paper_id, embedding in paper_embeddings:
                    try:
                        formatted_vector = self.format_vector_for_tidb(embedding)
                        
                        # ✅ FIXED: Create parameter dictionary for each update
                        batch_params.append({
                            'formatted_vector': formatted_vector,
                            'paper_id': paper_id
                        })
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to format embedding for paper {paper_id}: {e}")
                        continue
                
                if batch_params:
                    # ✅ FIXED: Use named parameters with executemany for batch operations
                    update_query = text("""
                        UPDATE research_papers 
                        SET embedding = :formatted_vector, updated_at = NOW()
                        WHERE id = :paper_id
                    """)
                    
                    # ✅ FIXED: Use executemany with list of parameter dictionaries
                    result = session.execute(update_query, batch_params)
                    session.commit()
                    
                    successful_updates = len(batch_params)
                    logger.info(f"✅ Updated {successful_updates}/{len(paper_embeddings)} embeddings")
                
                return successful_updates
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"❌ Batch embedding update failed: {e}")
            return 0


    
    async def find_paper_clusters(self, similarity_threshold: float = 0.8, 
                                min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """Find clusters of similar papers using vector similarity"""
        try:
            session = self.db_client.SessionLocal()
            
            try:
                query = text("""
                    SELECT id, title, research_domain, embedding, context_quality_score
                    FROM research_papers 
                    WHERE processing_status = 'completed' 
                    AND LENGTH(embedding) > 10
                    ORDER BY context_quality_score DESC
                    LIMIT 1000
                """)
                
                result = session.execute(query)
                papers = []
                
                for row in result.fetchall():
                    papers.append({
                        'id': row[0],
                        'title': row[1] or '',
                        'research_domain': row[2] or 'Unknown',
                        'embedding': row[3] or '',
                        'context_quality_score': float(row[4]) if row[4] else 0.5
                    })
                
                if len(papers) < min_cluster_size:
                    return []
                
                # Simple clustering algorithm
                clusters = []
                used_papers = set()
                
                for i, paper1 in enumerate(papers):
                    if paper1['id'] in used_papers:
                        continue
                    
                    cluster = [paper1]
                    used_papers.add(paper1['id'])
                    
                    # Find similar papers
                    embedding1 = self.parse_vector_from_tidb(paper1['embedding'])
                    
                    for j, paper2 in enumerate(papers[i+1:], i+1):
                        if paper2['id'] in used_papers:
                            continue
                        
                        embedding2 = self.parse_vector_from_tidb(paper2['embedding'])
                        
                        # Calculate similarity
                        similarity = self._calculate_cosine_similarity(embedding1, embedding2)
                        
                        if similarity >= similarity_threshold:
                            cluster.append(paper2)
                            used_papers.add(paper2['id'])
                    
                    # Add cluster if it meets minimum size
                    if len(cluster) >= min_cluster_size:
                        clusters.append({
                            'cluster_id': len(clusters) + 1,
                            'size': len(cluster),
                            'papers': cluster,
                            'average_quality': sum(p['context_quality_score'] for p in cluster) / len(cluster),
                            'dominant_domain': self._get_dominant_domain(cluster)
                        })
                
                logger.info(f"✅ Found {len(clusters)} paper clusters")
                return clusters
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"❌ Paper clustering failed: {e}")
            return []
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _get_dominant_domain(self, papers: List[Dict[str, Any]]) -> str:
        """Get the most common research domain in a cluster"""
        try:
            domain_counts = {}
            for paper in papers:
                domain = paper.get('research_domain', 'Unknown')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            if domain_counts:
                return max(domain_counts.keys(), key=domain_counts.get)
            return 'Unknown'
            
        except Exception:
            return 'Unknown'
    
    async def get_vector_statistics(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        try:
            session = self.db_client.SessionLocal()
            
            try:
                # Count papers with embeddings
                count_query = text("""
                    SELECT 
                        COUNT(*) as total_papers,
                        SUM(CASE WHEN LENGTH(embedding) > 10 THEN 1 ELSE 0 END) as papers_with_embeddings,
                        AVG(context_quality_score) as avg_quality
                    FROM research_papers 
                    WHERE processing_status = 'completed'
                """)
                
                result = session.execute(count_query).fetchone()
                
                # Language distribution
                lang_query = text("""
                    SELECT language, COUNT(*) as count
                    FROM research_papers 
                    WHERE processing_status = 'completed' 
                    AND LENGTH(embedding) > 10
                    GROUP BY language 
                    ORDER BY count DESC
                """)
                
                lang_result = session.execute(lang_query)
                language_distribution = [
                    {'language': row[0], 'count': row[1]} 
                    for row in lang_result.fetchall()
                ]
                
                return {
                    'total_papers': result[0] if result else 0,
                    'papers_with_embeddings': result[1] if result else 0,
                    'average_quality_score': float(result[2]) if result and result[2] else 0.0,
                    'embedding_coverage_percent': (result[1] / max(result[0], 1) * 100) if result else 0,
                    'language_distribution': language_distribution,
                    'vector_dimension': self.vector_dimension
                }
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"❌ Failed to get vector statistics: {e}")
            return {}
