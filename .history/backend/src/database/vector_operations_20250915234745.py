import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from sqlalchemy import text
import traceback

logger = logging.getLogger(__name__)

class VectorOperations:
    """
    Enhanced VectorOperations with comprehensive debug logging for troubleshooting
    """
    
    def __init__(self, db_client):
        self.db_client = db_client
        self.vector_dimension = 384  # Default for multilingual models
        
        # Statistics tracking
        self.stats = {
            "searches_performed": 0,
            "vectors_formatted": 0,
            "vectors_parsed": 0,
            "db_queries_executed": 0,
            "papers_found": 0,
            "errors_encountered": 0
        }
        
        logger.info(f"🚀 VectorOperations initialized with dimension {self.vector_dimension}")
        logger.debug(f"📊 Stats tracking initialized: {self.stats}")
    
    @staticmethod
    def format_vector_for_tidb(vector) -> str:
        """Convert numpy vector to TiDB-compatible vector format with debug logging"""
        try:
            logger.debug(f"🔧 Formatting vector of type: {type(vector)}")
            
            if isinstance(vector, np.ndarray):
                logger.debug(f"📐 NumPy array shape: {vector.shape}, dtype: {vector.dtype}")
                vector_list = vector.astype(np.float32).tolist()
            elif isinstance(vector, list):
                logger.debug(f"📝 List length: {len(vector)}")
                vector_list = [float(x) for x in vector]
            elif hasattr(vector, 'tolist'):  # PyTorch tensor or similar
                logger.debug(f"🔄 Converting object with tolist() method: {type(vector)}")
                vector_list = vector.tolist()
            else:
                raise ValueError(f"Unsupported vector type: {type(vector)}")
            
            # Check vector dimensions
            if len(vector_list) != 384:
                logger.warning(f"⚠️ Vector dimension mismatch: {len(vector_list)} vs expected 384")
            
            # Check for problematic values
            has_nan = any(np.isnan(x) for x in vector_list[:10])  # Quick check
            has_inf = any(np.isinf(x) for x in vector_list[:10])
            
            if has_nan:
                logger.error("❌ Vector contains NaN values!")
            if has_inf:
                logger.error("❌ Vector contains infinite values!")
            
            # Calculate basic statistics
            vector_array = np.array(vector_list)
            norm = np.linalg.norm(vector_array)
            mean_val = np.mean(vector_array)
            
            logger.debug(f"📊 Vector stats - Norm: {norm:.6f}, Mean: {mean_val:.6f}")
            logger.debug(f"📊 Vector range - Min: {np.min(vector_array):.6f}, Max: {np.max(vector_array):.6f}")
            
            # TiDB expects vector in format: '[1.0,2.0,3.0]'
            vector_str = '[' + ','.join(f'{x:.6f}' for x in vector_list) + ']'
            
            logger.debug(f"✅ Formatted vector - Length: {len(vector_str)} chars")
            logger.debug(f"🔍 Vector preview: {vector_str[:100]}...")
            
            return vector_str
            
        except Exception as e:
            logger.error(f"❌ Vector formatting failed: {e}")
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            return f"[{','.join(['0.0'] * 384)}]"  # Fallback zero vector
    
    @staticmethod
    def parse_vector_from_tidb(vector_str: str) -> np.ndarray:
        """Parse TiDB vector string back to numpy array with debug logging"""
        try:
            logger.debug(f"🔄 Parsing vector string of length: {len(vector_str)}")
            
            if not vector_str or vector_str == '[]':
                logger.warning("⚠️ Empty vector string detected - returning zero vector")
                return np.zeros(384, dtype=np.float32)
            
            # Remove brackets and split
            clean_str = vector_str.strip('[]')
            if not clean_str:
                logger.warning("⚠️ Vector string has no content after bracket removal")
                return np.zeros(384, dtype=np.float32)
            
            logger.debug(f"🧹 Clean string sample: {clean_str[:100]}...")
            
            # Parse values
            values = [float(x.strip()) for x in clean_str.split(',')]
            logger.debug(f"📏 Parsed {len(values)} values from vector string")
            
            if len(values) != 384:
                logger.warning(f"⚠️ Parsed vector has {len(values)} dimensions, expected 384")
            
            vector_array = np.array(values, dtype=np.float32)
            
            # Validate parsed vector
            norm = np.linalg.norm(vector_array)
            has_nan = np.any(np.isnan(vector_array))
            has_inf = np.any(np.isinf(vector_array))
            
            logger.debug(f"📊 Parsed vector stats - Norm: {norm:.6f}, Has NaN: {has_nan}, Has Inf: {has_inf}")
            
            if has_nan or has_inf:
                logger.error("❌ Parsed vector contains invalid values!")
            
            return vector_array
            
        except Exception as e:
            logger.error(f"❌ Vector parsing failed: {e}")
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            return np.zeros(384, dtype=np.float32)
    
    async def debug_database_state(self) -> Dict[str, Any]:
        """Debug method to check database state"""
        logger.info("🔍 Starting database state debug check")
        
        try:
            session = self.db_client.SessionLocal()
            debug_info = {}
            
            try:
                # Check total papers
                total_query = text("SELECT COUNT(*) FROM research_papers")
                total_papers = session.execute(total_query).scalar() or 0
                debug_info['total_papers'] = total_papers
                logger.info(f"📊 Total papers in database: {total_papers}")
                
                # Check completed papers
                completed_query = text("SELECT COUNT(*) FROM research_papers WHERE processing_status = 'completed'")
                completed_papers = session.execute(completed_query).scalar() or 0
                debug_info['completed_papers'] = completed_papers
                logger.info(f"✅ Completed papers: {completed_papers}")
                
                # Check papers with embeddings
                embedding_query = text("""
                    SELECT COUNT(*) FROM research_papers 
                    WHERE processing_status = 'completed' 
                    AND embedding IS NOT NULL 
                    AND LENGTH(embedding) > 10
                """)
                papers_with_embeddings = session.execute(embedding_query).scalar() or 0
                debug_info['papers_with_embeddings'] = papers_with_embeddings
                logger.info(f"🧠 Papers with embeddings: {papers_with_embeddings}")
                
                # Sample embeddings for analysis
                sample_query = text("""
                    SELECT id, title, LENGTH(embedding) as embedding_length, 
                           LEFT(embedding, 100) as embedding_sample
                    FROM research_papers 
                    WHERE embedding IS NOT NULL 
                    AND LENGTH(embedding) > 10
                    LIMIT 5
                """)
                
                sample_result = session.execute(sample_query)
                samples = []
                
                for row in sample_result.fetchall():
                    sample = {
                        'id': row[0],
                        'title': row[1][:50],
                        'embedding_length': row[2],
                        'embedding_sample': row[3]
                    }
                    samples.append(sample)
                    logger.debug(f"📄 Sample paper {sample['id']}: '{sample['title']}' - embedding length: {sample['embedding_length']}")
                
                debug_info['sample_embeddings'] = samples
                
                # Check embedding format consistency
                if samples:
                    first_sample = samples[0]
                    logger.debug(f"🔍 First embedding sample: {first_sample['embedding_sample']}")
                    
                    # Try to parse a sample embedding
                    try:
                        full_embedding_query = text("SELECT embedding FROM research_papers WHERE id = :paper_id")
                        full_embedding = session.execute(full_embedding_query, {'paper_id': first_sample['id']}).scalar()
                        
                        if full_embedding:
                            parsed_vector = self.parse_vector_from_tidb(full_embedding)
                            debug_info['sample_vector_stats'] = {
                                'shape': parsed_vector.shape,
                                'norm': float(np.linalg.norm(parsed_vector)),
                                'mean': float(np.mean(parsed_vector)),
                                'has_nan': bool(np.any(np.isnan(parsed_vector))),
                                'has_inf': bool(np.any(np.isinf(parsed_vector)))
                            }
                            logger.info(f"📊 Sample vector parsed successfully: norm={debug_info['sample_vector_stats']['norm']:.6f}")
                        else:
                            logger.warning("⚠️ Could not retrieve full embedding for sample")
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to parse sample embedding: {e}")
                        debug_info['sample_parsing_error'] = str(e)
                
                return debug_info
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"❌ Database debug check failed: {e}")
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    async def search_similar(self, query_vector, limit: int = 10, 
                       similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Enhanced vector similarity search with comprehensive debug logging"""
        search_id = f"search_{self.stats['searches_performed'] + 1}"
        logger.info(f"🔍 [{search_id}] Starting vector similarity search")
        logger.debug(f"🎯 [{search_id}] Parameters - limit: {limit}, threshold: {similarity_threshold}")
        
        try:
            self.stats['searches_performed'] += 1
            
            # Step 1: Format query vector
            logger.debug(f"📝 [{search_id}] Formatting query vector of type: {type(query_vector)}")
            
            if isinstance(query_vector, (list, tuple)):
                logger.debug(f"📏 [{search_id}] Converting list/tuple of length {len(query_vector)}")
                formatted_vector = self.format_vector_for_tidb(np.array(query_vector))
            else:
                formatted_vector = self.format_vector_for_tidb(query_vector)
            
            self.stats['vectors_formatted'] += 1
            logger.debug(f"✅ [{search_id}] Vector formatted - length: {len(formatted_vector)} chars")
            logger.debug(f"🔍 [{search_id}] Formatted vector preview: {formatted_vector[:100]}...")
            
            # Step 2: Database connection
            logger.debug(f"💾 [{search_id}] Creating database session")
            session = self.db_client.SessionLocal()
            
            try:
                # Step 3: Build and execute query
                logger.debug(f"🏗️ [{search_id}] Building SQL query")
                
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
                    AND embedding IS NOT NULL
                    AND LENGTH(embedding) > 10
                    AND (1 - VEC_COSINE_DISTANCE(
                        CAST(embedding AS VECTOR(:dim)), 
                        CAST(:formatted_vector AS VECTOR(:dim))
                    )) >= :similarity_threshold
                    ORDER BY similarity_score DESC 
                    LIMIT :limit
                """)
                
                query_params = {
                    'dim': self.vector_dimension,
                    'formatted_vector': formatted_vector,
                    'similarity_threshold': similarity_threshold,
                    'limit': limit
                }
                
                logger.debug(f"🔢 [{search_id}] Query parameters: {query_params}")
                logger.debug(f"🚀 [{search_id}] Executing database query")
                
                result = session.execute(query, query_params)
                self.stats['db_queries_executed'] += 1
                
                # Step 4: Process results
                rows = result.fetchall()
                logger.info(f"📊 [{search_id}] Database returned {len(rows)} rows")
                
                if len(rows) == 0:
                    logger.warning(f"⚠️ [{search_id}] No rows returned - checking possible causes")
                    
                    # Debug: Check if any papers exist at all
                    debug_query = text("SELECT COUNT(*) FROM research_papers WHERE processing_status = 'completed' AND embedding IS NOT NULL AND LENGTH(embedding) > 10")
                    eligible_papers = session.execute(debug_query).scalar() or 0
                    logger.warning(f"🔍 [{search_id}] Eligible papers for search: {eligible_papers}")
                    
                    if eligible_papers > 0:
                        # Try with very low threshold
                        test_threshold = 0.01
                        logger.warning(f"🧪 [{search_id}] Testing with very low threshold: {test_threshold}")
                        
                        test_query = text("""
                            SELECT COUNT(*), MAX((1 - VEC_COSINE_DISTANCE(
                                CAST(embedding AS VECTOR(:dim)), 
                                CAST(:formatted_vector AS VECTOR(:dim))
                            ))) as max_similarity
                            FROM research_papers
                            WHERE processing_status = 'completed'
                            AND embedding IS NOT NULL
                            AND LENGTH(embedding) > 10
                        """)
                        
                        test_result = session.execute(test_query, {
                            'dim': self.vector_dimension,
                            'formatted_vector': formatted_vector
                        }).fetchone()
                        
                        if test_result:
                            logger.warning(f"📊 [{search_id}] Test results - Papers: {test_result[0]}, Max similarity: {test_result[1]:.6f}")
                
                papers = []
                for i, row in enumerate(rows):
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
                        
                        logger.debug(f"📄 [{search_id}] Paper {i+1}: ID={paper_dict['id']}, "
                                   f"Similarity={paper_dict['similarity_score']:.6f}, "
                                   f"Title='{paper_dict['title'][:50]}...'")
                        
                        papers.append(paper_dict)
                        
                    except Exception as e:
                        logger.error(f"❌ [{search_id}] Error processing row {i}: {e}")
                        continue
                
                self.stats['papers_found'] += len(papers)
                
                logger.info(f"✅ [{search_id}] Found {len(papers)} similar papers above threshold {similarity_threshold}")
                
                if papers:
                    best_similarity = max(p['similarity_score'] for p in papers)
                    worst_similarity = min(p['similarity_score'] for p in papers)
                    logger.info(f"📊 [{search_id}] Similarity range: {worst_similarity:.6f} - {best_similarity:.6f}")
                
                return papers
                
            finally:
                session.close()
                logger.debug(f"💾 [{search_id}] Database session closed")
                
        except Exception as e:
            self.stats['errors_encountered'] += 1
            logger.error(f"❌ [{search_id}] Vector similarity search failed: {e}")
            logger.error(f"📜 [{search_id}] Full traceback: {traceback.format_exc()}")
            return []

    async def search_similar_multilingual(self, query_embeddings: Dict[str, Any],
                                    limit: int = 20,
                                    similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Enhanced multilingual vector search with debug logging"""
        logger.info(f"🌍 Starting multilingual vector search with {len(query_embeddings) if isinstance(query_embeddings, dict) else 'unknown'} language(s)")
        
        try:
            all_results = []
            seen_paper_ids = set()
            
            # Handle different input formats
            if isinstance(query_embeddings, dict):
                if 'embeddings' in query_embeddings:
                    embeddings = query_embeddings['embeddings']
                    logger.debug(f"📦 Extracted embeddings from nested structure")
                else:
                    embeddings = query_embeddings
                    logger.debug(f"📦 Using direct language->vector mapping")
            else:
                logger.error("❌ Invalid query_embeddings format - expected dict")
                return []
            
            if not embeddings:
                logger.warning("⚠️ No embeddings provided for multilingual search")
                return []
            
            logger.info(f"🔍 Processing {len(embeddings)} language vectors")
            
            # Search with each language vector
            for i, (lang_code, vector_data) in enumerate(embeddings.items()):
                try:
                    logger.info(f"🔍 [{i+1}/{len(embeddings)}] Searching with {lang_code} vector")
                    
                    # Handle different vector formats
                    if isinstance(vector_data, (list, tuple)):
                        query_vector = np.array(vector_data)
                        logger.debug(f"📝 Converted {lang_code} list to numpy array: shape {query_vector.shape}")
                    elif isinstance(vector_data, np.ndarray):
                        query_vector = vector_data
                        logger.debug(f"📐 Using {lang_code} numpy array: shape {query_vector.shape}")
                    elif hasattr(vector_data, 'tolist'):
                        query_vector = np.array(vector_data.tolist())
                        logger.debug(f"🔄 Converted {lang_code} tensor to numpy array")
                    else:
                        logger.warning(f"⚠️ Unsupported vector format for {lang_code}: {type(vector_data)}")
                        continue
                    
                    # Perform similarity search
                    lang_limit = limit // len(embeddings) + 3  # Extra for deduplication
                    logger.debug(f"🎯 Searching {lang_code} with limit {lang_limit}")
                    
                    lang_results = await self.search_similar(
                        query_vector, 
                        limit=lang_limit,
                        similarity_threshold=similarity_threshold
                    )
                    
                    logger.info(f"📊 {lang_code} search returned {len(lang_results)} papers")
                    
                    # Add language metadata and deduplicate
                    added_count = 0
                    for paper in lang_results:
                        paper_id = paper.get('id')
                        if paper_id and paper_id not in seen_paper_ids:
                            paper['matched_language'] = lang_code
                            paper['query_language'] = lang_code
                            paper['vector_search'] = True
                            all_results.append(paper)
                            seen_paper_ids.add(paper_id)
                            added_count += 1
                    
                    logger.debug(f"✅ Added {added_count} unique papers from {lang_code} search")
                
                except Exception as e:
                    logger.error(f"❌ Vector similarity search failed for {lang_code}: {e}")
                    continue
            
            # Sort by similarity score and limit
            logger.debug(f"📊 Sorting {len(all_results)} total results by similarity")
            all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            final_results = all_results[:limit]
            
            logger.info(f"✅ Multilingual search found {len(final_results)} unique papers")
            
            if final_results:
                best_score = final_results[0].get('similarity_score', 0)
                worst_score = final_results[-1].get('similarity_score', 0)
                logger.info(f"📊 Final similarity range: {worst_score:.6f} - {best_score:.6f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Multilingual vector search failed: {e}")
            logger.error(f"📜 Full traceback: {traceback.format_exc()}")
            return []

    def _deduplicate_vector_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate papers with debug logging"""
        logger.debug(f"🔄 Deduplicating {len(results)} vector search results")
        
        try:
            seen_ids = set()
            seen_titles = set()
            unique_results = []
            
            for i, paper in enumerate(results):
                paper_id = paper.get('id')
                title = paper.get('title', '').strip().lower()
                
                if paper_id and paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    unique_results.append(paper)
                    logger.debug(f"✅ Keeping paper ID {paper_id}")
                elif title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(paper)
                    logger.debug(f"✅ Keeping paper by title: '{title[:50]}...'")
                else:
                    logger.debug(f"⏭️ Skipping duplicate: ID {paper_id}, title '{title[:50]}...'")
            
            logger.info(f"🔄 Deduplication: {len(unique_results)} unique from {len(results)} total")
            return unique_results
            
        except Exception as e:
            logger.error(f"❌ Vector result deduplication failed: {e}")
            return results

    async def update_paper_embedding(self, paper_id: int, embedding: np.ndarray) -> bool:
        """Update paper embedding with debug logging"""
        logger.info(f"🔄 Updating embedding for paper ID {paper_id}")
        
        try:
            formatted_vector = self.format_vector_for_tidb(embedding)
            logger.debug(f"✅ Formatted embedding for paper {paper_id}")
            
            session = self.db_client.SessionLocal()
            try:
                update_query = text("""
                    UPDATE research_papers 
                    SET embedding = :formatted_vector, updated_at = NOW()
                    WHERE id = :paper_id
                """)
                
                result = session.execute(update_query, {
                    'formatted_vector': formatted_vector,
                    'paper_id': paper_id
                })
                session.commit()
                
                if result.rowcount > 0:
                    logger.info(f"✅ Updated embedding for paper {paper_id}")
                    return True
                else:
                    logger.warning(f"⚠️ No paper found with ID {paper_id}")
                    return False
                    
            finally:
                session.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update embedding for paper {paper_id}: {e}")
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            return False

    async def batch_update_embeddings(self, paper_embeddings: List[Tuple[int, np.ndarray]]) -> int:
        """Batch update embeddings with debug logging"""
        logger.info(f"🔄 Starting batch update of {len(paper_embeddings)} embeddings")
        
        try:
            if not paper_embeddings:
                logger.warning("⚠️ No embeddings provided for batch update")
                return 0
            
            session = self.db_client.SessionLocal()
            successful_updates = 0
            
            try:
                batch_params = []
                
                for i, (paper_id, embedding) in enumerate(paper_embeddings):
                    try:
                        formatted_vector = self.format_vector_for_tidb(embedding)
                        
                        batch_params.append({
                            'formatted_vector': formatted_vector,
                            'paper_id': paper_id
                        })
                        
                        if (i + 1) % 100 == 0:
                            logger.debug(f"📊 Processed {i + 1}/{len(paper_embeddings)} embeddings for batch update")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to format embedding for paper {paper_id}: {e}")
                        continue
                
                if batch_params:
                    logger.debug(f"🚀 Executing batch update for {len(batch_params)} embeddings")
                    
                    update_query = text("""
                        UPDATE research_papers 
                        SET embedding = :formatted_vector, updated_at = NOW()
                        WHERE id = :paper_id
                    """)
                    
                    # Execute updates in smaller batches to avoid memory issues
                    batch_size = 50
                    for i in range(0, len(batch_params), batch_size):
                        batch_chunk = batch_params[i:i + batch_size]
                        result = session.execute(update_query, batch_chunk)
                        logger.debug(f"📊 Batch chunk {i//batch_size + 1}: updated {result.rowcount} records")
                    
                    session.commit()
                    successful_updates = len(batch_params)
                    
                    logger.info(f"✅ Batch update completed: {successful_updates}/{len(paper_embeddings)} embeddings updated")
                
                return successful_updates
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"❌ Batch embedding update failed: {e}")
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            return 0

    async def get_vector_statistics(self) -> Dict[str, Any]:
        """Get vector database statistics with debug logging"""
        logger.info("📊 Generating vector database statistics")
        
        try:
            session = self.db_client.SessionLocal()
            
            try:
                # Count papers with embeddings
                count_query = text("""
                    SELECT 
                        COUNT(*) as total_papers,
                        SUM(CASE WHEN embedding IS NOT NULL AND LENGTH(embedding) > 10 THEN 1 ELSE 0 END) as papers_with_embeddings,
                        AVG(context_quality_score) as avg_quality
                    FROM research_papers 
                    WHERE processing_status = 'completed'
                """)
                
                result = session.execute(count_query).fetchone()
                logger.debug(f"📊 Basic counts retrieved: {result}")
                
                # Language distribution
                lang_query = text("""
                    SELECT language, COUNT(*) as count
                    FROM research_papers 
                    WHERE processing_status = 'completed' 
                    AND embedding IS NOT NULL
                    AND LENGTH(embedding) > 10
                    GROUP BY language 
                    ORDER BY count DESC
                """)
                
                lang_result = session.execute(lang_query)
                language_distribution = [
                    {'language': row[0], 'count': row[1]} 
                    for row in lang_result.fetchall()
                ]
                
                logger.debug(f"📊 Language distribution: {language_distribution}")
                
                # Operation statistics
                operation_stats = self.stats.copy()
                
                statistics = {
                    'total_papers': result[0] if result else 0,
                    'papers_with_embeddings': result[1] if result else 0,
                    'average_quality_score': float(result[2]) if result and result[2] else 0.0,
                    'embedding_coverage_percent': (result[1] / max(result[0], 1) * 100) if result else 0,
                    'language_distribution': language_distribution,
                    'vector_dimension': self.vector_dimension,
                    'operation_statistics': operation_stats
                }
                
                logger.info(f"✅ Vector statistics generated: {statistics['papers_with_embeddings']} papers with embeddings")
                return statistics
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"❌ Failed to get vector statistics: {e}")
            logger.error(f"📜 Traceback: {traceback.format_exc()}")
            return {'error': str(e)}

    def get_debug_summary(self) -> Dict[str, Any]:
        """Get comprehensive debug summary"""
        return {
            'vector_dimension': self.vector_dimension,
            'operation_stats': self.stats,
            'class_info': {
                'methods_available': [method for method in dir(self) if not method.startswith('_')],
                'static_methods': ['format_vector_for_tidb', 'parse_vector_from_tidb']
            }
        }

    def reset_stats(self):
        """Reset operation statistics"""
        old_stats = self.stats.copy()
        self.stats = {
            "searches_performed": 0,
            "vectors_formatted": 0,
            "vectors_parsed": 0,
            "db_queries_executed": 0,
            "papers_found": 0,
            "errors_encountered": 0
        }
        logger.info(f"📊 Statistics reset. Previous: {old_stats}")
