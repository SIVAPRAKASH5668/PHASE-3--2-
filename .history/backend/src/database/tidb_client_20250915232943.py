import json
import logging
from typing import List, Dict, Any, Optional, Union
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

# Phase 2: Enhanced imports
from config.settings import settings
from database.vector_operations import VectorOperations

logger = logging.getLogger(__name__)

class EnhancedTiDBClient:
    """
    🚀 **Enhanced TiDB Client v2.0 with Multilingual & Vector Support**
    
    **New Features:**
    - 🌍 Multilingual paper storage and search
    - 🧠 Vector embeddings integration with TiDB
    - ⚡ Async operations with connection pooling
    - 📊 Advanced analytics and statistics
    - 🔍 Semantic similarity search
    - 🌐 Cross-language relationship tracking
    - 📈 Performance monitoring and optimization
    - 🛡️ Enhanced error handling and recovery
    - 💾 Intelligent caching with TTL
    - 🔧 AI agent metadata tracking
    """
    
    def __init__(self):
        # Enhanced database URL with optimal settings
        self.database_url = self._build_enhanced_database_url()
        
        # ✅ FIXED: Enhanced engine configuration with PyMySQL-compatible settings
        self.engine = create_engine(
            self.database_url,
            echo=False,
            pool_size=15,           # Increased pool size
            max_overflow=30,        # Increased overflow
            pool_recycle=3600,
            pool_pre_ping=True,
            pool_timeout=30,        # Connection timeout
            pool_reset_on_return='rollback',  # Enhanced cleanup
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
        
        # Phase 2: Enhanced components with error handling
        try:
            self.vector_ops = VectorOperations(self)
        except Exception as e:
            logger.warning(f"⚠️ VectorOperations initialization failed: {e}")
            self.vector_ops = None
        
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Enhanced performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_query_time": 0.0,
            "total_query_time": 0.0,
            "vector_operations": 0,
            "multilingual_operations": 0,
            "batch_operations": 0,
            "last_reset": time.time()
        }
        
        # Enhanced caching system
        self.query_cache = {}
        self.cache_ttl = getattr(settings, 'CACHE_TTL_DEFAULT', 3600)
        self.max_cache_size = getattr(settings, 'MAX_CACHE_SIZE', 5000)
        
        # ✅ ENHANCED: Test enhanced connection with better error handling
        try:
            self._test_enhanced_connection()
            logger.info("✅ Enhanced TiDB connection established successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Enhanced TiDB: {e}")
            try:
                self._fallback_connection_test()
                logger.warning("⚠️ Using fallback connection method")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback connection also failed: {fallback_error}")
                raise Exception(f"Database connection failed: {e}")

    def _fallback_connection_test(self):
        """Fallback connection test with basic parameters"""
        try:
            basic_url = f"mysql+pymysql://{settings.TIDB_USER}:{settings.TIDB_PASSWORD}@{settings.TIDB_HOST}:{settings.TIDB_PORT}/{settings.TIDB_DATABASE}?ssl_verify_cert=true"
            
            test_engine = create_engine(basic_url, pool_pre_ping=True)
            
            with test_engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()[0]
                
                if test_value == 1:
                    self.engine = test_engine
                    logger.info("✅ Fallback connection successful")
                else:
                    raise Exception("Basic connection test failed")
                    
        except Exception as e:
            raise Exception(f"Fallback connection failed: {e}")

    def _build_enhanced_database_url(self) -> str:
        """Build enhanced database URL with PyMySQL-compatible parameters"""
        base_url = (
            f"mysql+pymysql://{settings.TIDB_USER}:{settings.TIDB_PASSWORD}"
            f"@{settings.TIDB_HOST}:{settings.TIDB_PORT}/{settings.TIDB_DATABASE}"
        )
        
        # ✅ FIXED: Only PyMySQL-compatible parameters
        params = [
            "ssl_verify_cert=true",
            "charset=utf8mb4"
        ]
        
        return f"{base_url}?{'&'.join(params)}"  # ✅ FIXED: & instead of &amp;

    def _test_enhanced_connection(self):
        """Enhanced connection test with comprehensive validation"""
        try:
            with self.engine.connect() as conn:
                # Test basic connectivity
                result = conn.execute(text("SELECT 1 as test"))
                if result.fetchone()[0] != 1:
                    raise Exception("Basic connectivity test failed")
                
                # Test database access
                result = conn.execute(text("SELECT DATABASE() as db"))
                current_db = result.fetchone()[0]
                logger.info(f"📊 Connected to database: {current_db}")
                
                # Test character set
                result = conn.execute(text("SELECT @@character_set_database as charset"))
                charset = result.fetchone()[0]
                logger.info(f"🔤 Database charset: {charset}")
                
                logger.info("✅ Enhanced connection test passed")
                
        except Exception as e:
            logger.error(f"❌ Enhanced connection test failed: {e}")
            raise
    
    async def init_database(self):
        """Enhanced database initialization with vector support and optimal schema"""
        def _init_sync():
            with self.engine.begin() as conn:
                try:
                    # Enhanced papers table with vector support and multilingual fields
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS research_papers (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            -- Basic paper information
                            title TEXT NOT NULL,
                            abstract TEXT,
                            authors TEXT,
                            language VARCHAR(10) DEFAULT 'en',
                            source VARCHAR(50) DEFAULT 'unknown',
                            paper_url VARCHAR(500),
                            pdf_url VARCHAR(500),
                            published_date DATE,
                            -- Vector embeddings (TiDB Vector format)
                            embedding VECTOR(384),
                            embedding_model VARCHAR(100) DEFAULT 'multilingual-MiniLM-L12-v2',
                            -- Enhanced context analysis fields
                            context_summary TEXT,
                            research_domain VARCHAR(100) DEFAULT 'General Research',
                            methodology TEXT,
                            key_findings JSON,
                            innovations JSON,
                            limitations JSON,
                            research_questions JSON,
                            contributions JSON,
                            future_work JSON,
                            related_concepts JSON,
                            -- Quality and confidence metrics
                            context_quality_score FLOAT DEFAULT 0.5,
                            analysis_confidence FLOAT DEFAULT 0.8,
                            processing_status VARCHAR(20) DEFAULT 'pending',
                            -- Phase 2: Enhanced multilingual and AI fields
                            search_language VARCHAR(10),
                            detected_language VARCHAR(10),
                            ai_agent_used VARCHAR(50),
                            analysis_method VARCHAR(50),
                            search_method VARCHAR(50) DEFAULT 'traditional',
                            similarity_score FLOAT DEFAULT 0.0,
                            translation_data JSON,
                            processing_time FLOAT DEFAULT 0.0,
                            -- Timestamps with better precision
                            created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
                            updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
                            -- Enhanced indexes for optimal performance
                            INDEX idx_domain_quality (research_domain, context_quality_score DESC),
                            INDEX idx_language_search (language, search_language),
                            INDEX idx_status_created (processing_status, created_at DESC),
                            INDEX idx_title_fulltext (title(255)),
                            INDEX idx_abstract_search (abstract(500)),
                            INDEX idx_quality_confidence (context_quality_score DESC, analysis_confidence DESC),
                            INDEX idx_ai_agent (ai_agent_used, analysis_method),
                            INDEX idx_source_date (source, published_date DESC),
                            INDEX idx_multilingual (detected_language, search_language)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                        ROW_FORMAT=DYNAMIC
                    """))
                    
                    # Enhanced relationships table with AI agent tracking
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS paper_relationships (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            paper1_id INT NOT NULL,
                            paper2_id INT NOT NULL,
                            
                            -- Relationship details
                            relationship_type VARCHAR(50) DEFAULT 'related',
                            relationship_strength FLOAT DEFAULT 0.5,
                            relationship_context TEXT,
                            connection_reasoning TEXT,
                            
                            -- Phase 2: Enhanced AI and semantic fields
                            ai_agent_used VARCHAR(50),
                            analysis_method VARCHAR(50),
                            confidence_score FLOAT DEFAULT 0.7,
                            semantic_similarity FLOAT DEFAULT 0.0,
                            language_similarity FLOAT DEFAULT 0.0,
                            domain_overlap FLOAT DEFAULT 0.0,
                            methodology_similarity FLOAT DEFAULT 0.0,
                            processing_time FLOAT DEFAULT 0.0,
                            
                            -- Cross-linguistic relationship tracking
                            is_cross_linguistic BOOLEAN DEFAULT FALSE,
                            language_pair VARCHAR(20),
                            
                            created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
                            updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
                            
                            -- Enhanced constraints and indexes
                            UNIQUE KEY unique_relationship (paper1_id, paper2_id),
                            INDEX idx_papers_strength (paper1_id, paper2_id, relationship_strength DESC),
                            INDEX idx_type_strength (relationship_type, relationship_strength DESC),
                            INDEX idx_ai_analysis (ai_agent_used, analysis_method),
                            INDEX idx_cross_linguistic (is_cross_linguistic, language_pair),
                            INDEX idx_semantic_sim (semantic_similarity DESC),
                            INDEX idx_confidence (confidence_score DESC),
                            INDEX idx_paper1_relations (paper1_id, relationship_strength DESC),
                            INDEX idx_paper2_relations (paper2_id, relationship_strength DESC),
                            
                            FOREIGN KEY (paper1_id) REFERENCES research_papers(id) ON DELETE CASCADE,
                            FOREIGN KEY (paper2_id) REFERENCES research_papers(id) ON DELETE CASCADE
                            
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """))
                    
                    # Phase 2: Additional analytics table for performance tracking
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS system_analytics (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            metric_name VARCHAR(100) NOT NULL,
                            metric_value FLOAT NOT NULL,
                            metric_metadata JSON,
                            recorded_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
                            INDEX idx_metric_time (metric_name, recorded_at DESC),
                            INDEX idx_recorded_at (recorded_at DESC)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """))
                    
                    logger.info("✅ Enhanced database tables created")
                    
                    # Vector index creation with proper error handling
                     try:
                        # Create new vector index with unique name
                        conn.execute(text("""
                            CREATE VECTOR INDEX IF NOT EXISTS idx_embedding_v3 
                            ON research_papers ((VEC_COSINE_DISTANCE(embedding)))
                        """))
                        logger.info("✅ Vector index created successfully")
                        
                    except Exception as vector_error:
                        # THIS IS WHERE YOUR ERROR IS LOGGED
                        error_msg = str(vector_error).lower()
                        if "already exist" in error_msg or "duplicate" in error_msg:
                            logger.info("✅ Vector index already exists - using existing index")
                        else:
                            logger.warning(f"⚠️ Vector index creation failed: {vector_error}")
                            logger.info("📝 Vector search will use full table scan")

                        
                except Exception as e:
                    logger.error(f"❌ Enhanced database initialization failed: {e}")
                    raise
        
        # Run initialization in thread pool for better async compatibility
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _init_sync)

    def _enhanced_cache_management(self):
        """Enhanced cache management with TTL and size limits"""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = [
            key for key, (_, timestamp) in self.query_cache.items()
            if current_time - timestamp > self.cache_ttl  # ✅ FIXED: > instead of &gt;
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
            
        # Manage cache size
        if len(self.query_cache) > self.max_cache_size:  # ✅ FIXED: > instead of &gt;
            # Remove 20% of oldest entries
            sorted_cache = sorted(
                self.query_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            remove_count = len(sorted_cache) // 5
            for key, _ in sorted_cache[:remove_count]:
                del self.query_cache[key]
    
    def _get_cache_key(self, query: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key for query and parameters"""
        cache_input = f"{query}_{json.dumps(params, sort_keys=True) if params else ''}"
        return hashlib.md5(cache_input.encode()).hexdigest()[:16]
    
    def _update_performance_stats(self, query_time: float, success: bool, operation_type: str = "query"):
        """Update enhanced performance statistics"""
        self.performance_stats["total_queries"] += 1
        
        if success:
            self.performance_stats["successful_queries"] += 1
            
            # Update average query time
            total_successful = self.performance_stats["successful_queries"]
            current_avg = self.performance_stats["average_query_time"]
            self.performance_stats["average_query_time"] = (
                (current_avg * (total_successful - 1) + query_time) / total_successful
            )
            
            self.performance_stats["total_query_time"] += query_time
        else:
            self.performance_stats["failed_queries"] += 1
        
        # Track operation types
        if operation_type == "vector":
            self.performance_stats["vector_operations"] += 1
        elif operation_type == "multilingual":
            self.performance_stats["multilingual_operations"] += 1
        elif operation_type == "batch":
            self.performance_stats["batch_operations"] += 1
    
    def _prepare_enhanced_paper_data(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced paper data preparation with multilingual and vector support"""
        clean_data = {}
        
        # Basic fields with enhanced validation
        clean_data['title'] = str(paper_data.get('title', '') or '')[:2000]
        clean_data['abstract'] = str(paper_data.get('abstract', '') or '')[:10000]
        clean_data['authors'] = str(paper_data.get('authors', '') or '')[:2000]
        clean_data['language'] = str(paper_data.get('language', 'en') or 'en')[:10]
        clean_data['source'] = str(paper_data.get('source', 'unknown') or 'unknown')[:50]
        clean_data['paper_url'] = str(paper_data.get('paper_url', '') or '')[:500]
        clean_data['pdf_url'] = str(paper_data.get('pdf_url', '') or '')[:500]
        
        # Enhanced date handling
        pub_date = paper_data.get('published_date')
        clean_data['published_date'] = self._parse_date(pub_date)
        
        # Vector embedding with proper format validation
        embedding = paper_data.get('embedding')
        if embedding:
            try:
                if isinstance(embedding, str):
                    # Ensure proper vector format for TiDB
                    if embedding.startswith('[') and embedding.endswith(']'):
                        clean_data['embedding'] = embedding
                    else:
                        clean_data['embedding'] = f"[{embedding}]"
                elif isinstance(embedding, list):
                    # Convert list to TiDB vector format
                    clean_data['embedding'] = '[' + ','.join(map(str, embedding)) + ']'
                elif hasattr(embedding, 'tolist'):  # NumPy array
                    vector_list = embedding.tolist()
                    clean_data['embedding'] = '[' + ','.join(map(str, vector_list)) + ']'
                else:
                    clean_data['embedding'] = None
            except Exception as e:
                logger.warning(f"⚠️ Invalid embedding data: {e}")
                clean_data['embedding'] = None
        else:
            clean_data['embedding'] = None
        
        clean_data['embedding_model'] = str(paper_data.get('embedding_model', 'multilingual-MiniLM-L12-v2'))[:100]
        
        # Enhanced context fields
        clean_data['context_summary'] = str(paper_data.get('context_summary', '') or '')[:5000]
        clean_data['research_domain'] = str(paper_data.get('research_domain', 'General Research') or 'General Research')[:100]
        clean_data['methodology'] = str(paper_data.get('methodology', '') or '')[:3000]
        
        # Enhanced JSON fields with validation
        json_fields = ['key_findings', 'innovations', 'limitations',
                      'research_questions', 'contributions', 'future_work', 'related_concepts']
        
        for field in json_fields:
            value = paper_data.get(field, [])
            clean_data[field] = self._prepare_json_field(value, max_items=10, max_length=500)
        
        # Enhanced numeric fields
        clean_data['context_quality_score'] = self._validate_score(paper_data.get('context_quality_score', 0.5))
        clean_data['analysis_confidence'] = self._validate_score(paper_data.get('analysis_confidence', 0.8))
        clean_data['similarity_score'] = self._validate_score(paper_data.get('similarity_score', 0.0))
        clean_data['processing_time'] = max(0.0, float(paper_data.get('processing_time', 0.0)))
        
        # Phase 2: Enhanced multilingual fields
        clean_data['search_language'] = str(paper_data.get('search_language', '') or '')[:10]
        clean_data['detected_language'] = str(paper_data.get('detected_language', '') or '')[:10]
        clean_data['ai_agent_used'] = str(paper_data.get('ai_agent_used', '') or '')[:50]
        clean_data['analysis_method'] = str(paper_data.get('analysis_method', '') or '')[:50]
        clean_data['search_method'] = str(paper_data.get('search_method', 'traditional') or 'traditional')[:50]
        
        # Translation data as JSON
        translation_data = paper_data.get('translation_data', paper_data.get('translations', {}))
        if translation_data and isinstance(translation_data, dict):
            clean_data['translation_data'] = json.dumps(translation_data)
        else:
            clean_data['translation_data'] = "{}"
        
        clean_data['processing_status'] = str(paper_data.get('processing_status', 'completed') or 'completed')[:20]
        
        return clean_data
    
    def _prepare_enhanced_relationship_data(self, relationship_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced relationship data preparation with AI metadata"""
        # Validate and clean paper IDs
        paper1_id = int(relationship_data.get('paper1_id', 0))
        paper2_id = int(relationship_data.get('paper2_id', 0))
        
        # Ensure consistent ordering
        if paper1_id > paper2_id:  # ✅ FIXED: > instead of &gt;
            paper1_id, paper2_id = paper2_id, paper1_id
        
        clean_data = {
            'paper1_id': paper1_id,
            'paper2_id': paper2_id,
            'relationship_type': str(relationship_data.get('relationship_type', 'related') or 'related')[:50],
            'relationship_strength': self._validate_score(relationship_data.get('relationship_strength', 0.5)),
            'relationship_context': str(relationship_data.get('relationship_context', '') or '')[:3000],
            'connection_reasoning': str(relationship_data.get('connection_reasoning', '') or '')[:3000],
            
            # Phase 2: Enhanced AI and semantic fields
            'ai_agent_used': str(relationship_data.get('ai_agent_used', '') or '')[:50],
            'analysis_method': str(relationship_data.get('analysis_method', '') or '')[:50],
            'confidence_score': self._validate_score(relationship_data.get('confidence_score', 0.7)),
            'semantic_similarity': self._validate_score(relationship_data.get('semantic_similarity', 0.0)),
            'language_similarity': self._validate_score(relationship_data.get('language_similarity', 0.0)),
            'domain_overlap': self._validate_score(relationship_data.get('domain_overlap', 0.0)),
            'methodology_similarity': self._validate_score(relationship_data.get('methodology_similarity', 0.0)),
            'processing_time': max(0.0, float(relationship_data.get('processing_time', 0.0))),
            
            # Cross-linguistic tracking
            'is_cross_linguistic': bool(relationship_data.get('is_cross_linguistic', False)),
            'language_pair': str(relationship_data.get('language_pair', '') or '')[:20]
        }
        
        return clean_data
    
    def _parse_date(self, date_value: Any) -> Optional[date]:
        """Enhanced date parsing with multiple format support"""
        if not date_value:
            return None
        
        if isinstance(date_value, (date, datetime)):
            return date_value.date() if isinstance(date_value, datetime) else date_value
        
        if isinstance(date_value, str):
            date_formats = [
                '%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S.%f'
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_value, fmt).date()
                except ValueError:
                    continue
        
        return None
    
    def _prepare_json_field(self, value: Any, max_items: int = 10, max_length: int = 500) -> str:
        """Enhanced JSON field preparation with validation"""
        if isinstance(value, list):
            clean_list = []
            for item in value[:max_items]:
                if item is not None:
                    item_str = str(item).strip()[:max_length]
                    if item_str:
                        clean_list.append(item_str)
            return json.dumps(clean_list)
        elif isinstance(value, str) and value.strip():
            return json.dumps([value.strip()[:max_length]])
        elif isinstance(value, dict):
            return json.dumps(value)
        else:
            return "[]"
    
    def _validate_score(self, score: Any) -> float:
        """Enhanced score validation with bounds checking"""
        try:
            float_score = float(score)
            return max(0.0, min(1.0, float_score))
        except (ValueError, TypeError):
            return 0.5
    
    def _convert_enhanced_paper_from_db(self, paper_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced conversion of database paper record to Python objects"""
        if not paper_dict:
            return paper_dict
        
        # JSON fields conversion
        json_fields = ['key_findings', 'innovations', 'limitations', 
                      'research_questions', 'contributions', 'future_work', 'related_concepts']
        
        for field in json_fields:
            if field in paper_dict and paper_dict[field]:
                try:
                    if isinstance(paper_dict[field], str):
                        paper_dict[field] = json.loads(paper_dict[field])
                except (json.JSONDecodeError, TypeError):
                    paper_dict[field] = []
            else:
                paper_dict[field] = []
        
        # Enhanced translation data conversion
        if 'translation_data' in paper_dict and paper_dict['translation_data']:
            try:
                if isinstance(paper_dict['translation_data'], str):
                    paper_dict['translation_data'] = json.loads(paper_dict['translation_data'])
            except (json.JSONDecodeError, TypeError):
                paper_dict['translation_data'] = {}
        
        return paper_dict
    
    def _safe_enhanced_row_to_dict(self, row) -> Optional[Dict[str, Any]]:
        """Enhanced row to dictionary conversion with comprehensive error handling"""
        if row is None:
            return None
        
        try:
            if hasattr(row, '_asdict'):
                result = row._asdict()
            elif hasattr(row, '_mapping'):
                result = dict(row._mapping)
            elif hasattr(row, 'keys'):
                result = dict(zip(row.keys(), row))
            else:
                result = dict(row)
            
            return self._convert_enhanced_paper_from_db(result)
            
        except Exception as e:
            logger.error(f"❌ Enhanced row conversion failed: {e}")
            return None
    
    async def store_paper(self, paper_data: Dict[str, Any]) -> Optional[int]:
        """Enhanced async paper storage with caching and performance tracking"""
        start_time = time.time()
        
        def _store_sync():
            if not paper_data.get('title'):
                logger.error("Cannot store paper without title")
                return None
            
            session = self.SessionLocal()
            try:
                # Check for existing paper
                title = paper_data.get('title', '').strip()
                existing_query = text("SELECT id FROM research_papers WHERE title = :title LIMIT 1")
                existing = session.execute(existing_query, {"title": title}).fetchone()
                
                if existing:
                    paper_id = existing[0]
                    logger.info(f"📄 Paper already exists with ID {paper_id}")
                    return paper_id
                
                # Prepare enhanced data
                clean_data = self._prepare_enhanced_paper_data(paper_data)
                
                # Enhanced insert query with all new fields
                insert_query = text("""
                    INSERT INTO research_papers (
                        title, abstract, authors, language, source, paper_url, pdf_url,
                        published_date, embedding, embedding_model, context_summary, research_domain,
                        methodology, key_findings, innovations, limitations,
                        research_questions, contributions, future_work, related_concepts,
                        context_quality_score, analysis_confidence, similarity_score,
                        search_language, detected_language, ai_agent_used, analysis_method,
                        search_method, translation_data, processing_time, processing_status
                    ) VALUES (
                        :title, :abstract, :authors, :language, :source, :paper_url, :pdf_url,
                        :published_date, :embedding, :embedding_model, :context_summary, :research_domain,
                        :methodology, :key_findings, :innovations, :limitations,
                        :research_questions, :contributions, :future_work, :related_concepts,
                        :context_quality_score, :analysis_confidence, :similarity_score,
                        :search_language, :detected_language, :ai_agent_used, :analysis_method,
                        :search_method, :translation_data, :processing_time, :processing_status
                    )
                """)
                
                result = session.execute(insert_query, clean_data)
                session.commit()
                
                paper_id = result.lastrowid
                logger.info(f"✅ Enhanced paper stored with ID {paper_id}: {clean_data['title'][:50]}")
                return paper_id
                
            except Exception as e:
                session.rollback()
                logger.error(f"❌ Enhanced paper storage failed: {e}")
                return None
            finally:
                session.close()
        
        # Execute in thread pool for async compatibility
        loop = asyncio.get_event_loop()
        paper_id = await loop.run_in_executor(self.thread_pool, _store_sync)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, paper_id is not None, "store")
        
        return paper_id
    
    # ✅ CRITICAL FIX: Replace the embedding handling section in update_paper method
    async def update_paper(self, paper_id: int, update_data: Dict[str, Any]) -> bool:
        """Enhanced async paper update with selective field updates - FIXED NumPy boolean error"""
        start_time = time.time()
        
        def _update_sync():
            session = self.SessionLocal()
            try:
                # Prepare update data
                clean_data = {}
                
                # Only include fields that are provided
                updateable_fields = [
                    'title', 'abstract', 'authors', 'language', 'source', 'paper_url', 'pdf_url',
                    'published_date', 'embedding', 'embedding_model', 'context_summary', 'research_domain',
                    'methodology', 'key_findings', 'innovations', 'limitations',
                    'research_questions', 'contributions', 'future_work', 'related_concepts',
                    'context_quality_score', 'analysis_confidence', 'similarity_score',
                    'search_language', 'detected_language', 'ai_agent_used', 'analysis_method',
                    'search_method', 'translation_data', 'processing_time', 'processing_status'
                ]
                
                # ✅ CRITICAL FIX: Handle embeddings properly to avoid NumPy boolean error
                for field in updateable_fields:
                    if field in update_data:
                        if field == 'embedding':
                            embedding_value = update_data[field]
                            # ✅ FIXED: Proper NumPy array handling
                            try:
                                # Check if it's None first
                                if embedding_value is None:
                                    clean_data[field] = None
                                # Check if it's already a string (TiDB vector format)
                                elif isinstance(embedding_value, str):
                                    clean_data[field] = embedding_value
                                # Check if it's a list
                                elif isinstance(embedding_value, list):
                                    clean_data[field] = '[' + ','.join(map(str, embedding_value)) + ']'
                                # Check if it's a NumPy array using proper method
                                elif hasattr(embedding_value, '__array__') or str(type(embedding_value)).find('numpy') != -1:
                                    # It's a NumPy array - convert safely
                                    if hasattr(embedding_value, 'tolist'):
                                        vector_list = embedding_value.tolist()
                                        clean_data[field] = '[' + ','.join(map(str, vector_list)) + ']'
                                    else:
                                        logger.warning(f"⚠️ Cannot convert embedding to list: {type(embedding_value)}")
                                        clean_data[field] = None
                                else:
                                    logger.warning(f"⚠️ Unknown embedding type: {type(embedding_value)}")
                                    clean_data[field] = None
                            except Exception as e:
                                logger.error(f"❌ Embedding conversion error: {e}")
                                clean_data[field] = None
                        else:
                            # Process other fields normally
                            full_data = self._prepare_enhanced_paper_data({field: update_data[field]})
                            clean_data[field] = full_data.get(field)
                
                if not clean_data:
                    logger.warning("No valid fields to update")
                    return False
                
                # Build dynamic update query
                set_clauses = [f"{field} = :{field}" for field in clean_data.keys()]
                update_query = text(f"""
                    UPDATE research_papers
                    SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP(3)
                    WHERE id = :paper_id
                """)
                
                clean_data['paper_id'] = paper_id
                result = session.execute(update_query, clean_data)
                session.commit()
                
                if result.rowcount > 0:  # ✅ FIXED: > instead of &gt;
                    logger.info(f"✅ Enhanced paper {paper_id} updated successfully")
                    return True
                else:
                    logger.warning(f"⚠️ Paper {paper_id} not found for update")
                    return False
                    
            except Exception as e:
                session.rollback()
                logger.error(f"❌ Enhanced paper update failed: {e}")
                return False
            finally:
                session.close()
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(self.thread_pool, _update_sync)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, success, "update")
        
        return success

    async def get_paper_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get paper by title - missing method that workflow expects"""
        start_time = time.time()
        
        def _get_by_title_sync():
            session = self.SessionLocal()
            try:
                query = text("SELECT * FROM research_papers WHERE title = :title LIMIT 1")
                result = session.execute(query, {"title": title.strip()})
                row = result.fetchone()
                return self._safe_enhanced_row_to_dict(row)
            except Exception as e:
                logger.error(f"❌ Get paper by title failed: {e}")
                return None
            finally:
                session.close()
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        paper = await loop.run_in_executor(self.thread_pool, _get_by_title_sync)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, paper is not None, "retrieve")
        
        return paper

    async def get_paper_by_id(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """Enhanced async paper retrieval with caching"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"paper_{paper_id}"
        if cache_key in self.query_cache:
            cached_result, timestamp = self.query_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:  # ✅ FIXED: < instead of &lt;
                self.performance_stats["cache_hits"] += 1
                return cached_result
        
        def _get_sync():
            session = self.SessionLocal()
            try:
                query = text("SELECT * FROM research_papers WHERE id = :id")
                result = session.execute(query, {"id": paper_id})
                row = result.fetchone()
                return self._safe_enhanced_row_to_dict(row)
            except Exception as e:
                logger.error(f"❌ Enhanced get paper by ID failed: {e}")
                return None
            finally:
                session.close()
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        paper = await loop.run_in_executor(self.thread_pool, _get_sync)
        
        # Cache result
        if paper:
            self.query_cache[cache_key] = (paper, time.time())
            self.performance_stats["cache_misses"] += 1
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, paper is not None, "retrieve")
        
        return paper
    
    async def store_paper_relationship(self, relationship_data: Dict[str, Any]) -> Optional[int]:
        """Enhanced async relationship storage with AI metadata"""
        start_time = time.time()
        
        def _store_relationship_sync():
            session = self.SessionLocal()
            try:
                clean_data = self._prepare_enhanced_relationship_data(relationship_data)
                
                # Enhanced insert with ON DUPLICATE KEY UPDATE
                insert_query = text("""
                    INSERT INTO paper_relationships (
                        paper1_id, paper2_id, relationship_type, relationship_strength,
                        relationship_context, connection_reasoning, ai_agent_used, analysis_method,
                        confidence_score, semantic_similarity, language_similarity, domain_overlap,
                        methodology_similarity, processing_time, is_cross_linguistic, language_pair
                    ) VALUES (
                        :paper1_id, :paper2_id, :relationship_type, :relationship_strength,
                        :relationship_context, :connection_reasoning, :ai_agent_used, :analysis_method,
                        :confidence_score, :semantic_similarity, :language_similarity, :domain_overlap,
                        :methodology_similarity, :processing_time, :is_cross_linguistic, :language_pair
                    )
                    ON DUPLICATE KEY UPDATE
                    relationship_strength = GREATEST(relationship_strength, VALUES(relationship_strength)),
                    relationship_context = VALUES(relationship_context),
                    connection_reasoning = VALUES(connection_reasoning),
                    ai_agent_used = VALUES(ai_agent_used),
                    analysis_method = VALUES(analysis_method),
                    confidence_score = VALUES(confidence_score),
                    semantic_similarity = VALUES(semantic_similarity),
                    language_similarity = VALUES(language_similarity),
                    domain_overlap = VALUES(domain_overlap),
                    methodology_similarity = VALUES(methodology_similarity),
                    processing_time = VALUES(processing_time),
                    is_cross_linguistic = VALUES(is_cross_linguistic),
                    language_pair = VALUES(language_pair),
                    updated_at = CURRENT_TIMESTAMP(3)
                """)
                
                result = session.execute(insert_query, clean_data)
                session.commit()
                
                rel_id = result.lastrowid
                if rel_id == 0:  # Updated existing
                    select_query = text("""
                        SELECT id FROM paper_relationships 
                        WHERE paper1_id = :paper1_id AND paper2_id = :paper2_id
                    """)
                    existing = session.execute(select_query, {
                        'paper1_id': clean_data['paper1_id'],
                        'paper2_id': clean_data['paper2_id']
                    }).fetchone()
                    rel_id = existing[0] if existing else None
                
                logger.info(f"✅ Enhanced relationship stored: {clean_data['relationship_type']} "
                        f"(strength: {clean_data['relationship_strength']:.2f}, ID: {rel_id})")
                return rel_id
                
            except Exception as e:
                session.rollback()
                logger.error(f"❌ Enhanced relationship storage failed: {e}")
                return None
            finally:
                session.close()
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        rel_id = await loop.run_in_executor(self.thread_pool, _store_relationship_sync)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, rel_id is not None, "relationship")
        
        return rel_id

    async def search_papers_text(self, query: str, limit: int = 20, 
                                language: str = None, domain: str = None,
                                include_translations: bool = False) -> List[Dict[str, Any]]:
        """Enhanced multilingual text search with translation support"""
        start_time = time.time()
        
        def _search_sync():
            if not query or not query.strip():
                return []
            
            session = self.SessionLocal()
            try:
                # Build enhanced search query with multilingual support
                base_conditions = []
                params = {"limit": limit}
                
                # Text search conditions
                search_pattern = f"%{query.strip()}%"
                text_conditions = [
                    "title LIKE :search_pattern",
                    "abstract LIKE :search_pattern",
                    "authors LIKE :search_pattern",
                    "context_summary LIKE :search_pattern"
                ]
                
                # Include translation search if requested
                if include_translations:
                    text_conditions.append("JSON_EXTRACT(translation_data, '$.*') LIKE :search_pattern")
                
                base_conditions.append("(" + " OR ".join(text_conditions) + ")")
                params["search_pattern"] = search_pattern
                
                # Language filter
                if language:
                    base_conditions.append("(language = :language OR detected_language = :language)")
                    params["language"] = language
                
                # Domain filter
                if domain:
                    base_conditions.append("research_domain = :domain")
                    params["domain"] = domain
                
                # Build final query with enhanced ranking
                where_clause = " AND ".join(base_conditions)
                search_query = text(f"""
                    SELECT *, 
                           CASE 
                               WHEN title LIKE :exact_pattern THEN 1
                               WHEN title LIKE :search_pattern THEN 2
                               WHEN abstract LIKE :search_pattern THEN 3
                               WHEN context_summary LIKE :search_pattern THEN 4
                               ELSE 5
                           END as relevance_rank
                    FROM research_papers 
                    WHERE {where_clause}
                    ORDER BY relevance_rank ASC, context_quality_score DESC, analysis_confidence DESC
                    LIMIT :limit
                """)
                
                params["exact_pattern"] = query.strip()
                result = session.execute(search_query, params)
                
                papers = []
                for row in result.fetchall():
                    paper_dict = self._safe_enhanced_row_to_dict(row)
                    if paper_dict:
                        # Add search metadata
                        paper_dict['search_relevance'] = 1.0 / paper_dict.get('relevance_rank', 5)
                        papers.append(paper_dict)
                
                return papers
                
            except Exception as e:
                logger.error(f"❌ Enhanced text search failed: {e}")
                return []
            finally:
                session.close()
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        papers = await loop.run_in_executor(self.thread_pool, _search_sync)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, True, "multilingual")
        
        return papers
    
    async def get_papers_with_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced paper retrieval with comprehensive filtering"""
        start_time = time.time()
        
        def _get_filtered_sync():
            session = self.SessionLocal()
            try:
                # Build dynamic filter query
                conditions = []
                params = {}
                
                # Pagination
                limit = filters.get('limit', 20)
                offset = filters.get('offset', 0)
                params['limit'] = limit
                params['offset'] = offset
                
                # Domain filter
                if filters.get('domain'):
                    conditions.append("research_domain = :domain")
                    params['domain'] = filters['domain']
                
                # Language filter
                if filters.get('language'):
                    conditions.append("(language = :language OR detected_language = :language)")
                    params['language'] = filters['language']
                
                # Quality filter - ✅ FIXED: > instead of &gt;
                if filters.get('min_quality', 0) > 0:
                    conditions.append("context_quality_score >= :min_quality")  # ✅ FIXED: >= instead of &gt;=
                    params['min_quality'] = filters['min_quality']
                
                # AI agent filter
                if filters.get('ai_agent'):
                    conditions.append("ai_agent_used = :ai_agent")
                    params['ai_agent'] = filters['ai_agent']
                
                # Status filter
                if filters.get('status'):
                    conditions.append("processing_status = :status")
                    params['status'] = filters['status']
                else:
                    conditions.append("processing_status = 'completed'")
                
                # Date range filter - ✅ FIXED: >= and <= instead of &gt;= and &lt;=
                if filters.get('date_from'):
                    conditions.append("published_date >= :date_from")
                    params['date_from'] = filters['date_from']
                
                if filters.get('date_to'):
                    conditions.append("published_date <= :date_to")
                    params['date_to'] = filters['date_to']
                
                # Build WHERE clause
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                # Build ORDER BY clause
                sort_by = filters.get('sort_by', 'created_at')
                sort_order = filters.get('sort_order', 'desc').upper()
                
                # Validate sort fields
                valid_sort_fields = [
                    'created_at', 'updated_at', 'title', 'context_quality_score',
                    'analysis_confidence', 'published_date', 'processing_time'
                ]
                
                if sort_by not in valid_sort_fields:
                    sort_by = 'created_at'
                
                if sort_order not in ['ASC', 'DESC']:
                    sort_order = 'DESC'
                
                # Get total count
                count_query = text(f"SELECT COUNT(*) FROM research_papers WHERE {where_clause}")
                count_params = {k: v for k, v in params.items() if k not in ['limit', 'offset']}
                total_count = session.execute(count_query, count_params).scalar() or 0
                
                # Get papers
                papers_query = text(f"""
                    SELECT * FROM research_papers 
                    WHERE {where_clause}
                    ORDER BY {sort_by} {sort_order}
                    LIMIT :limit OFFSET :offset
                """)
                
                result = session.execute(papers_query, params)
                papers = []
                
                for row in result.fetchall():
                    paper_dict = self._safe_enhanced_row_to_dict(row)
                    if paper_dict:
                        papers.append(paper_dict)
                
                return {
                    'papers': papers,
                    'total': total_count,
                    'limit': limit,
                    'offset': offset,
                    'filters_applied': filters
                }
                
            except Exception as e:
                logger.error(f"❌ Enhanced filtered retrieval failed: {e}")
                return {
                    'papers': [],
                    'total': 0,
                    'limit': 0,
                    'offset': 0,
                    'error': str(e)
                }
            finally:
                session.close()
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.thread_pool, _get_filtered_sync)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, True, "filter")
        
        return result
    
    async def get_enhanced_database_stats(self) -> Dict[str, Any]:
        """Enhanced database statistics with multilingual and AI metrics"""
        start_time = time.time()
        
        def _get_stats_sync():
            session = self.SessionLocal()
            try:
                stats = {}
                
                # Basic counts
                total_papers = session.execute(text("SELECT COUNT(*) FROM research_papers")).scalar() or 0
                completed_papers = session.execute(text(
                    "SELECT COUNT(*) FROM research_papers WHERE processing_status = 'completed'"
                )).scalar() or 0
                total_relationships = session.execute(text("SELECT COUNT(*) FROM paper_relationships")).scalar() or 0
                
                stats['overview'] = {
                    'total_papers': total_papers,
                    'completed_papers': completed_papers,
                    'pending_papers': total_papers - completed_papers,
                    'total_relationships': total_relationships
                }
                
                # Language distribution
                lang_query = text("""
                    SELECT language, COUNT(*) as count 
                    FROM research_papers 
                    WHERE processing_status = 'completed'
                    GROUP BY language 
                    ORDER BY count DESC
                """)
                lang_result = session.execute(lang_query)
                stats['language_distribution'] = [
                    {"language": row[0], "count": row[1]} 
                    for row in lang_result.fetchall()
                ]
                
                # Domain distribution
                domain_query = text("""
                    SELECT research_domain, COUNT(*) as count 
                    FROM research_papers 
                    WHERE processing_status = 'completed'
                    GROUP BY research_domain 
                    ORDER BY count DESC LIMIT 10
                """)
                domain_result = session.execute(domain_query)
                stats['domain_distribution'] = [
                    {"domain": row[0], "count": row[1]} 
                    for row in domain_result.fetchall()
                ]
                
                # AI agent usage
                agent_query = text("""
                    SELECT ai_agent_used, COUNT(*) as count,
                           AVG(context_quality_score) as avg_quality,
                           AVG(analysis_confidence) as avg_confidence,
                           AVG(processing_time) as avg_time
                    FROM research_papers 
                    WHERE processing_status = 'completed' AND ai_agent_used IS NOT NULL
                    GROUP BY ai_agent_used 
                    ORDER BY count DESC
                """)
                agent_result = session.execute(agent_query)
                stats['ai_agent_usage'] = [
                    {
                        "agent": row[0],
                        "count": row[1],
                        "avg_quality": float(row[2]) if row[2] else 0.0,
                        "avg_confidence": float(row[3]) if row[3] else 0.0,
                        "avg_processing_time": float(row[4]) if row[4] else 0.0
                    }
                    for row in agent_result.fetchall()
                ]
                
                # Quality metrics - ✅ FIXED: All comparison operators
                quality_query = text("""
                SELECT 
                    AVG(context_quality_score) as avg_quality,
                    AVG(analysis_confidence) as avg_confidence,
                    COUNT(CASE WHEN context_quality_score > 0.8 THEN 1 END) as high_quality,
                    COUNT(CASE WHEN context_quality_score BETWEEN 0.5 AND 0.8 THEN 1 END) as medium_quality,
                    COUNT(CASE WHEN context_quality_score < 0.5 THEN 1 END) as low_quality
                FROM research_papers 
                WHERE processing_status = 'completed'
            """)
                quality_result = session.execute(quality_query).fetchone()
                if quality_result:
                    stats['quality_metrics'] = {
                        'average_quality_score': float(quality_result[0]) if quality_result[0] else 0.0,
                        'average_confidence': float(quality_result[1]) if quality_result[1] else 0.0,
                        'high_quality_papers': quality_result[2] or 0,
                        'medium_quality_papers': quality_result[3] or 0,
                        'low_quality_papers': quality_result[4] or 0
                    }
                
                # Relationship analytics
                rel_query = text("""
                    SELECT 
                        relationship_type, COUNT(*) as count,
                        AVG(relationship_strength) as avg_strength,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(CASE WHEN is_cross_linguistic = 1 THEN 1 END) as cross_linguistic
                    FROM paper_relationships
                    GROUP BY relationship_type
                    ORDER BY count DESC
                """)
                rel_result = session.execute(rel_query)
                stats['relationship_analytics'] = [
                    {
                        "type": row[0],
                        "count": row[1],
                        "avg_strength": float(row[2]) if row[2] else 0.0,
                        "avg_confidence": float(row[3]) if row[3] else 0.0,
                        "cross_linguistic": row[4] or 0
                    }
                    for row in rel_result.fetchall()
                ]
                
                # Vector and embedding stats
                vector_query = text("""
                    SELECT 
                        COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as papers_with_embeddings,
                        embedding_model, COUNT(*) as model_count
                    FROM research_papers 
                    WHERE processing_status = 'completed'
                    GROUP BY embedding_model
                """)
                vector_result = session.execute(vector_query)
                stats['vector_stats'] = [
                    {
                        "model": row[1] if row[1] else "unknown",
                        "count": row[2]
                    }
                    for row in vector_result.fetchall()
                ]
                
                # Recent activity - ✅ FIXED: >= instead of &gt;=
                recent_query = text("""
                SELECT COUNT(*) 
                FROM research_papers 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            """)
                recent_papers = session.execute(recent_query).scalar() or 0
                stats['recent_activity'] = {
                    'papers_24h': recent_papers
                }
                
                # Performance metrics from instance
                stats['performance'] = self.performance_stats
                
                stats['connection_status'] = 'healthy'
                stats['timestamp'] = datetime.now().isoformat()
                
                return stats
                
            except Exception as e:
                logger.error(f"❌ Enhanced stats retrieval failed: {e}")
                return {
                    'connection_status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            finally:
                session.close()
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(self.thread_pool, _get_stats_sync)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, True, "stats")
        
        return stats
    
    async def batch_store_papers(self, papers_data: List[Dict[str, Any]]) -> List[Optional[int]]:
        """Enhanced batch paper storage with improved performance"""
        start_time = time.time()
        
        def _batch_store_sync():
            if not papers_data:
                return []
            
            session = self.SessionLocal()
            paper_ids = []
            
            try:
                for i, paper_data in enumerate(papers_data):
                    if not paper_data.get('title'):
                        logger.warning(f"⚠️ Skipping paper {i} without title")
                        paper_ids.append(None)
                        continue
                    
                    # Check for existing paper
                    title = paper_data.get('title', '').strip()
                    existing_query = text("SELECT id FROM research_papers WHERE title = :title LIMIT 1")
                    existing = session.execute(existing_query, {"title": title}).fetchone()
                    
                    if existing:
                        paper_ids.append(existing[0])
                        continue
                    
                    # Prepare and insert new paper
                    clean_data = self._prepare_enhanced_paper_data(paper_data)
                    
                    insert_query = text("""
                        INSERT INTO research_papers (
                            title, abstract, authors, language, source, paper_url, pdf_url,
                            published_date, embedding, embedding_model, context_summary, research_domain,
                            methodology, key_findings, innovations, limitations,
                            research_questions, contributions, future_work, related_concepts,
                            context_quality_score, analysis_confidence, similarity_score,
                            search_language, detected_language, ai_agent_used, analysis_method,
                            search_method, translation_data, processing_time, processing_status
                        ) VALUES (
                            :title, :abstract, :authors, :language, :source, :paper_url, :pdf_url,
                            :published_date, :embedding, :embedding_model, :context_summary, :research_domain,
                            :methodology, :key_findings, :innovations, :limitations,
                            :research_questions, :contributions, :future_work, :related_concepts,
                            :context_quality_score, :analysis_confidence, :similarity_score,
                            :search_language, :detected_language, :ai_agent_used, :analysis_method,
                            :search_method, :translation_data, :processing_time, :processing_status
                        )
                    """)
                    
                    result = session.execute(insert_query, clean_data)
                    paper_ids.append(result.lastrowid)
                
                session.commit()
                successful_saves = sum(1 for pid in paper_ids if pid is not None)
                logger.info(f"✅ Enhanced batch stored {successful_saves}/{len(papers_data)} papers")
                
                return paper_ids
                
            except Exception as e:
                session.rollback()
                logger.error(f"❌ Enhanced batch store failed: {e}")
                return [None] * len(papers_data)
            finally:
                session.close()
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        paper_ids = await loop.run_in_executor(self.thread_pool, _batch_store_sync)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, True, "batch")
        
        return paper_ids
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        # Calculate derived metrics
        total_ops = self.performance_stats["total_queries"]
        success_rate = (
            self.performance_stats["successful_queries"] / total_ops 
            if total_ops > 0 else 0.0  # ✅ FIXED: > instead of &gt;
        )
        
        cache_total = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
        cache_hit_rate = (
            self.performance_stats["cache_hits"] / cache_total 
            if cache_total > 0 else 0.0  # ✅ FIXED: > instead of &gt;
        )
        
        return {
            **self.performance_stats,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.query_cache),
            "uptime_hours": (time.time() - self.performance_stats["last_reset"]) / 3600,
            "operations_per_second": total_ops / max((time.time() - self.performance_stats["last_reset"]), 1)
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_query_time": 0.0,
            "total_query_time": 0.0,
            "vector_operations": 0,
            "multilingual_operations": 0,
            "batch_operations": 0,
            "last_reset": time.time()
        }
        
        # Clear cache
        self.query_cache.clear()
        logger.info("📊 Enhanced performance statistics reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with enhanced metrics"""
        try:
            start_time = time.time()
            
            # Test basic connectivity
            def _test_connection():
                session = self.SessionLocal()
                try:
                    session.execute(text("SELECT 1")).fetchone()
                    return True
                except Exception as e:
                    logger.error(f"❌ Health check connection failed: {e}")
                    return False
                finally:
                    session.close()
            
            # Execute health checks
            loop = asyncio.get_event_loop()
            connection_healthy = await loop.run_in_executor(self.thread_pool, _test_connection)
            
            # Get basic stats
            stats = await self.get_enhanced_database_stats()
            performance = self.get_performance_stats()
            
            # Calculate health score
            health_components = {
                "connection": 1.0 if connection_healthy else 0.0,
                "success_rate": performance.get("success_rate", 0.0),
                "cache_efficiency": performance.get("cache_hit_rate", 0.0),
                "response_time": min(1.0, 2.0 / max(performance.get("average_query_time", 2.0), 0.1))
            }
            
            overall_health = sum(health_components.values()) / len(health_components)
            
            # Determine status - ✅ FIXED: > instead of &gt;
            if overall_health > 0.8:
                status = "excellent"
            elif overall_health > 0.6:
                status = "healthy"
            elif overall_health > 0.4:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return {
                "status": status,
                "health_score": overall_health,
                "health_components": health_components,
                "database_stats": stats.get("overview", {}),
                "performance_metrics": performance,
                "capabilities": {
                    "vector_operations": True,
                    "multilingual_support": True,
                    "async_operations": True,
                    "caching": True,
                    "batch_processing": True
                },
                "check_duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "health_score": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup_database(self, days_old: int = 30) -> Dict[str, int]:
        """Enhanced database cleanup with comprehensive statistics"""
        start_time = time.time()
        
        def _cleanup_sync():
            session = self.SessionLocal()
            try:
                # Clean up pending papers older than specified days - ✅ FIXED: < instead of &lt;
                cleanup_papers_query = text("""
                    DELETE FROM research_papers 
                    WHERE processing_status = 'pending' 
                    AND created_at < DATE_SUB(NOW(), INTERVAL :days DAY)
                """)
                papers_result = session.execute(cleanup_papers_query, {"days": days_old})
                
                # Clean up orphaned relationships
                cleanup_relationships_query = text("""
                    DELETE pr FROM paper_relationships pr
                    LEFT JOIN research_papers p1 ON pr.paper1_id = p1.id
                    LEFT JOIN research_papers p2 ON pr.paper2_id = p2.id
                    WHERE p1.id IS NULL OR p2.id IS NULL
                """)
                relationships_result = session.execute(cleanup_relationships_query)
                
                # Clean up old analytics data (keep last 90 days) - ✅ FIXED: < instead of &lt;
                cleanup_analytics_query = text("""
                    DELETE FROM system_analytics 
                    WHERE recorded_at < DATE_SUB(NOW(), INTERVAL 90 DAY)
                """)
                analytics_result = session.execute(cleanup_analytics_query)
                
                session.commit()
                
                return {
                    "papers_removed": papers_result.rowcount,
                    "relationships_removed": relationships_result.rowcount,
                    "analytics_records_removed": analytics_result.rowcount
                }
                
            except Exception as e:
                session.rollback()
                logger.error(f"❌ Enhanced cleanup failed: {e}")
                return {
                    "papers_removed": 0,
                    "relationships_removed": 0,
                    "analytics_records_removed": 0,
                    "error": str(e)
                }
            finally:
                session.close()
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        cleanup_stats = await loop.run_in_executor(self.thread_pool, _cleanup_sync)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, "error" not in cleanup_stats, "cleanup")
        
        cleanup_stats["cleanup_duration"] = processing_time
        logger.info(f"🧹 Enhanced cleanup completed: {cleanup_stats}")
        
        return cleanup_stats
    
    async def close(self):
        """Enhanced graceful shutdown"""
        try:
            logger.info("🔒 Enhanced TiDB Client shutdown initiated...")
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Final cache cleanup
            self._enhanced_cache_management()
            
            # Close database connections
            self.engine.dispose()
            
            # Log final performance stats
            perf_stats = self.get_performance_stats()
            logger.info(f"📈 Final performance: {perf_stats['success_rate']:.1%} success rate, "
                       f"{perf_stats['total_queries']} total operations, "
                       f"{perf_stats['cache_hit_rate']:.1%} cache hit rate")
            
            logger.info("✅ Enhanced TiDB Client shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Enhanced shutdown failed: {e}")
    
    def __del__(self):
        """Enhanced cleanup when client is destroyed"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
            if hasattr(self, 'engine'):
                self.engine.dispose()
            logger.debug("🔒 Enhanced TiDB Client cleanup completed")
        except:
            pass  # Ignore errors during cleanup

# Enhanced backward compatibility
TiDBClient = EnhancedTiDBClient

# Legacy compatibility functions
def get_database_stats() -> Dict[str, Any]:
    """Legacy compatibility function"""
    client = EnhancedTiDBClient()
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(client.get_enhanced_database_stats())
    except Exception as e:
        logger.error(f"❌ Legacy stats retrieval failed: {e}")
        return {"error": str(e)}
    finally:
        asyncio.run(client.close())

async def batch_store_papers_async(papers: List[Dict[str, Any]]) -> List[Optional[int]]:
    """Enhanced async batch storage function"""
    client = EnhancedTiDBClient()
    try:
        return await client.batch_store_papers(papers)
    finally:
        await client.close()
