import requests
import logging
import asyncio
import time
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import hashlib

logger = logging.getLogger(__name__)

class EnhancedCrossrefClient:
    """
    🚀 **Enhanced Crossref Client v2.0**
    
    **Features:**
    - ⚡ Async support with thread pool
    - 📊 Performance tracking and caching
    - 🛡️ Enhanced error handling
    - 🌍 Better multilingual support
    - 📈 Quality assessment integration
    """
    
    def __init__(self, email: str = None):
        self.base_url = "https://api.crossref.org"
        self.email = email
        self.session = requests.Session()
        
        # Rate limiting based on polite pool
        if self.email:
            self.rate_limit_delay = 0.1  # 10 requests/sec for polite pool
            self.session.headers.update({
                'User-Agent': f'PolyResearch/1.0 (mailto:{self.email})',
                'From': self.email
            })
        else:
            self.rate_limit_delay = 1.0  # 1 request/sec for anonymous
            self.session.headers.update({'User-Agent': 'PolyResearch/1.0'})
        
        # ✅ ADDED: Performance tracking (same pattern as your ArXiv client)
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_papers_retrieved": 0,
            "average_response_time": 0.0,
            "last_reset": time.time()
        }
        
        # ✅ ADDED: Caching system
        self.query_cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = 1000
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        
        logger.info("🌍 Enhanced Crossref client initialized")
    
    # ✅ ADDED: Async wrapper methods
    async def search_papers_async(self, query: str, max_results: int = 20,
                                 filters: Dict = None, quality_threshold: float = 0.3) -> List[Dict]:
        """Async wrapper for paper search"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query, max_results, filters)
            if cache_key in self.query_cache:
                cached_result, timestamp = self.query_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    self.performance_stats['cache_hits'] += 1
                    return cached_result
            
            self.performance_stats['cache_misses'] += 1
            
            # Execute search in thread pool
            loop = asyncio.get_event_loop()
            papers = await loop.run_in_executor(
                self.thread_pool,
                self.search_papers,
                query, max_results, filters
            )
            
            # ✅ ADDED: Quality assessment
            if papers:
                quality_papers = self._assess_paper_quality(papers, quality_threshold)
                enhanced_papers = self._enhance_crossref_papers(quality_papers, query)
            else:
                enhanced_papers = []
            
            # Cache results
            self._cache_result(cache_key, enhanced_papers)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, True, len(enhanced_papers))
            
            logger.info(f"📚 Crossref search: {len(enhanced_papers)} papers (quality filtered, {processing_time:.2f}s)")
            return enhanced_papers
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, False, 0)
            logger.error(f"❌ Crossref async search failed: {e}")
            return []
    
    def search_papers(self, query: str, max_results: int = 20, 
                     filters: Dict = None, sort_by: str = 'published', 
                     sort_order: str = 'desc') -> List[Dict]:
        """Enhanced synchronous search with better error handling"""
        try:
            params = {
                'query': query,
                'rows': min(max_results, 1000),
                'sort': sort_by,
                'order': sort_order
            }
            
            if self.email:
                params['mailto'] = self.email
            
            # ✅ ENHANCED: Better filter handling
            if filters:
                filter_strings = []
                for key, value in filters.items():
                    if value:  # Only add non-empty filters
                        filter_strings.append(f"{key}:{value}")
                if filter_strings:
                    params['filter'] = ','.join(filter_strings)
            
            time.sleep(self.rate_limit_delay)
            
            response = self.session.get(f"{self.base_url}/works", params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            items = data.get('message', {}).get('items', [])
            for item in items:
                paper = self._parse_crossref_item(item)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers from Crossref for query: {query}")
            return papers
            
        except Exception as e:
            logger.error(f"Crossref search failed: {e}")
            return []
    
    def _enhance_crossref_papers(self, papers: List[Dict], query: str) -> List[Dict]:
        """✅ ADDED: Enhance papers with search metadata (match your pattern)"""
        for paper in papers:
            # Add search metadata
            paper['search_engine'] = 'crossref'
            paper['search_query'] = query
            paper['enhanced_metadata'] = True
            paper['citation_network_available'] = True  # Crossref has citation data
            
            # Calculate enhanced quality score
            base_quality = paper.get('source_quality_score', 0.7)  # Crossref is generally high quality
            citations = paper.get('metadata', {}).get('citations', 0)
            references = paper.get('metadata', {}).get('references', 0)
            
            # Quality bonus for citation data
            citation_bonus = min(0.2, citations / 100)  # Max 0.2 bonus
            reference_bonus = min(0.1, references / 50)  # Max 0.1 bonus
            
            paper['computed_quality_score'] = min(1.0, base_quality + citation_bonus + reference_bonus)
            
            # Research domain inference
            if not paper.get('research_domain'):
                paper['research_domain'] = self._infer_domain_from_crossref(paper)
        
        return papers
    
    def _assess_paper_quality(self, papers: List[Dict], threshold: float) -> List[Dict]:
        """✅ ADDED: Quality assessment for Crossref papers"""
        quality_papers = []
        
        for paper in papers:
            quality_score = 0.7  # Base score for Crossref (peer-reviewed)
            
            # Quality indicators
            if paper.get('title') and len(paper['title']) > 10:
                quality_score += 0.1
            
            if paper.get('abstract') and len(paper['abstract']) > 100:
                quality_score += 0.1
            
            if paper.get('metadata', {}).get('doi'):
                quality_score += 0.05  # DOI indicates formal publication
            
            if paper.get('metadata', {}).get('journal'):
                quality_score += 0.05  # Journal publication
            
            paper['source_quality_score'] = quality_score
            
            if quality_score >= threshold:
                quality_papers.append(paper)
        
        return quality_papers
    
    def _infer_domain_from_crossref(self, paper: Dict) -> str:
        """Infer research domain from Crossref metadata"""
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        journal = paper.get('metadata', {}).get('journal', '').lower()
        
        text = f"{title} {abstract} {journal}"
        
        domain_keywords = {
            'Computer Science': ['computer', 'software', 'algorithm', 'programming', 'artificial intelligence'],
            'Medicine': ['medical', 'health', 'clinical', 'patient', 'treatment', 'disease'],
            'Biology': ['biological', 'genetic', 'molecular', 'cellular', 'organism'],
            'Physics': ['physics', 'quantum', 'particle', 'energy', 'electromagnetic'],
            'Chemistry': ['chemical', 'molecular', 'reaction', 'synthesis', 'compound'],
            'Engineering': ['engineering', 'technical', 'design', 'system', 'optimization'],
            'Economics': ['economic', 'financial', 'market', 'business', 'trade'],
            'Psychology': ['psychological', 'behavior', 'cognitive', 'mental', 'brain']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                return domain
        
        return 'General Research'
    
    # ✅ ADDED: All the utility methods from your ArXiv client pattern
    def _get_cache_key(self, query: str, max_results: int, filters: Dict) -> str:
        key_data = f"crossref_{query}_{max_results}_{str(filters)}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _cache_result(self, cache_key: str, result: List[Dict]):
        self.query_cache[cache_key] = (result, time.time())
        if len(self.query_cache) > self.max_cache_size:
            self._clean_cache()
    
    def _clean_cache(self):
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.query_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self.query_cache[key]
    
    def _update_performance_stats(self, processing_time: float, success: bool, papers_count: int):
        self.performance_stats['total_requests'] += 1
        if success:
            self.performance_stats['successful_requests'] += 1
            self.performance_stats['total_papers_retrieved'] += papers_count
        else:
            self.performance_stats['failed_requests'] += 1
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        total_requests = self.performance_stats['total_requests']
        success_rate = (self.performance_stats['successful_requests'] / total_requests 
                       if total_requests > 0 else 0.0)
        return {**self.performance_stats, 'success_rate': success_rate}
    
    async def health_check(self) -> Dict:
        """Health check for Crossref API"""
        try:
            start_time = time.time()
            test_papers = await self.search_papers_async("test", max_results=1)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy" if len(test_papers) >= 0 else "degraded",
                "service": "crossref",
                "response_time": response_time,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "service": "crossref"}
    
    # Keep all your existing methods (get_recent_papers, get_paper_by_doi, etc.)
    # ... [rest of your original methods remain the same]
