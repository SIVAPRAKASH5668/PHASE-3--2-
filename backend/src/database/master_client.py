import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
import time

# Import your enhanced Crossref client
from integrations.crossref_client import EnhancedCrossrefClient
from config import settings

logger = logging.getLogger(__name__)

class MasterSearchClient:
    """
    🚀 **Simplified Master Search Client v1.0 - Crossref Focus**
    
    **Features:**
    - 📚 Crossref integration for high-quality journal articles
    - ⚡ Async search with error handling
    - 🔧 Simple deduplication and ranking
    - 📊 Performance tracking
    - 🛡️ Robust error recovery
    """
    
    def __init__(self, email: str = None):
        # Initialize only Crossref client for now
        self.crossref_client = EnhancedCrossrefClient(
            email=email or getattr(settings, 'CROSSREF_EMAIL', None)
        )
        
        # Simple source configuration
        self.enabled_sources = {
            'crossref': getattr(settings, 'ENABLE_CROSSREF_SEARCH', True)
        }
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'total_papers_found': 0,
            'average_search_time': 0.0,
            'last_reset': time.time()
        }
        
        logger.info("🎯 Master Search Client initialized with Crossref")
    
    async def comprehensive_search(self, query: str, languages: List[str] = None, 
                                 max_results: int = 50) -> List[Dict]:
        """
        🌟 **Comprehensive search using Crossref**
        
        Args:
            query: Search query
            languages: Target languages (for future expansion)
            max_results: Maximum number of results
            
        Returns:
            List of enhanced paper dictionaries
        """
        start_time = time.time()
        
        try:
            all_results = []
            active_sources = []
            search_tasks = []
            
            # Only Crossref search for now
            if self.enabled_sources.get('crossref', True):
                # For now, use query as-is (add translation later)
                english_query = query
                
                # Build Crossref search task
                search_tasks.append(('crossref', self.crossref_client.search_papers_async(
                    english_query,
                    max_results,
                    filters={
                        'type': 'journal-article',
                        'has-abstract': 'true',
                        'from-pub-date': '2020-01-01'  # Recent papers only
                    },
                    quality_threshold=0.3
                )))
                active_sources.append('crossref')
            
            if not active_sources:
                logger.warning("⚠️ No search sources enabled")
                return []
            
            logger.info(f"🔍 Starting search across {len(active_sources)} sources: {active_sources}")
            
            # Execute searches in parallel
            named_results = await asyncio.gather(
                *[task for _, task in search_tasks], 
                return_exceptions=True
            )
            
            # Process results
            for i, (source_name, result) in enumerate(zip([name for name, _ in search_tasks], named_results)):
                if isinstance(result, Exception):
                    logger.error(f"❌ {source_name} search failed: {result}")
                    continue
                
                if isinstance(result, list):
                    # Add source metadata to each paper
                    for paper in result:
                        paper['search_source'] = source_name
                        paper['multi_source_search'] = False  # Only one source for now
                        paper['search_timestamp'] = datetime.now().isoformat()
                    
                    all_results.extend(result)
                    logger.info(f"✅ {source_name}: {len(result)} papers")
                else:
                    logger.warning(f"⚠️ {source_name}: Unexpected result format")
            
            # Deduplication and ranking
            unique_results = self._cross_source_deduplication(all_results)
            ranked_results = self._multi_source_ranking(unique_results, query)
            
            # Update performance stats
            search_time = time.time() - start_time
            self._update_search_stats(search_time, True, len(ranked_results))
            
            logger.info(f"🎯 Search completed: {len(ranked_results)}/{len(all_results)} unique papers "
                       f"from {len(active_sources)} sources ({search_time:.2f}s)")
            
            return ranked_results[:max_results]
            
        except Exception as e:
            search_time = time.time() - start_time
            self._update_search_stats(search_time, False, 0)
            logger.error(f"❌ Comprehensive search failed: {e}")
            return []
    
    def _cross_source_deduplication(self, papers: List[Dict]) -> List[Dict]:
        """
        🔧 **Cross-source deduplication with DOI and title matching**
        """
        if not papers:
            return papers
        
        unique_papers = []
        seen_dois = set()
        seen_titles = set()
        
        for paper in papers:
            is_duplicate = False
            
            # Check DOI first (most reliable)
            doi = paper.get('metadata', {}).get('doi') or paper.get('doi', '')
            if doi:
                doi_clean = doi.strip().lower()
                if doi_clean in seen_dois:
                    is_duplicate = True
                else:
                    seen_dois.add(doi_clean)
            
            # Check title similarity if no DOI match
            if not is_duplicate:
                title = paper.get('title', '').strip().lower()
                if title:
                    # Simple title matching (can be enhanced later)
                    if title in seen_titles:
                        is_duplicate = True
                    else:
                        seen_titles.add(title)
            
            if not is_duplicate:
                unique_papers.append(paper)
        
        logger.info(f"🔧 Deduplication: {len(unique_papers)}/{len(papers)} unique papers")
        return unique_papers
    
    def _multi_source_ranking(self, papers: List[Dict], query: str) -> List[Dict]:
        """
        📊 **Multi-source ranking algorithm**
        """
        try:
            scored_papers = []
            
            for paper in papers:
                score = self._calculate_paper_score(paper, query)
                paper['relevance_score'] = score
                scored_papers.append(paper)
            
            # Sort by relevance score (descending)
            ranked_papers = sorted(scored_papers, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            logger.info(f"📊 Ranked {len(ranked_papers)} papers by relevance")
            return ranked_papers
            
        except Exception as e:
            logger.error(f"❌ Ranking failed: {e}")
            return papers
    
    def _calculate_paper_score(self, paper: Dict, query: str) -> float:
        """
        🎯 **Calculate relevance score for a paper**
        """
        try:
            score = 0.0
            
            # Base quality score
            quality = paper.get('computed_quality_score', paper.get('source_quality_score', 0.5))
            score += quality * 0.3
            
            # Title relevance
            title = paper.get('title', '').lower()
            query_words = set(query.lower().split())
            title_words = set(title.split())
            
            if query_words and title_words:
                title_match = len(query_words.intersection(title_words)) / len(query_words)
                score += title_match * 0.4
            
            # Abstract relevance
            abstract = paper.get('abstract', '').lower()
            if abstract:
                abstract_words = set(abstract.split())
                if query_words and abstract_words:
                    abstract_match = len(query_words.intersection(abstract_words)) / len(query_words)
                    score += abstract_match * 0.2
            
            # Citation bonus (if available)
            citations = paper.get('metadata', {}).get('citations', 0)
            if citations > 0:
                citation_score = min(0.1, citations / 100)  # Max 0.1 bonus
                score += citation_score
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"❌ Score calculation failed: {e}")
            return 0.5
    
    def _update_search_stats(self, search_time: float, success: bool, papers_count: int):
        """Update search performance statistics"""
        self.search_stats['total_searches'] += 1
        
        if success:
            self.search_stats['successful_searches'] += 1
            self.search_stats['total_papers_found'] += papers_count
            
            # Update average search time
            successful = self.search_stats['successful_searches']
            current_avg = self.search_stats['average_search_time']
            self.search_stats['average_search_time'] = (
                (current_avg * (successful - 1) + search_time) / successful
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        🏥 **Health check for all clients**
        """
        try:
            # Check Crossref health
            crossref_health = await self.crossref_client.health_check()
            
            overall_status = "healthy" if crossref_health.get('status') == 'healthy' else "degraded"
            
            return {
                'overall_status': overall_status,
                'services': {
                    'crossref': crossref_health
                },
                'search_stats': self.get_search_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'overall_status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        total_searches = self.search_stats['total_searches']
        success_rate = (
            self.search_stats['successful_searches'] / total_searches
            if total_searches > 0 else 0.0
        )
        
        return {
            **self.search_stats,
            'success_rate': success_rate,
            'enabled_sources': list(self.enabled_sources.keys()),
            'uptime_hours': (time.time() - self.search_stats['last_reset']) / 3600
        }
    
    def reset_stats(self):
        """Reset search statistics"""
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'total_papers_found': 0,
            'average_search_time': 0.0,
            'last_reset': time.time()
        }
        logger.info("📊 Search statistics reset")

# Usage example
async def main():
    """Example usage"""
    client = MasterSearchClient(email="your-email@example.com")
    
    results = await client.comprehensive_search(
        query="machine learning healthcare",
        languages=["en"],
        max_results=20
    )
    
    print(f"Found {len(results)} papers")
    for i, paper in enumerate(results[:3]):
        print(f"{i+1}. {paper.get('title', 'No title')}")
        print(f"   Source: {paper.get('search_source')}")
        print(f"   Quality: {paper.get('computed_quality_score', 0):.2f}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
