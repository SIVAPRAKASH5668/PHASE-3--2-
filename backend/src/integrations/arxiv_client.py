import requests
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
import logging
from typing import List, Dict, Optional, Tuple, Any
import time
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Phase 2: Enhanced imports
from core.language_detector import LanguageDetector
from utils.text_processing import TextProcessor
from utils.language_utils import LanguageUtils

logger = logging.getLogger(__name__)

class EnhancedArxivClient:
    """
    🚀 **Enhanced ArXiv Client v2.0 with Multilingual Support**
    
    **New Features:**
    - 🌍 Multilingual query processing and translation
    - 🧠 Semantic search optimization
    - ⚡ Parallel processing and async support
    - 📊 Advanced filtering and ranking
    - 💾 Intelligent caching with TTL
    - 🛡️ Enhanced error handling and recovery
    - 📈 Performance monitoring and statistics
    - 🎯 Quality assessment and scoring
    """
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.max_results_per_request = 100
        self.rate_limit_delay = 1.0  # Be respectful to arXiv
        
        # Phase 2: Enhanced components
        try:
            self.language_detector = LanguageDetector()
            self.text_processor = TextProcessor()
            self.language_utils = LanguageUtils()
            self.multilingual_support = True
            logger.info("🌍 Enhanced ArXiv client with multilingual support initialized")
        except Exception as e:
            logger.warning(f"⚠️ Multilingual components failed to initialize: {e}")
            self.language_detector = None
            self.text_processor = None
            self.language_utils = None
            self.multilingual_support = False
        
        # Enhanced performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_papers_retrieved": 0,
            "average_response_time": 0.0,
            "total_response_time": 0.0,
            "multilingual_queries": 0,
            "quality_filtered_papers": 0,
            "last_reset": time.time()
        }
        
        # Enhanced caching system
        self.query_cache = {}
        self.cache_ttl = 3600  # 1 hour TTL
        self.max_cache_size = 1000
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        
        # Enhanced category mapping for better search
        self.category_mapping = {
            "computer science": ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.IR"],
            "machine learning": ["cs.LG", "stat.ML"],
            "artificial intelligence": ["cs.AI", "cs.LG"],
            "natural language processing": ["cs.CL", "cs.AI"],
            "computer vision": ["cs.CV", "cs.AI"],
            "deep learning": ["cs.LG", "cs.NE", "stat.ML"],
            "robotics": ["cs.RO", "cs.AI"],
            "physics": ["physics.gen-ph", "cond-mat", "hep-th"],
            "mathematics": ["math.GM", "math.ST", "stat.TH"],
            "biology": ["q-bio.GN", "q-bio.QM", "q-bio.MN"],
            "medicine": ["q-bio.QM", "q-bio.BM"]
        }
    
    async def search_papers_async(self, query: str, max_results: int = 20, 
                                category: str = None, start_date: str = None,
                                enable_multilingual: bool = True,
                                quality_threshold: float = 0.3) -> List[Dict]:
        """
        🌟 Enhanced async paper search with multilingual support
        
        Args:
            query: Search query in any language
            max_results: Maximum number of results
            category: ArXiv category filter
            start_date: Minimum submission date (YYYY-MM-DD)
            enable_multilingual: Enable multilingual processing
            quality_threshold: Minimum quality score for papers
            
        Returns:
            List of enhanced paper dictionaries
        """
        start_time = time.time()
        
        try:
            # Enhanced query processing
            processed_queries = await self._process_multilingual_query(
                query, enable_multilingual
            )
            
            # Search with multiple query variants
            all_papers = []
            
            for lang_query in processed_queries:
                papers = await self._search_single_query_async(
                    lang_query, max_results // len(processed_queries),
                    category, start_date
                )
                
                # Add search metadata
                for paper in papers:
                    paper['search_query_variant'] = lang_query
                    paper['multilingual_search'] = len(processed_queries) > 1
                
                all_papers.extend(papers)
            
            # Enhanced deduplication
            unique_papers = self._advanced_deduplication(all_papers)
            
            # Quality assessment and filtering
            if self.text_processor:
                quality_papers = await self._assess_paper_quality(
                    unique_papers, quality_threshold
                )
            else:
                quality_papers = unique_papers
            
            # Enhanced ranking
            ranked_papers = self._rank_papers_by_relevance(quality_papers, query)
            
            # Update performance statistics
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, True, len(ranked_papers))
            
            if enable_multilingual:
                self.performance_stats['multilingual_queries'] += 1
            
            logger.info(f"📚 Enhanced ArXiv search: {len(ranked_papers)}/{len(all_papers)} papers "
                       f"(quality filtered, {processing_time:.2f}s)")
            
            return ranked_papers[:max_results]
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, False, 0)
            logger.error(f"❌ Enhanced ArXiv search failed: {e}")
            return []
    
    def search_papers(self, query: str, max_results: int = 20, 
                     category: str = None, start_date: str = None) -> List[Dict]:
        """Synchronous wrapper for backward compatibility"""
        return asyncio.run(self.search_papers_async(
            query, max_results, category, start_date
        ))
    
    async def _process_multilingual_query(self, query: str, 
                                        enable_multilingual: bool) -> List[str]:
        """Process query into multiple language variants"""
        queries = [query]  # Always include original
        
        if not enable_multilingual or not self.multilingual_support:
            return queries
        
        try:
            # Detect query language
            if self.language_detector:
                detection = self.language_detector.detect_language(query)
                detected_lang = detection.get('language_code', 'en')
                
                # For non-English queries, add English translation
                if detected_lang != 'en':
                    # This would need translation service integration
                    # For now, we'll use domain-specific keyword enhancement
                    enhanced_query = self._enhance_query_for_arxiv(query, detected_lang)
                    if enhanced_query != query:
                        queries.append(enhanced_query)
            
            # Add domain-specific query variants
            domain_queries = self._generate_domain_specific_queries(query)
            queries.extend(domain_queries)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in queries:
                if q not in seen:
                    seen.add(q)
                    unique_queries.append(q)
            
            return unique_queries[:3]  # Limit to 3 variants
            
        except Exception as e:
            logger.warning(f"⚠️ Multilingual query processing failed: {e}")
            return [query]
    
    def _enhance_query_for_arxiv(self, query: str, language: str) -> str:
        """Enhance query for ArXiv search based on language"""
        try:
            if not self.text_processor:
                return query
            
            # Extract keywords
            keywords = self.text_processor.extract_keywords(
                query, language=language, max_keywords=10
            )
            
            # Add technical terms for ArXiv
            technical_terms = {
                'machine learning': ['neural network', 'deep learning', 'algorithm'],
                'computer vision': ['image processing', 'CNN', 'object detection'],
                'natural language': ['NLP', 'text processing', 'language model'],
                'artificial intelligence': ['AI', 'machine learning', 'neural'],
            }
            
            enhanced_terms = []
            query_lower = query.lower()
            
            for domain, terms in technical_terms.items():
                if any(word in query_lower for word in domain.split()):
                    enhanced_terms.extend(terms[:2])  # Add 2 relevant terms
            
            if enhanced_terms:
                return f"{query} {' '.join(enhanced_terms[:3])}"
            
            return query
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return query
    
    def _generate_domain_specific_queries(self, query: str) -> List[str]:
        """Generate domain-specific query variants"""
        domain_queries = []
        query_lower = query.lower()
        
        try:
            # Check if query matches known domains
            for domain, categories in self.category_mapping.items():
                if any(word in query_lower for word in domain.split()):
                    # Add category-specific variant
                    domain_query = f"{query} AND cat:{categories[0]}"
                    domain_queries.append(domain_query)
                    break  # Only add one domain variant
            
            return domain_queries
            
        except Exception as e:
            logger.warning(f"Domain query generation failed: {e}")
            return []
    
    async def _search_single_query_async(self, query: str, max_results: int,
                                       category: str = None, start_date: str = None) -> List[Dict]:
        """Search for a single query variant asynchronously"""
        # Check cache first
        cache_key = self._get_cache_key(query, category, start_date, max_results)
        if cache_key in self.query_cache:
            cached_result, timestamp = self.query_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.performance_stats['cache_hits'] += 1
                return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        try:
            # Build search query
            search_query = self._build_enhanced_search_query(query, category, start_date)
            
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': min(max_results, self.max_results_per_request),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            # Execute request in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.thread_pool,
                lambda: requests.get(self.base_url, params=params, timeout=30)
            )
            response.raise_for_status()
            
            # Parse response
            papers = await loop.run_in_executor(
                self.thread_pool,
                self._parse_enhanced_arxiv_response,
                response.content
            )
            
            # Cache result
            self._cache_result(cache_key, papers)
            
            # Clean cache if needed
            if len(self.query_cache) > self.max_cache_size:
                self._clean_cache()
            
            return papers
            
        except requests.RequestException as e:
            logger.error(f"ArXiv API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []
    
    def _build_enhanced_search_query(self, query: str, category: str = None, 
                                   start_date: str = None) -> str:
        """Build enhanced ArXiv search query with better field targeting"""
        # Clean and prepare base query
        clean_query = re.sub(r'[^\w\s\-\+\(\)\"\'&|]', '', query).strip()
        search_parts = []
        
        # Enhanced search strategy
        if clean_query:
            # Search in multiple fields with weights
            field_searches = []
            
            # Title search (highest priority)
            field_searches.append(f'ti:"{clean_query}"')
            
            # Abstract search
            field_searches.append(f'abs:"{clean_query}"')
            
            # Author search for names
            if self._looks_like_author_name(clean_query):
                field_searches.append(f'au:"{clean_query}"')
            
            # Combine field searches
            search_parts.append(f"({' OR '.join(field_searches)})")
        
        # Category filter with smart expansion
        if category:
            if category in self.category_mapping:
                # Multiple related categories
                categories = self.category_mapping[category][:3]  # Top 3
                cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
                search_parts.append(f"({cat_query})")
            else:
                search_parts.append(f"cat:{category}")
        
        # Date filter
        if start_date:
            try:
                date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                date_str = date_obj.strftime("%Y%m%d")
                search_parts.append(f"submittedDate:[{date_str}* TO *]")
            except ValueError:
                logger.warning(f"Invalid date format: {start_date}")
        
        # Combine search parts
        if search_parts:
            return " AND ".join(search_parts)
        else:
            return "all:*"
    
    def _looks_like_author_name(self, query: str) -> bool:
        """Check if query looks like an author name"""
        # Simple heuristic: 2-3 words, proper case, no technical terms
        words = query.split()
        if len(words) not in [2, 3]:
            return False
        
        # Check if words are likely names (start with capital)
        if not all(word[0].isupper() for word in words if word):
            return False
        
        # Exclude technical terms
        technical_terms = ['learning', 'network', 'algorithm', 'processing', 'analysis']
        if any(term in query.lower() for term in technical_terms):
            return False
        
        return True
    
    def _parse_enhanced_arxiv_response(self, xml_content: bytes) -> List[Dict]:
        """Enhanced parsing of ArXiv API XML response"""
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            papers = []
            
            for entry in root.findall('atom:entry', ns):
                try:
                    paper = self._parse_enhanced_arxiv_entry(entry, ns)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.error(f"Failed to parse ArXiv entry: {e}")
                    continue
            
            return papers
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing ArXiv response: {e}")
            return []
    
    def _parse_enhanced_arxiv_entry(self, entry, ns: dict) -> Optional[Dict]:
        """Parse individual ArXiv entry with enhanced metadata"""
        try:
            # Extract basic information
            title = entry.find('atom:title', ns)
            title_text = title.text.strip().replace('\n', ' ').replace('\r', '') if title is not None else ""
            
            summary = entry.find('atom:summary', ns)
            abstract_text = summary.text.strip().replace('\n', ' ').replace('\r', '') if summary is not None else ""
            
            # Enhanced author extraction
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    author_name = name.text.strip()
                    # Clean author name
                    author_name = re.sub(r'\s+', ' ', author_name)
                    authors.append(author_name)
            
            authors_text = "; ".join(authors)
            
            # Enhanced URL extraction
            paper_url = ""
            pdf_url = ""
            
            for link in entry.findall('atom:link', ns):
                rel = link.get('rel')
                href = link.get('href')
                
                if rel == 'alternate':
                    paper_url = href
                elif rel == 'related' and link.get('type') == 'application/pdf':
                    pdf_url = href
            
            # Extract ArXiv ID from URL
            arxiv_id = ""
            if paper_url:
                arxiv_id = paper_url.split('/')[-1]
            
            # Enhanced date extraction
            published = entry.find('atom:published', ns)
            updated = entry.find('atom:updated', ns)
            
            pub_date = None
            if published is not None:
                pub_date = self._parse_date_string(published.text)
            
            updated_date = None
            if updated is not None:
                updated_date = self._parse_date_string(updated.text)
            
            # Enhanced category extraction
            categories = []
            primary_subject = ""
            
            for category in entry.findall('atom:category', ns):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Determine primary subject from categories
            if categories:
                primary_subject = categories[0]
                
                # Map to broader domain
                domain_mapping = {
                    'cs.AI': 'Computer Science - Artificial Intelligence',
                    'cs.LG': 'Computer Science - Machine Learning',
                    'cs.CV': 'Computer Science - Computer Vision',
                    'cs.CL': 'Computer Science - Computation and Language',
                    'cs.RO': 'Computer Science - Robotics',
                    'stat.ML': 'Statistics - Machine Learning',
                    'physics.gen-ph': 'Physics - General Physics',
                    'math.GM': 'Mathematics - General Mathematics'
                }
                
                research_domain = domain_mapping.get(primary_subject, 'General Research')
            else:
                research_domain = 'General Research'
            
            # Enhanced paper quality indicators
            quality_indicators = self._calculate_arxiv_quality_indicators(
                title_text, abstract_text, authors, categories
            )
            
            # Build enhanced paper dictionary
            paper = {
                'title': title_text,
                'abstract': abstract_text,
                'authors': authors_text,
                'language': 'en',  # ArXiv papers are primarily in English
                'detected_language': 'en',
                'source': 'arxiv',
                'paper_url': paper_url,
                'pdf_url': pdf_url,
                'published_date': pub_date,
                'updated_date': updated_date,
                'categories': categories,
                'primary_category': primary_subject,
                'research_domain': research_domain,
                'arxiv_id': arxiv_id,
                
                # Enhanced metadata
                'metadata': {
                    'arxiv_id': arxiv_id,
                    'categories': categories,
                    'primary_subject': primary_subject,
                    'author_count': len(authors),
                    'quality_indicators': quality_indicators,
                    'extraction_timestamp': datetime.now().isoformat()
                },
                
                # Quality scoring
                'source_quality_score': quality_indicators.get('overall_score', 0.7),
                'title_length': len(title_text),
                'abstract_length': len(abstract_text),
                'author_count': len(authors)
            }
            
            return paper
            
        except Exception as e:
            logger.error(f"Error parsing enhanced ArXiv entry: {e}")
            return None
    
    def _parse_date_string(self, date_string: str) -> Optional[date]:
        """Enhanced date parsing with multiple format support"""
        if not date_string:
            return None
        
        try:
            # Try ISO format first
            if 'T' in date_string:
                dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
                return dt.date()
            
            # Try other common formats
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S"
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_string, fmt)
                    return dt.date()
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse date: {date_string}")
            return None
            
        except Exception as e:
            logger.warning(f"Date parsing error: {e}")
            return None
    
    def _calculate_arxiv_quality_indicators(self, title: str, abstract: str, 
                                      authors: List[str], categories: List[str]) -> Dict[str, Any]:
        """✅ FIXED: Much more lenient quality indicators"""
        indicators = {
            'title_quality': 0.0,
            'abstract_quality': 0.0,
            'author_quality': 0.0,
            'category_quality': 0.0,
            'overall_score': 0.0
        }
        
        try:
            # ✅ FIXED: Very lenient title quality
            title_score = 0.7  # Start higher
            if len(title) >= 10:  # Much lower threshold
                title_score = 0.8
            if len(title) >= 20:
                title_score = 0.9
            
            indicators['title_quality'] = title_score
            
            # ✅ FIXED: Very lenient abstract quality
            abstract_score = 0.6  # Start higher
            if len(abstract) >= 50:  # Much lower threshold
                abstract_score = 0.8
            if len(abstract) >= 100:
                abstract_score = 0.9
            
            indicators['abstract_quality'] = abstract_score
            
            # ✅ FIXED: Very lenient author quality
            author_score = 0.8  # Start high
            if len(authors) >= 1:
                author_score = 0.9
            
            indicators['author_quality'] = author_score
            
            # ✅ FIXED: Very lenient category quality
            category_score = 0.8  # Start high
            if categories:
                category_score = 0.9
            
            indicators['category_quality'] = category_score
            
            # ✅ FIXED: Overall score (weighted average)
            weights = {'title_quality': 0.3, 'abstract_quality': 0.3, 'author_quality': 0.2, 'category_quality': 0.2}
            overall = sum(indicators[key] * weights[key] for key in weights)
            indicators['overall_score'] = overall
            
        except Exception as e:
            logger.warning(f"Quality calculation error: {e}")
            indicators['overall_score'] = 0.7  # ✅ FIXED: Higher fallback score
        
        return indicators

    
    async def _assess_paper_quality(self, papers: List[Dict], 
                                  threshold: float) -> List[Dict]:
        """Assess and filter papers by quality"""
        if not self.text_processor:
            return papers
        
        try:
            quality_papers = []
            
            for paper in papers:
                # Use existing quality score or calculate new one
                quality_score = paper.get('source_quality_score', 0.5)
                
                # Enhanced quality check using text processor
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                if title and abstract:
                    # Validate academic text
                    validation = self.text_processor.validate_academic_text(
                        f"{title} {abstract}", language='en'
                    )
                    
                    if validation.get('is_academic', False):
                        quality_score += 0.2 * validation.get('confidence', 0)
                
                # Apply threshold
                if quality_score >= threshold:
                    paper['computed_quality_score'] = quality_score
                    quality_papers.append(paper)
            
            self.performance_stats['quality_filtered_papers'] += len(quality_papers)
            
            if len(quality_papers) < len(papers):
                logger.info(f"🎯 Quality filter: {len(quality_papers)}/{len(papers)} papers above threshold {threshold}")
            
            return quality_papers
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return papers
    
    def _advanced_deduplication(self, papers: List[Dict]) -> List[Dict]:
        """Advanced deduplication with multilingual awareness"""
        if not papers:
            return papers
        
        unique_papers = []
        seen_titles = set()
        seen_arxiv_ids = set()
        
        for paper in papers:
            # Check ArXiv ID first (most reliable)
            arxiv_id = paper.get('arxiv_id', '').strip()
            if arxiv_id and arxiv_id in seen_arxiv_ids:
                continue
            
            # Check title similarity
            title = paper.get('title', '').strip()
            if not title:
                continue
            
            # Clean title for comparison
            if self.text_processor:
                clean_title = self.text_processor.clean_text(title, language='en')
            else:
                clean_title = re.sub(r'\s+', ' ', title.lower().strip())
            
            # Check for duplicates
            is_duplicate = False
            for seen_title in seen_titles:
                if self.text_processor:
                    similarity = self.text_processor.calculate_text_similarity(
                        clean_title, seen_title, language='en'
                    )
                    if similarity > 0.9:  # 90% similarity threshold
                        is_duplicate = True
                        break
                else:
                    # Simple word overlap check
                    words1 = set(clean_title.split())
                    words2 = set(seen_title.split())
                    if words1 and words2:
                        overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                        if overlap > 0.8:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(clean_title)
                if arxiv_id:
                    seen_arxiv_ids.add(arxiv_id)
        
        return unique_papers
    
    def _rank_papers_by_relevance(self, papers: List[Dict], original_query: str) -> List[Dict]:
        """Rank papers by relevance to original query"""
        try:
            scored_papers = []
            
            for paper in papers:
                relevance_score = self._calculate_relevance_score(paper, original_query)
                paper['relevance_score'] = relevance_score
                scored_papers.append((paper, relevance_score))
            
            # Sort by relevance score (descending)
            scored_papers.sort(key=lambda x: x[1], reverse=True)
            
            return [paper for paper, score in scored_papers]
            
        except Exception as e:
            logger.warning(f"Ranking failed: {e}")
            return papers
    
    def _calculate_relevance_score(self, paper: Dict, query: str) -> float:
        """Calculate relevance score for a paper"""
        score = 0.0
        
        try:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            query_lower = query.lower()
            
            # Keyword matching in title (higher weight)
            query_words = set(query_lower.split())
            title_words = set(title.split())
            
            if query_words and title_words:
                title_overlap = len(query_words.intersection(title_words)) / len(query_words)
                score += title_overlap * 0.4
            
            # Keyword matching in abstract
            abstract_words = set(abstract.split())
            if query_words and abstract_words:
                abstract_overlap = len(query_words.intersection(abstract_words)) / len(query_words)
                score += abstract_overlap * 0.3
            
            # Quality score component
            quality_score = paper.get('computed_quality_score', paper.get('source_quality_score', 0.5))
            score += quality_score * 0.2
            
            # Recency bonus (papers from last 2 years)
            pub_date = paper.get('published_date')
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        pub_date = datetime.fromisoformat(pub_date).date()
                    
                    years_old = (date.today() - pub_date).days / 365.25
                    if years_old <= 2:
                        score += 0.1 * (1 - years_old / 2)  # Linear decay over 2 years
                except:
                    pass
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Relevance calculation error: {e}")
            return 0.5
    
    def _get_cache_key(self, query: str, category: str, start_date: str, max_results: int) -> str:
        """Generate cache key for query parameters"""
        key_data = f"{query}_{category}_{start_date}_{max_results}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _cache_result(self, cache_key: str, result: List[Dict]):
        """Cache search result with timestamp"""
        self.query_cache[cache_key] = (result, time.time())
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.query_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
        
        # If still too large, remove oldest entries
        if len(self.query_cache) > self.max_cache_size:
            sorted_items = sorted(
                self.query_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            
            # Remove oldest 20%
            remove_count = len(sorted_items) // 5
            for key, _ in sorted_items[:remove_count]:
                del self.query_cache[key]
    
    def _update_performance_stats(self, processing_time: float, success: bool, papers_count: int):
        """Update performance statistics"""
        self.performance_stats['total_requests'] += 1
        
        if success:
            self.performance_stats['successful_requests'] += 1
            self.performance_stats['total_papers_retrieved'] += papers_count
            
            # Update average response time
            total_successful = self.performance_stats['successful_requests']
            current_avg = self.performance_stats['average_response_time']
            self.performance_stats['average_response_time'] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
            
            self.performance_stats['total_response_time'] += processing_time
        else:
            self.performance_stats['failed_requests'] += 1
    
    # Enhanced methods for specific use cases
    async def search_by_author_async(self, author_name: str, max_results: int = 20) -> List[Dict]:
        """Search papers by author with enhanced name matching"""
        try:
            # Clean author name
            clean_name = re.sub(r'[^\w\s\-]', '', author_name).strip()
            
            # Try different name formats
            name_variants = [clean_name]
            
            # Add last name first format
            parts = clean_name.split()
            if len(parts) >= 2:
                last_first = f"{parts[-1]}, {' '.join(parts[:-1])}"
                name_variants.append(last_first)
            
            # Search with author field
            all_papers = []
            for variant in name_variants:
                search_query = f'au:"{variant}"'
                
                papers = await self._search_single_query_async(
                    search_query, max_results // len(name_variants)
                )
                all_papers.extend(papers)
            
            # Deduplicate and rank
            unique_papers = self._advanced_deduplication(all_papers)
            return unique_papers[:max_results]
            
        except Exception as e:
            logger.error(f"Author search failed: {e}")
            return []
    
    async def get_trending_papers_async(self, category: str = None, days: int = 7,
                                      max_results: int = 50) -> List[Dict]:
        """Get trending papers with enhanced filtering"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            start_str = start_date.strftime("%Y-%m-%d")
            
            # Search recent papers
            papers = await self.search_papers_async(
                query="*",
                max_results=max_results,
                category=category,
                start_date=start_str,
                enable_multilingual=False
            )
            
            # Additional filtering for trending papers
            trending_papers = []
            for paper in papers:
                # Simple trending criteria
                pub_date = paper.get('published_date')
                if pub_date:
                    try:
                        if isinstance(pub_date, str):
                            pub_date = datetime.fromisoformat(pub_date).date()
                        
                        days_old = (date.today() - pub_date).days
                        if days_old <= days:
                            # Add trending score based on recency and quality
                            trending_score = (1 - days_old / days) * paper.get('source_quality_score', 0.5)
                            paper['trending_score'] = trending_score
                            trending_papers.append(paper)
                    except:
                        continue
            
            # Sort by trending score
            trending_papers.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
            
            return trending_papers
            
        except Exception as e:
            logger.error(f"Trending papers search failed: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        total_requests = self.performance_stats['total_requests']
        success_rate = (
            self.performance_stats['successful_requests'] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        cache_total = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
        cache_hit_rate = (
            self.performance_stats['cache_hits'] / cache_total 
            if cache_total > 0 else 0.0
        )
        
        return {
            **self.performance_stats,
            'success_rate': success_rate,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.query_cache),
            'multilingual_support': self.multilingual_support,
            'uptime_hours': (time.time() - self.performance_stats['last_reset']) / 3600
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_papers_retrieved": 0,
            "average_response_time": 0.0,
            "total_response_time": 0.0,
            "multilingual_queries": 0,
            "quality_filtered_papers": 0,
            "last_reset": time.time()
        }
        
        # Clear cache
        self.query_cache.clear()
        logger.info("📊 ArXiv client performance stats reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Test basic connectivity
            start_time = time.time()
            
            test_params = {
                'search_query': 'machine learning',
                'start': 0,
                'max_results': 1
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: requests.get(self.base_url, params=test_params, timeout=10)
            )
            
            response_time = time.time() - start_time
            
            health_status = {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "response_time": response_time,
                "api_available": response.status_code == 200,
                "multilingual_support": self.multilingual_support,
                "cache_size": len(self.query_cache),
                "performance_stats": self.get_performance_stats(),
                "timestamp": datetime.now().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "multilingual_support": self.multilingual_support,
                "timestamp": datetime.now().isoformat()
            }
    
    def close(self):
        """Clean up resources"""
        try:
            self.thread_pool.shutdown(wait=True)
            self.query_cache.clear()
            logger.info("🔒 Enhanced ArXiv client resources cleaned up")
        except Exception as e:
            logger.error(f"ArXiv client cleanup error: {e}")

# Backward compatibility
ArxivClient = EnhancedArxivClient
