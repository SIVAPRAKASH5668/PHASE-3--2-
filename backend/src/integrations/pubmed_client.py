import requests
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
import logging
from typing import List, Dict, Optional, Any, Tuple
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

class EnhancedPubMedClient:
    """
    🚀 **Enhanced PubMed Client v2.0 with Medical Intelligence**
    
    **New Features:**
    - 🏥 Medical domain expertise and terminology
    - 🌍 Multilingual medical query processing
    - 🧠 Clinical research categorization
    - ⚡ Parallel processing and async support
    - 📊 Medical quality scoring and validation
    - 💾 Intelligent caching with medical context
    - 🛡️ Enhanced error handling and recovery
    - 📈 Performance monitoring and statistics
    - 🎯 Evidence-based ranking and scoring
    """
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.search_url = f"{self.base_url}/esearch.fcgi"
        self.fetch_url = f"{self.base_url}/efetch.fcgi"
        self.rate_limit_delay = 0.34  # PubMed rate limit: 3 requests per second
        
        # Phase 2: Enhanced components
        try:
            self.language_detector = LanguageDetector()
            self.text_processor = TextProcessor()
            self.language_utils = LanguageUtils()
            self.multilingual_support = True
            logger.info("🏥 Enhanced PubMed client with medical intelligence initialized")
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
            "medical_queries": 0,
            "clinical_papers": 0,
            "evidence_based_papers": 0,
            "last_reset": time.time()
        }
        
        # Enhanced caching system
        self.query_cache = {}
        self.cache_ttl = 1800  # 30 minutes TTL (shorter for medical data)
        self.max_cache_size = 500  # Smaller cache for medical precision
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)  # Respect PubMed rate limits
        
        # Medical domain expertise
        self.medical_terminologies = {
            "disease": ["disease", "disorder", "syndrome", "condition", "pathology"],
            "treatment": ["treatment", "therapy", "intervention", "medication", "drug"],
            "diagnosis": ["diagnosis", "diagnostic", "screening", "test", "biomarker"],
            "prevention": ["prevention", "preventive", "prophylaxis", "vaccination", "immunization"],
            "clinical_trial": ["clinical trial", "randomized", "controlled", "RCT", "trial"],
            "systematic_review": ["systematic review", "meta-analysis", "review", "evidence"],
            "case_study": ["case study", "case report", "case series"],
            "epidemiology": ["epidemiology", "prevalence", "incidence", "risk factors"]
        }
        
        # Publication types for quality assessment
        self.high_quality_publication_types = [
            "Clinical Trial", "Randomized Controlled Trial", "Meta-Analysis",
            "Systematic Review", "Practice Guideline", "Multicenter Study"
        ]
        
        self.medium_quality_publication_types = [
            "Comparative Study", "Controlled Clinical Trial", "Journal Article",
            "Observational Study", "Cohort Study", "Case-Control Study"
        ]
    
    async def search_papers_async(self, query: str, max_results: int = 20,
                                start_date: str = None, end_date: str = None,
                                enable_multilingual: bool = True,
                                medical_focus: bool = True,
                                evidence_level: str = "all") -> List[Dict]:
        """
        🌟 Enhanced async medical paper search
        
        Args:
            query: Medical search query
            max_results: Maximum number of results
            start_date: Start date (YYYY/MM/DD)
            end_date: End date (YYYY/MM/DD)
            enable_multilingual: Enable multilingual processing
            medical_focus: Focus on medical terminology
            evidence_level: "high", "medium", "all"
            
        Returns:
            List of enhanced medical paper dictionaries
        """
        start_time = time.time()
        
        try:
            # Enhanced medical query processing
            processed_queries = await self._process_medical_multilingual_query(
                query, enable_multilingual, medical_focus
            )
            
            # Search with medical query variants
            all_paper_ids = []
            
            for medical_query in processed_queries:
                paper_ids = await self._search_paper_ids_async(
                    medical_query, max_results // len(processed_queries),
                    start_date, end_date, evidence_level
                )
                all_paper_ids.extend(paper_ids)
            
            # Remove duplicate IDs
            unique_ids = list(dict.fromkeys(all_paper_ids))  # Preserves order
            
            if not unique_ids:
                return []
            
            # Fetch enhanced paper details
            papers = await self._fetch_enhanced_paper_details(unique_ids)
            
            # Medical quality assessment
            if self.text_processor and medical_focus:
                papers = await self._assess_medical_quality(papers)
            
            # Rank by medical relevance
            ranked_papers = self._rank_papers_by_medical_relevance(papers, query)
            
            # Update performance statistics
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, True, len(ranked_papers))
            
            if medical_focus:
                self.performance_stats['medical_queries'] += 1
            
            # Count clinical and evidence-based papers
            clinical_count = len([p for p in ranked_papers if p.get('is_clinical', False)])
            evidence_count = len([p for p in ranked_papers if p.get('evidence_level', 'low') in ['high', 'medium']])
            
            self.performance_stats['clinical_papers'] += clinical_count
            self.performance_stats['evidence_based_papers'] += evidence_count
            
            logger.info(f"🏥 Enhanced PubMed search: {len(ranked_papers)} papers "
                       f"({clinical_count} clinical, {evidence_count} evidence-based, {processing_time:.2f}s)")
            
            return ranked_papers[:max_results]
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, False, 0)
            logger.error(f"❌ Enhanced PubMed search failed: {e}")
            return []
    
    def search_papers(self, query: str, max_results: int = 20,
                     start_date: str = None, end_date: str = None) -> List[Dict]:
        """Synchronous wrapper for backward compatibility"""
        return asyncio.run(self.search_papers_async(
            query, max_results, start_date, end_date
        ))
    
    async def _process_medical_multilingual_query(self, query: str, 
                                                enable_multilingual: bool,
                                                medical_focus: bool) -> List[str]:
        """Process query into medical and multilingual variants"""
        queries = [query]  # Always include original
        
        try:
            # Medical terminology enhancement
            if medical_focus:
                enhanced_query = self._enhance_medical_query(query)
                if enhanced_query != query:
                    queries.append(enhanced_query)
            
            # Multilingual processing
            if enable_multilingual and self.multilingual_support:
                if self.language_detector:
                    detection = self.language_detector.detect_language(query)
                    detected_lang = detection.get('language_code', 'en')
                    
                    # For non-English medical queries, add English medical terms
                    if detected_lang != 'en':
                        english_medical_query = self._translate_to_medical_english(query)
                        if english_medical_query != query:
                            queries.append(english_medical_query)
            
            # MeSH term enhancement
            mesh_query = self._enhance_with_mesh_terms(query)
            if mesh_query != query:
                queries.append(mesh_query)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in queries:
                if q not in seen:
                    seen.add(q)
                    unique_queries.append(q)
            
            return unique_queries[:3]  # Limit to 3 variants
            
        except Exception as e:
            logger.warning(f"⚠️ Medical multilingual query processing failed: {e}")
            return [query]
    
    def _enhance_medical_query(self, query: str) -> str:
        """Enhance query with medical terminology"""
        try:
            query_lower = query.lower()
            enhanced_terms = []
            
            # Add medical synonyms and related terms
            for category, terms in self.medical_terminologies.items():
                for term in terms:
                    if term in query_lower:
                        # Add related medical terms
                        if category == "disease":
                            enhanced_terms.extend(["pathophysiology", "etiology"])
                        elif category == "treatment":
                            enhanced_terms.extend(["efficacy", "safety", "adverse effects"])
                        elif category == "diagnosis":
                            enhanced_terms.extend(["sensitivity", "specificity"])
                        break
            
            # Add evidence-based medicine terms for quality
            evidence_terms = ["evidence", "clinical", "randomized", "controlled"]
            for term in evidence_terms:
                if term not in query_lower:
                    enhanced_terms.append(term)
                    break  # Add only one evidence term
            
            if enhanced_terms:
                return f"{query} {' '.join(enhanced_terms[:2])}"  # Limit additions
            
            return query
            
        except Exception as e:
            logger.warning(f"Medical query enhancement failed: {e}")
            return query
    
    def _translate_to_medical_english(self, query: str) -> str:
        """Translate medical query to English medical terminology"""
        # This is a placeholder for medical translation
        # In practice, this would use medical translation services
        
        medical_translations = {
            # Common medical terms in other languages
            "enfermedad": "disease",
            "tratamiento": "treatment", 
            "diagnóstico": "diagnosis",
            "medicamento": "medication",
            "病気": "disease",
            "治療": "treatment",
            "診断": "diagnosis",
            "薬": "medication",
            "maladie": "disease",
            "traitement": "treatment",
            "diagnostic": "diagnosis",
            "médicament": "medication"
        }
        
        translated_query = query
        for foreign_term, english_term in medical_translations.items():
            translated_query = translated_query.replace(foreign_term, english_term)
        
        return translated_query
    
    def _enhance_with_mesh_terms(self, query: str) -> str:
        """Enhance query with potential MeSH (Medical Subject Headings) terms"""
        try:
            # Common MeSH term patterns
            mesh_patterns = {
                r'\b(cancer|tumor|neoplasm)\b': 'Neoplasms[MeSH]',
                r'\b(diabetes|diabetic)\b': 'Diabetes Mellitus[MeSH]',
                r'\b(heart|cardiac|cardiovascular)\b': 'Cardiovascular Diseases[MeSH]',
                r'\b(brain|neurological|neural)\b': 'Nervous System Diseases[MeSH]',
                r'\b(infection|infectious|bacterial|viral)\b': 'Infection[MeSH]',
                r'\b(inflammation|inflammatory)\b': 'Inflammation[MeSH]',
                r'\b(drug|medication|pharmaceutical)\b': 'Pharmaceutical Preparations[MeSH]'
            }
            
            query_lower = query.lower()
            mesh_terms = []
            
            for pattern, mesh_term in mesh_patterns.items():
                if re.search(pattern, query_lower):
                    mesh_terms.append(mesh_term)
                    break  # Add only one MeSH term to avoid over-specification
            
            if mesh_terms:
                return f"{query} AND {mesh_terms[0]}"
            
            return query
            
        except Exception as e:
            logger.warning(f"MeSH term enhancement failed: {e}")
            return query
    
    async def _search_paper_ids_async(self, query: str, max_results: int,
                                    start_date: str = None, end_date: str = None,
                                    evidence_level: str = "all") -> List[str]:
        """Enhanced async search for paper IDs with evidence filtering"""
        
        # Check cache first
        cache_key = self._get_cache_key(query, start_date, end_date, max_results, evidence_level)
        if cache_key in self.query_cache:
            cached_result, timestamp = self.query_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.performance_stats['cache_hits'] += 1
                return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        try:
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'xml',
                'sort': 'pub+date',
                'tool': 'polyresearch_medical',
                'email': 'medical.research@example.com'  # Replace with actual email
            }
            
            # Enhanced date range filtering
            if start_date and end_date:
                date_range = f"({start_date}[PDAT] : {end_date}[PDAT])"
                params['term'] = f"{query} AND {date_range}"
            elif start_date:
                params['term'] = f"{query} AND {start_date}[PDAT] : 3000[PDAT]"
            
            # Evidence level filtering
            if evidence_level == "high":
                evidence_filter = " AND (systematic review[pt] OR meta analysis[pt] OR randomized controlled trial[pt])"
                params['term'] += evidence_filter
            elif evidence_level == "medium":
                evidence_filter = " AND (clinical trial[pt] OR controlled clinical trial[pt] OR comparative study[pt])"
                params['term'] += evidence_filter
            
            # Execute request in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.thread_pool,
                lambda: requests.get(self.search_url, params=params, timeout=30)
            )
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Extract paper IDs
            id_list = root.find('IdList')
            paper_ids = []
            if id_list is not None:
                paper_ids = [id_elem.text for id_elem in id_list.findall('Id')]
            
            # Cache result
            self._cache_result(cache_key, paper_ids)
            
            logger.info(f"🔍 Found {len(paper_ids)} PubMed paper IDs for: {query[:50]}")
            return paper_ids
            
        except Exception as e:
            logger.error(f"PubMed ID search failed: {e}")
            return []
    
    async def _fetch_enhanced_paper_details(self, paper_ids: List[str]) -> List[Dict]:
        """Enhanced async fetch of paper details with medical metadata"""
        try:
            if not paper_ids:
                return []
            
            # Process in smaller batches for medical precision
            batch_size = 50  # Smaller batches for better quality
            all_papers = []
            
            for i in range(0, len(paper_ids), batch_size):
                batch_ids = paper_ids[i:i + batch_size]
                
                params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_ids),
                    'retmode': 'xml',
                    'rettype': 'abstract',
                    'tool': 'polyresearch_medical',
                    'email': 'medical.research@example.com'
                }
                
                # Rate limiting for PubMed
                await asyncio.sleep(self.rate_limit_delay)
                
                # Execute request
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: requests.get(self.fetch_url, params=params, timeout=30)
                )
                response.raise_for_status()
                
                # Parse enhanced response
                papers = await loop.run_in_executor(
                    self.thread_pool,
                    self._parse_enhanced_pubmed_response,
                    response.content
                )
                all_papers.extend(papers)
            
            return all_papers
            
        except Exception as e:
            logger.error(f"Enhanced PubMed detail fetch failed: {e}")
            return []
    
    def _parse_enhanced_pubmed_response(self, xml_content: bytes) -> List[Dict]:
        """Enhanced parsing of PubMed XML with medical intelligence"""
        try:
            root = ET.fromstring(xml_content)
            papers = []
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    paper = self._parse_enhanced_pubmed_article(article)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.error(f"Failed to parse enhanced PubMed article: {e}")
                    continue
            
            return papers
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse enhanced PubMed XML: {e}")
            return []
    
    def _parse_enhanced_pubmed_article(self, article) -> Optional[Dict]:
        """Parse individual PubMed article with medical intelligence"""
        try:
            # Extract PMID
            pmid_elem = article.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Enhanced title extraction with HTML cleanup
            title_elem = article.find('.//ArticleTitle')
            title = ""
            if title_elem is not None:
                title = title_elem.text if title_elem.text else ""
                # Handle nested elements
                for child in title_elem:
                    if child.text:
                        title += child.text
                    if child.tail:
                        title += child.tail
                title = re.sub(r'<[^>]+>', '', title)  # Remove HTML tags
                title = re.sub(r'\s+', ' ', title).strip()
            
            # Enhanced abstract extraction
            abstract_parts = []
            abstract_section = article.find('.//Abstract')
            
            if abstract_section is not None:
                # Handle structured abstracts
                for abstract_text in abstract_section.findall('.//AbstractText'):
                    if abstract_text.text:
                        label = abstract_text.get('Label', '')
                        text = abstract_text.text
                        
                        # Handle nested elements
                        for child in abstract_text:
                            if child.text:
                                text += child.text
                            if child.tail:
                                text += child.tail
                        
                        text = re.sub(r'<[^>]+>', '', text)
                        
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
            
            abstract = " ".join(abstract_parts)
            abstract = re.sub(r'\s+', ' ', abstract).strip()
            
            # Enhanced author extraction
            authors = []
            author_list = article.find('.//AuthorList')
            if author_list is not None:
                for author in author_list.findall('Author'):
                    last_name = author.find('LastName')
                    first_name = author.find('ForeName')
                    initials = author.find('Initials')
                    
                    if last_name is not None:
                        name_parts = [last_name.text]
                        
                        if first_name is not None and first_name.text:
                            name_parts.append(first_name.text)
                        elif initials is not None and initials.text:
                            name_parts.append(initials.text)
                        
                        authors.append(" ".join(name_parts))
            
            authors_text = "; ".join(authors)
            
            # Enhanced journal and publication information
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Enhanced publication date extraction
            pub_date = self._extract_enhanced_publication_date(article)
            
            # Extract publication types for quality assessment
            publication_types = []
            pub_type_list = article.find('.//PublicationTypeList')
            if pub_type_list is not None:
                for pub_type in pub_type_list.findall('PublicationType'):
                    if pub_type.text:
                        publication_types.append(pub_type.text)
            
            # Extract MeSH terms for medical classification
            mesh_terms = []
            mesh_list = article.find('.//MeshHeadingList')
            if mesh_list is not None:
                for mesh_heading in mesh_list.findall('MeshHeading'):
                    descriptor = mesh_heading.find('DescriptorName')
                    if descriptor is not None and descriptor.text:
                        mesh_terms.append(descriptor.text)
            
            # Extract keywords
            keywords = []
            keyword_list = article.find('.//KeywordList')
            if keyword_list is not None:
                for keyword in keyword_list.findall('Keyword'):
                    if keyword.text:
                        keywords.append(keyword.text)
            
            # Build URLs
            paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            
            # Extract DOI and PMC ID
            doi = ""
            pmc_id = ""
            
            for article_id in article.findall('.//ArticleId'):
                id_type = article_id.get('IdType')
                if id_type == 'doi' and article_id.text:
                    doi = article_id.text
                elif id_type == 'pmc' and article_id.text:
                    pmc_id = article_id.text
            
            # Enhanced PDF URL construction
            pdf_url = ""
            if pmc_id:
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
            elif doi:
                pdf_url = f"https://doi.org/{doi}"
            
            # Medical quality assessment
            medical_quality = self._assess_medical_publication_quality(
                publication_types, mesh_terms, journal, abstract
            )
            
            # Determine research domain with medical focus
            research_domain = self._classify_medical_domain(title, abstract, mesh_terms)
            
            # Build enhanced paper dictionary
            paper = {
                'title': title,
                'abstract': abstract,
                'authors': authors_text,
                'language': 'en',  # PubMed papers are primarily in English
                'detected_language': 'en',
                'source': 'pubmed',
                'paper_url': paper_url,
                'pdf_url': pdf_url,
                'published_date': pub_date,
                'research_domain': research_domain,
                
                # Enhanced medical metadata
                'pmid': pmid,
                'doi': doi,
                'pmc_id': pmc_id,
                'journal': journal,
                'publication_types': publication_types,
                'mesh_terms': mesh_terms,
                'keywords': keywords,
                
                # Medical quality indicators
                'evidence_level': medical_quality['evidence_level'],
                'is_clinical': medical_quality['is_clinical'],
                'is_systematic_review': medical_quality['is_systematic_review'],
                'is_rct': medical_quality['is_rct'],
                'medical_quality_score': medical_quality['quality_score'],
                
                # Enhanced metadata
                'metadata': {
                    'pmid': pmid,
                    'doi': doi,
                    'pmc_id': pmc_id,
                    'journal': journal,
                    'publication_types': publication_types,
                    'mesh_terms': mesh_terms,
                    'keywords': keywords,
                    'author_count': len(authors),
                    'medical_quality': medical_quality,
                    'extraction_timestamp': datetime.now().isoformat()
                },
                
                # Quality scoring
                'source_quality_score': medical_quality['quality_score'],
                'title_length': len(title),
                'abstract_length': len(abstract),
                'author_count': len(authors)
            }
            
            return paper
            
        except Exception as e:
            logger.error(f"Error parsing enhanced PubMed article: {e}")
            return None
    
    def _extract_enhanced_publication_date(self, article) -> Optional[date]:
        """Enhanced publication date extraction with multiple fallbacks"""
        try:
            # Try electronic publication date first
            epub_date = article.find('.//ArticleDate[@DateType="Electronic"]')
            if epub_date is not None:
                year = epub_date.find('Year')
                month = epub_date.find('Month')
                day = epub_date.find('Day')
                
                if year is not None:
                    return self._construct_date(year.text, 
                                             month.text if month is not None else "1",
                                             day.text if day is not None else "1")
            
            # Try journal publication date
            pub_date_elem = article.find('.//Journal/JournalIssue/PubDate')
            if pub_date_elem is not None:
                year = pub_date_elem.find('Year')
                month = pub_date_elem.find('Month')
                day = pub_date_elem.find('Day')
                
                if year is not None:
                    return self._construct_date(year.text,
                                             month.text if month is not None else "1",
                                             day.text if day is not None else "1")
            
            # Try article publication date
            article_date = article.find('.//PubDate')
            if article_date is not None:
                year = article_date.find('Year')
                month = article_date.find('Month')
                day = article_date.find('Day')
                
                if year is not None:
                    return self._construct_date(year.text,
                                             month.text if month is not None else "1",
                                             day.text if day is not None else "1")
            
            return None
            
        except Exception as e:
            logger.warning(f"Enhanced date extraction failed: {e}")
            return None
    
    def _construct_date(self, year_str: str, month_str: str, day_str: str) -> Optional[date]:
        """Construct date from string components"""
        try:
            year = int(year_str)
            
            # Handle month names
            if month_str.isdigit():
                month = int(month_str)
            else:
                month_names = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
                    'January': 1, 'February': 2, 'March': 3, 'April': 4,
                    'June': 6, 'July': 7, 'August': 8, 'September': 9,
                    'October': 10, 'November': 11, 'December': 12
                }
                month = month_names.get(month_str, 1)
            
            day = int(day_str) if day_str.isdigit() else 1
            
            # Validate ranges
            month = max(1, min(12, month))
            day = max(1, min(31, day))
            
            return date(year, month, day)
            
        except (ValueError, TypeError):
            return None
    
    def _assess_medical_publication_quality(self, publication_types: List[str],
                                          mesh_terms: List[str], journal: str,
                                          abstract: str) -> Dict[str, Any]:
        """Assess medical publication quality with evidence-based criteria"""
        quality_assessment = {
            'evidence_level': 'low',
            'is_clinical': False,
            'is_systematic_review': False,
            'is_rct': False,
            'quality_score': 0.3
        }
        
        try:
            # Check publication types for evidence level
            high_evidence_types = [
                'Meta-Analysis', 'Systematic Review', 'Practice Guideline',
                'Randomized Controlled Trial', 'Clinical Trial, Phase III'
            ]
            
            medium_evidence_types = [
                'Clinical Trial', 'Controlled Clinical Trial', 'Comparative Study',
                'Multicenter Study', 'Clinical Trial, Phase II'
            ]
            
            clinical_types = [
                'Clinical Trial', 'Controlled Clinical Trial', 'Randomized Controlled Trial',
                'Clinical Trial, Phase I', 'Clinical Trial, Phase II', 'Clinical Trial, Phase III',
                'Clinical Trial, Phase IV'
            ]
            
            # Assess evidence level
            if any(pt in high_evidence_types for pt in publication_types):
                quality_assessment['evidence_level'] = 'high'
                quality_assessment['quality_score'] = 0.9
            elif any(pt in medium_evidence_types for pt in publication_types):
                quality_assessment['evidence_level'] = 'medium'
                quality_assessment['quality_score'] = 0.7
            else:
                quality_assessment['quality_score'] = 0.5
            
            # Check for clinical studies
            quality_assessment['is_clinical'] = any(pt in clinical_types for pt in publication_types)
            
            # Check for systematic reviews and meta-analyses
            quality_assessment['is_systematic_review'] = any(
                'systematic review' in pt.lower() or 'meta-analysis' in pt.lower()
                for pt in publication_types
            )
            
            # Check for RCTs
            quality_assessment['is_rct'] = any(
                'randomized controlled trial' in pt.lower()
                for pt in publication_types
            )
            
            # Bonus for high-impact journals (simplified check)
            high_impact_indicators = [
                'new england journal', 'lancet', 'jama', 'nature', 'science',
                'cell', 'nejm', 'bmj', 'annals of internal medicine'
            ]
            
            if any(indicator in journal.lower() for indicator in high_impact_indicators):
                quality_assessment['quality_score'] += 0.1
            
            # Bonus for structured abstract
            if abstract and any(indicator in abstract.lower() for indicator in 
                               ['objective:', 'methods:', 'results:', 'conclusions:']):
                quality_assessment['quality_score'] += 0.05
            
            # Cap quality score
            quality_assessment['quality_score'] = min(1.0, quality_assessment['quality_score'])
            
        except Exception as e:
            logger.warning(f"Medical quality assessment failed: {e}")
        
        return quality_assessment
    
    def _classify_medical_domain(self, title: str, abstract: str, mesh_terms: List[str]) -> str:
        """Classify medical research domain"""
        try:
            text = f"{title} {abstract}".lower()
            
            # Medical domain classification
            domain_keywords = {
                'Cardiovascular Medicine': ['heart', 'cardiac', 'cardiovascular', 'coronary', 'hypertension'],
                'Oncology': ['cancer', 'tumor', 'neoplasm', 'oncology', 'carcinoma', 'malignancy'],
                'Neurology': ['brain', 'neural', 'neurological', 'alzheimer', 'parkinson', 'stroke'],
                'Infectious Diseases': ['infection', 'bacterial', 'viral', 'antimicrobial', 'antibiotic'],
                'Endocrinology': ['diabetes', 'hormone', 'endocrine', 'insulin', 'thyroid'],
                'Psychiatry': ['mental health', 'depression', 'anxiety', 'psychiatric', 'psychology'],
                'Pediatrics': ['pediatric', 'children', 'infant', 'adolescent', 'child'],
                'Surgery': ['surgical', 'surgery', 'operative', 'laparoscopic', 'minimally invasive'],
                'Emergency Medicine': ['emergency', 'trauma', 'critical care', 'intensive care'],
                'Public Health': ['public health', 'epidemiology', 'population', 'preventive']
            }
            
            # Check MeSH terms first (more reliable)
            for domain, keywords in domain_keywords.items():
                if any(keyword in ' '.join(mesh_terms).lower() for keyword in keywords):
                    return domain
            
            # Check text content
            domain_scores = {}
            for domain, keywords in domain_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text)
                if score > 0:
                    domain_scores[domain] = score
            
            if domain_scores:
                return max(domain_scores.keys(), key=domain_scores.get)
            
            return 'General Medicine'
            
        except Exception as e:
            logger.warning(f"Medical domain classification failed: {e}")
            return 'General Medicine'
    
    async def _assess_medical_quality(self, papers: List[Dict]) -> List[Dict]:
        """Assess and enhance medical quality of papers"""
        if not self.text_processor:
            return papers
        
        try:
            enhanced_papers = []
            
            for paper in papers:
                # Use existing medical quality score
                quality_score = paper.get('medical_quality_score', 0.5)
                
                # Additional academic validation
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                if title and abstract:
                    validation = self.text_processor.validate_academic_text(
                        f"{title} {abstract}", language='en'
                    )
                    
                    if validation.get('is_academic', False):
                        # Boost score for academic validation
                        quality_score += 0.1 * validation.get('confidence', 0)
                
                # Medical-specific quality boosts
                if paper.get('is_rct', False):
                    quality_score += 0.1
                if paper.get('is_systematic_review', False):
                    quality_score += 0.15
                if paper.get('evidence_level') == 'high':
                    quality_score += 0.1
                
                paper['final_medical_quality'] = min(1.0, quality_score)
                enhanced_papers.append(paper)
            
            return enhanced_papers
            
        except Exception as e:
            logger.warning(f"Medical quality assessment failed: {e}")
            return papers
    
    def _rank_papers_by_medical_relevance(self, papers: List[Dict], original_query: str) -> List[Dict]:
        """Rank papers by medical relevance and evidence quality"""
        try:
            scored_papers = []
            
            for paper in papers:
                relevance_score = self._calculate_medical_relevance_score(paper, original_query)
                paper['medical_relevance_score'] = relevance_score
                scored_papers.append((paper, relevance_score))
            
            # Sort by medical relevance score (descending)
            scored_papers.sort(key=lambda x: x[1], reverse=True)
            
            return [paper for paper, score in scored_papers]
            
        except Exception as e:
            logger.warning(f"Medical ranking failed: {e}")
            return papers
    
    def _calculate_medical_relevance_score(self, paper: Dict, query: str) -> float:
        """Calculate medical relevance score"""
        score = 0.0
        
        try:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            mesh_terms = [term.lower() for term in paper.get('mesh_terms', [])]
            query_lower = query.lower()
            
            # Query term matching (weighted by section)
            query_words = set(query_lower.split())
            
            # Title matching (highest weight)
            title_words = set(title.split())
            if query_words and title_words:
                title_overlap = len(query_words.intersection(title_words)) / len(query_words)
                score += title_overlap * 0.3
            
            # Abstract matching
            abstract_words = set(abstract.split())
            if query_words and abstract_words:
                abstract_overlap = len(query_words.intersection(abstract_words)) / len(query_words)
                score += abstract_overlap * 0.2
            
            # MeSH terms matching (medical specificity)
            if mesh_terms and query_words:
                mesh_text = ' '.join(mesh_terms)
                mesh_overlap = len([word for word in query_words if word in mesh_text]) / len(query_words)
                score += mesh_overlap * 0.2
            
            # Evidence level bonus
            evidence_level = paper.get('evidence_level', 'low')
            if evidence_level == 'high':
                score += 0.15
            elif evidence_level == 'medium':
                score += 0.1
            
            # Clinical trial bonus
            if paper.get('is_clinical', False):
                score += 0.05
            
            # Systematic review bonus
            if paper.get('is_systematic_review', False):
                score += 0.1
            
            # Quality score component
            quality_score = paper.get('final_medical_quality', paper.get('medical_quality_score', 0.5))
            score += quality_score * 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Medical relevance calculation error: {e}")
            return 0.5
    
    # Helper methods for caching and performance
    def _get_cache_key(self, query: str, start_date: str, end_date: str, 
                      max_results: int, evidence_level: str) -> str:
        """Generate cache key for medical query parameters"""
        key_data = f"{query}_{start_date}_{end_date}_{max_results}_{evidence_level}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _cache_result(self, cache_key: str, result: List[str]):
        """Cache search result with timestamp"""
        self.query_cache[cache_key] = (result, time.time())
        
        # Clean cache if needed
        if len(self.query_cache) > self.max_cache_size:
            self._clean_cache()
    
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
    
    # Enhanced medical-specific methods
    async def search_clinical_trials_async(self, condition: str, max_results: int = 20) -> List[Dict]:
        """Search specifically for clinical trials"""
        clinical_query = f"{condition} AND (clinical trial[pt] OR randomized controlled trial[pt])"
        
        return await self.search_papers_async(
            clinical_query, max_results, evidence_level="medium", medical_focus=True
        )
    
    async def search_systematic_reviews_async(self, topic: str, max_results: int = 20) -> List[Dict]:
        """Search specifically for systematic reviews and meta-analyses"""
        review_query = f"{topic} AND (systematic review[pt] OR meta analysis[pt])"
        
        return await self.search_papers_async(
            review_query, max_results, evidence_level="high", medical_focus=True
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics with medical metrics"""
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
            'medical_specialization': True,
            'uptime_hours': (time.time() - self.performance_stats['last_reset']) / 3600
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with medical API validation"""
        try:
            # Test basic connectivity with medical query
            start_time = time.time()
            
            test_params = {
                'db': 'pubmed',
                'term': 'medical research',
                'retmax': 1,
                'retmode': 'xml'
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: requests.get(self.search_url, params=test_params, timeout=10)
            )
            
            response_time = time.time() - start_time
            
            health_status = {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "response_time": response_time,
                "api_available": response.status_code == 200,
                "multilingual_support": self.multilingual_support,
                "medical_specialization": True,
                "cache_size": len(self.query_cache),
                "performance_stats": self.get_performance_stats(),
                "medical_features": {
                    "mesh_term_support": True,
                    "evidence_level_filtering": True,
                    "clinical_trial_search": True,
                    "systematic_review_search": True,
                    "quality_assessment": True
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "multilingual_support": self.multilingual_support,
                "medical_specialization": True,
                "timestamp": datetime.now().isoformat()
            }
    
    def close(self):
        """Clean up resources"""
        try:
            self.thread_pool.shutdown(wait=True)
            self.query_cache.clear()
            logger.info("🔒 Enhanced PubMed client resources cleaned up")
        except Exception as e:
            logger.error(f"PubMed client cleanup error: {e}")

# Backward compatibility
PubMedClient = EnhancedPubMedClient
