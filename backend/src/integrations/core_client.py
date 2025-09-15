import requests
from datetime import datetime, date
import logging
from typing import List, Dict, Optional
import time
import re

logger = logging.getLogger(__name__)

class COREClient:
    """Client for CORE.ac.uk API to fetch open access research papers"""
    
    def __init__(self, api_key: str = "3wyaMDkxpeIzV7H4YhvGtg9mJnTBXsod"):
        self.api_key = api_key
        self.base_url = "https://api.core.ac.uk/v3"
        self.rate_limit_delay = 2.0  # CORE allows 5 requests per 10 seconds for single methods
        
    def search_by_subject(self, subject: str, max_results: int = 20) -> List[Dict]:
        """
        Search papers by subject/field
        
        Args:
            subject: Subject or field name
            max_results: Maximum number of results
            
        Returns:
            List of papers in the subject area
        """
        query = f"subjects:\"{subject}\" OR title:\"{subject}\" OR description:\"{subject}\""
        
        url = f"{self.base_url}/search/articles"
        params = {
            'q': query,
            'limit': min(max_results, 100),
            'apiKey': self.api_key,
            'scroll': 'false'
        }
        
        time.sleep(self.rate_limit_delay)
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'OK':
                return self._parse_core_response(data.get('data', []))
            
        except Exception as e:
            logger.error(f"CORE subject search failed: {e}")
        
        return []

    def get_trending_papers(self, max_results: int = 20) -> List[Dict]:
        """Get trending/popular papers"""
        # Search for recent papers with high citation potential
        current_year = datetime.now().year
        
        query = f"year:>={current_year-1}"  # Last 2 years
        
        url = f"{self.base_url}/search/articles"
        params = {
            'q': query,
            'limit': min(max_results, 100),
            'apiKey': self.api_key,
            'scroll': 'false',
            'sort': 'year:desc'
        }
        
        time.sleep(self.rate_limit_delay)
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'OK':
                return self._parse_core_response(data.get('data', []))
            
        except Exception as e:
            logger.error(f"CORE trending papers failed: {e}")
        
        return []

    
    def search_papers(self, query: str, max_results: int = 20, 
                     language: str = None, year_from: int = None) -> List[Dict]:
        """
        Search for papers on CORE
        
        Args:
            query: Search query
            max_results: Maximum number of results (max 100 per request)
            language: Language filter (e.g., 'English')
            year_from: Minimum publication year
            
        Returns:
            List of paper dictionaries
        """
        try:
            # Build search query with filters
            search_query = self._build_search_query(query, language, year_from)
            
            # CORE API endpoint for searching articles
            url = f"{self.base_url}/search/articles"
            
            params = {
                'q': search_query,
                'limit': min(max_results, 100),  # CORE max limit is 100
                'apiKey': self.api_key,
                'scroll': 'false'  # Don't use scrolling for simple searches
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK':
                papers = self._parse_core_response(data.get('data', []))
                logger.info(f"Found {len(papers)} papers from CORE for query: {query}")
                return papers
            else:
                logger.error(f"CORE API error: {data}")
                return []
            
        except requests.RequestException as e:
            logger.error(f"CORE API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"CORE search failed: {e}")
            return []
    
    def get_recent_papers(self, days: int = 7, max_results: int = 50, 
                         subject: str = None) -> List[Dict]:
        """
        Get recent papers from CORE
        
        Args:
            days: Number of days back to search
            max_results: Maximum number of results
            subject: Subject filter (optional)
            
        Returns:
            List of recent papers
        """
        try:
            from datetime import datetime, timedelta
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Build query for recent papers
            query_parts = []
            
            if subject:
                query_parts.append(f"title:{subject} OR description:{subject}")
            else:
                query_parts.append("*")  # All papers
            
            # Add date range filter
            start_year = start_date.year
            query_parts.append(f"year:>={start_year}")
            
            search_query = " AND ".join(query_parts)
            
            url = f"{self.base_url}/search/articles"
            
            params = {
                'q': search_query,
                'limit': min(max_results, 100),
                'apiKey': self.api_key,
                'scroll': 'false',
                'sort': 'year:desc'  # Sort by year descending
            }
            
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK':
                papers = self._parse_core_response(data.get('data', []))
                logger.info(f"Found {len(papers)} recent papers from CORE")
                return papers
            else:
                logger.error(f"CORE API error: {data}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get recent CORE papers: {e}")
            return []
    
    def get_paper_by_id(self, core_id: str) -> Optional[Dict]:
        """
        Get specific paper by CORE ID
        
        Args:
            core_id: CORE paper ID
            
        Returns:
            Paper dictionary or None
        """
        try:
            url = f"{self.base_url}/articles/get/{core_id}"
            
            params = {
                'apiKey': self.api_key
            }
            
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('data'):
                paper = self._parse_core_article(data['data'])
                logger.info(f"Found paper CORE ID: {core_id}")
                return paper
            else:
                logger.warning(f"Paper CORE ID {core_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get paper CORE ID {core_id}: {e}")
            return None
    
    def search_by_fulltext(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Search papers by full text content
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of papers
        """
        try:
            url = f"{self.base_url}/search/articles"
            
            # Search in full text
            search_query = f"fullText:\"{query}\""
            
            params = {
                'q': search_query,
                'limit': min(max_results, 100),
                'apiKey': self.api_key,
                'scroll': 'false'
            }
            
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK':
                papers = self._parse_core_response(data.get('data', []))
                logger.info(f"Found {len(papers)} papers with full-text search")
                return papers
            else:
                logger.error(f"CORE full-text search error: {data}")
                return []
                
        except Exception as e:
            logger.error(f"CORE full-text search failed: {e}")
            return []
    
    def get_similar_papers(self, core_id: str, max_results: int = 10) -> List[Dict]:
        """
        Get papers similar to a given CORE ID
        
        Args:
            core_id: CORE paper ID
            max_results: Maximum number of results
            
        Returns:
            List of similar papers
        """
        try:
            url = f"{self.base_url}/articles/similar/{core_id}"
            
            params = {
                'apiKey': self.api_key,
                'limit': min(max_results, 100)
            }
            
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK':
                papers = self._parse_core_response(data.get('data', []))
                logger.info(f"Found {len(papers)} similar papers")
                return papers
            else:
                logger.error(f"CORE similar papers error: {data}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get similar papers: {e}")
            return []
    
    def _build_search_query(self, query: str, language: str = None, 
                      year_from: int = None) -> str:
        """Build CORE search query string"""
        
        # Simplified query format for CORE API
        clean_query = re.sub(r'[^\w\s\-]', '', query)
        query_parts = []
        
        # Main search - simpler format
        if clean_query.strip():
            query_parts.append(clean_query.strip())
        
        # Add language filter differently
        if language and language != "Unknown" and language.lower() != "english":
            query_parts.append(f"language:{language}")
        
        # Add year filter
        if year_from:
            query_parts.append(f"year:{year_from}")
        
        return " ".join(query_parts) if query_parts else query.strip()

    
    def _parse_core_response(self, articles: List[Dict]) -> List[Dict]:
        """Parse CORE API response"""
        papers = []
        
        for article in articles:
            try:
                paper = self._parse_core_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.error(f"Failed to parse CORE article: {e}")
                continue
        
        return papers
    
    
    def _parse_core_article(self, article: Dict) -> Optional[Dict]:
        """Parse individual CORE article"""
        try:
            # Extract basic information
            title = article.get('title', '').strip()
            description = article.get('description', '').strip()
            
            # CORE uses 'description' field for abstract
            abstract = description if description else article.get('abstract', '')
            
            # Extract authors
            authors_list = article.get('authors', [])
            authors = []
            
            if isinstance(authors_list, list):
                for author in authors_list:
                    if isinstance(author, dict):
                        name = author.get('name', '').strip()
                        if name:
                            authors.append(name)
                    elif isinstance(author, str):
                        authors.append(author.strip())
            
            authors_text = "; ".join(authors) if authors else "Unknown authors"
            
            # Extract URLs and identifiers
            core_id = str(article.get('id', ''))
            
            # Build CORE URL
            paper_url = f"https://core.ac.uk/reader/{core_id}" if core_id else ""
            
            # Extract download URL
            download_url = ""
            download_info = article.get('downloadUrl')
            if download_info:
                download_url = download_info
            elif article.get('fullTextIdentifier'):
                download_url = article['fullTextIdentifier']
            
            # Extract DOI
            doi = article.get('doi', '')
            
            # Extract publication info
            pub_date = None
            year_published = article.get('yearPublished')
            date_published = article.get('datePublished')
            
            if date_published:
                try:
                    pub_date = datetime.strptime(date_published[:10], "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    pass
            
            if not pub_date and year_published:
                try:
                    pub_date = date(int(year_published), 1, 1)
                except (ValueError, TypeError):
                    pass
            
            # Extract language
            language = 'en'  # Default
            lang_info = article.get('language')
            if isinstance(lang_info, dict):
                language = lang_info.get('code', 'en')
            elif isinstance(lang_info, str):
                language = lang_info[:2].lower()
            
            # Extract publisher/journal
            publisher = article.get('publisher', '')
            journal = article.get('journals', [])
            if journal and isinstance(journal, list) and len(journal) > 0:
                publisher = journal[0].get('title', publisher)
            
            # Extract subjects/topics
            subjects = article.get('subjects', [])
            topics = []
            if isinstance(subjects, list):
                for subject in subjects:
                    if isinstance(subject, str):
                        topics.append(subject)
                    elif isinstance(subject, dict):
                        name = subject.get('name', '')
                        if name:
                            topics.append(name)
            
            paper = {
                'title': title,
                'abstract': abstract,
                'authors': authors_text,
                'language': language,
                'source': 'core',
                'paper_url': paper_url,
                'pdf_url': download_url,
                'published_date': pub_date,
                'metadata': {
                    'core_id': core_id,
                    'doi': doi,
                    'publisher': publisher,
                    'subjects': topics,
                    'year_published': year_published,
                    'repositories': article.get('repositories', [])
                }
            }
            
            return paper
            
        except Exception as e:
            logger.error(f"Error parsing CORE article: {e}")
            return None
        
    