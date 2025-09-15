import asyncio
import httpx
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)

@dataclass
class KimiAnalysisResult:
    """Kimi analysis result structure"""
    analysis_type: str
    content: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_used: str
    token_count: Optional[int] = None

class KimiClient:
    """
    Kimi AI client for detailed paper analysis and context understanding
    """
    
    def __init__(self):
        # Kimi API configuration (you'll need to add actual API key)
        self.api_key = "your_kimi_api_key_here"  # Replace with actual API key
        self.base_url = "https://api.moonshot.cn/v1"
        self.model = "moonshot-v1-8k"
        
        # Client configuration
        self.timeout = 30
        self.max_retries = 3
        self.rate_limit_delay = 1.0
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'average_response_time': 0.0
        }
        
        # HTTP client
        self.client = None
        
        logger.info("🌙 Kimi AI client initialized")
    
    async def initialize(self):
        """Initialize HTTP client and test connection"""
        try:
            self.client = httpx.AsyncClient(
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                timeout=self.timeout
            )
            
            # Test connection (if API key is valid)
            if self.api_key != "your_kimi_api_key_here":
                await self._test_connection()
            
            logger.info("✅ Kimi client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Kimi client initialization failed: {e}")
            # Create a mock client for testing
            self.client = None
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            if not self.client:
                return False
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                }
            )
            
            if response.status_code == 200:
                logger.info("✅ Kimi API connection successful")
                return True
            else:
                logger.warning(f"⚠️ Kimi API test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Kimi API test failed: {e}")
            return False
    
    async def _make_request(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> Optional[str]:
        """Make request to Kimi API with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Mock response if no real API key (for development)
                if not self.client or self.api_key == "your_kimi_api_key_here":
                    await asyncio.sleep(0.5)  # Simulate API delay
                    return self._generate_mock_response(messages)
                
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.1,
                        "stream": False
                    }
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    
                    # Update stats
                    self.stats['total_requests'] += 1
                    self.stats['successful_requests'] += 1
                    self.stats['total_tokens'] += data.get('usage', {}).get('total_tokens', 0)
                    
                    avg_time = self.stats['average_response_time']
                    total_requests = self.stats['successful_requests']
                    self.stats['average_response_time'] = (
                        (avg_time * (total_requests - 1) + processing_time) / total_requests
                    )
                    
                    return content
                    
                else:
                    logger.warning(f"⚠️ Kimi API request failed: {response.status_code}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.rate_limit_delay * (2 ** attempt))
                        continue
                    
            except Exception as e:
                logger.error(f"❌ Kimi API request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.rate_limit_delay * (2 ** attempt))
                    continue
        
        # All attempts failed
        self.stats['failed_requests'] += 1
        return None
    
    def _generate_mock_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate mock response for development/testing"""
        user_message = messages[-1]['content'] if messages else ""
        
        if "analyze paper" in user_message.lower():
            return json.dumps({
                "detailed_analysis": {
                    "research_significance": "High - addresses important gap in current literature",
                    "methodological_rigor": "Strong experimental design with appropriate controls",
                    "novelty_assessment": "Introduces novel approach with significant improvements",
                    "technical_depth": "Comprehensive technical implementation with detailed analysis",
                    "impact_potential": "High potential for advancing the field",
                    "limitations_analysis": "Some limitations in sample size and generalizability",
                    "future_directions": "Multiple promising directions for extension",
                    "related_work_context": "Well-positioned within existing literature"
                },
                "confidence_score": 0.85,
                "analysis_model": "kimi_detailed_mock"
            })
        
        elif "relationship" in user_message.lower():
            return json.dumps({
                "relationship_analysis": {
                    "semantic_similarity": 0.75,
                    "methodological_overlap": "Moderate - both use similar computational approaches",
                    "thematic_connection": "Strong connection in underlying research themes",
                    "complementarity": "Papers complement each other in scope and methodology",
                    "temporal_relationship": "Second paper builds upon findings of the first",
                    "citation_potential": "High likelihood of mutual citation"
                },
                "confidence_score": 0.78
            })
        
        else:
            return json.dumps({
                "analysis": "Detailed analysis completed using Kimi AI capabilities",
                "confidence_score": 0.80
            })
    
    async def analyze_paper_detailed(self, title: str, abstract: str, 
                                   content: str = "", analysis_depth: str = "deep") -> KimiAnalysisResult:
        """
        Perform detailed paper analysis using Kimi AI
        
        Args:
            title: Paper title
            abstract: Paper abstract  
            content: Full paper content (optional)
            analysis_depth: 'deep' or 'standard'
            
        Returns:
            KimiAnalysisResult with detailed analysis
        """
        start_time = time.time()
        
        try:
            # Prepare comprehensive analysis prompt
            paper_text = f"Title: {title}\n\nAbstract: {abstract}"
            if content:
                paper_text += f"\n\nContent: {content[:2000]}"  # Limit content size
            
            if analysis_depth == "deep":
                prompt = f"""Perform a comprehensive deep analysis of this research paper:

{paper_text}

Provide detailed analysis in JSON format with the following aspects:

{{
    "detailed_analysis": {{
        "research_significance": "Assessment of the paper's significance to the field",
        "methodological_rigor": "Evaluation of research methodology and experimental design", 
        "novelty_assessment": "Analysis of novel contributions and innovations",
        "technical_depth": "Assessment of technical complexity and implementation",
        "impact_potential": "Potential impact on the field and future research",
        "limitations_analysis": "Identified limitations and potential weaknesses",
        "future_directions": "Suggested future research directions",
        "related_work_context": "How this work relates to existing literature"
    }},
    "confidence_score": "Float between 0.0 and 1.0",
    "analysis_model": "kimi_detailed"
}}

Provide thorough, insightful analysis that goes beyond surface-level observations."""

            else:
                prompt = f"""Analyze this research paper and provide structured insights:

{paper_text}

Provide analysis in JSON format focusing on key aspects of the research."""
            
            messages = [
                {"role": "system", "content": "You are an expert research analyst. Provide detailed, insightful analysis of academic papers."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._make_request(messages, max_tokens=1500)
            processing_time = time.time() - start_time
            
            if response:
                try:
                    # Clean and parse JSON response
                    clean_response = self._clean_json_response(response)
                    analysis_data = json.loads(clean_response)
                    
                    return KimiAnalysisResult(
                        analysis_type="detailed_paper_analysis",
                        content=analysis_data,
                        confidence_score=float(analysis_data.get('confidence_score', 0.8)),
                        processing_time=processing_time,
                        model_used=self.model,
                        token_count=len(response.split())
                    )
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse Kimi response: {e}")
                    return self._create_fallback_analysis(title, abstract, processing_time)
            
            else:
                return self._create_fallback_analysis(title, abstract, processing_time)
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Kimi paper analysis failed: {e}")
            return self._create_fallback_analysis(title, abstract, processing_time)
    
    async def analyze_relationship_detailed(self, paper1: Dict[str, Any], 
                                          paper2: Dict[str, Any]) -> KimiAnalysisResult:
        """
        Analyze detailed relationship between two papers using Kimi AI
        
        Args:
            paper1: First paper data
            paper2: Second paper data
            
        Returns:
            KimiAnalysisResult with relationship analysis
        """
        start_time = time.time()
        
        try:
            # Prepare relationship analysis prompt
            prompt = f"""Analyze the detailed relationship between these two research papers:

Paper 1:
Title: {paper1.get('title', '')}
Abstract: {paper1.get('abstract', '')[:500]}
Domain: {paper1.get('research_domain', 'Unknown')}

Paper 2:
Title: {paper2.get('title', '')}
Abstract: {paper2.get('abstract', '')[:500]}
Domain: {paper2.get('research_domain', 'Unknown')}

Provide comprehensive relationship analysis in JSON format:

{{
    "relationship_analysis": {{
        "semantic_similarity": "Float score 0.0-1.0",
        "methodological_overlap": "Description of methodological similarities/differences",
        "thematic_connection": "Analysis of thematic relationships",
        "complementarity": "How the papers complement each other",
        "temporal_relationship": "Analysis of temporal/evolutionary relationship",
        "citation_potential": "Likelihood and reasoning for mutual citation"
    }},
    "confidence_score": "Float between 0.0 and 1.0"
}}

Focus on deep semantic understanding and research context."""

            messages = [
                {"role": "system", "content": "You are an expert research analyst specializing in identifying relationships between academic papers."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._make_request(messages, max_tokens=800)
            processing_time = time.time() - start_time
            
            if response:
                try:
                    clean_response = self._clean_json_response(response)
                    relationship_data = json.loads(clean_response)
                    
                    return KimiAnalysisResult(
                        analysis_type="detailed_relationship_analysis",
                        content=relationship_data,
                        confidence_score=float(relationship_data.get('confidence_score', 0.7)),
                        processing_time=processing_time,
                        model_used=self.model
                    )
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse Kimi relationship response: {e}")
                    return self._create_fallback_relationship(processing_time)
            
            else:
                return self._create_fallback_relationship(processing_time)
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Kimi relationship analysis failed: {e}")
            return self._create_fallback_relationship(processing_time)
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from Kimi API"""
        try:
            # Remove markdown code blocks
            response = response.replace('``````', '')
            
            # Find JSON object
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                return response[start_idx:end_idx]
            
            return response
            
        except Exception:
            return response
    
    def _create_fallback_analysis(self, title: str, abstract: str, processing_time: float) -> KimiAnalysisResult:
        """Create fallback analysis when Kimi fails"""
        return KimiAnalysisResult(
            analysis_type="fallback_analysis",
            content={
                "detailed_analysis": {
                    "research_significance": "Analysis unavailable - using fallback",
                    "methodological_rigor": "Unable to assess methodology",
                    "novelty_assessment": "Novelty assessment not available",
                    "technical_depth": "Technical analysis not completed",
                    "impact_potential": "Impact assessment not available",
                    "limitations_analysis": "Limitations not analyzed",
                    "future_directions": "Future directions not identified",
                    "related_work_context": "Context analysis not available"
                },
                "confidence_score": 0.3
            },
            confidence_score=0.3,
            processing_time=processing_time,
            model_used="fallback"
        )
    
    def _create_fallback_relationship(self, processing_time: float) -> KimiAnalysisResult:
        """Create fallback relationship analysis"""
        return KimiAnalysisResult(
            analysis_type="fallback_relationship",
            content={
                "relationship_analysis": {
                    "semantic_similarity": 0.5,
                    "methodological_overlap": "Analysis not available",
                    "thematic_connection": "Connection assessment not completed",
                    "complementarity": "Complementarity not analyzed",
                    "temporal_relationship": "Temporal analysis not available",
                    "citation_potential": "Citation potential not assessed"
                },
                "confidence_score": 0.3
            },
            confidence_score=0.3,
            processing_time=processing_time,
            model_used="fallback"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Kimi client statistics"""
        return {
            **self.stats,
            'model': self.model,
            'base_url': self.base_url,
            'success_rate': (
                self.stats['successful_requests'] / max(self.stats['total_requests'], 1) * 100
            ),
            'api_status': 'connected' if self.client else 'mock_mode'
        }
    
    async def close(self):
        """Close HTTP client"""
        try:
            if self.client:
                await self.client.aclose()
            logger.info("🔒 Kimi client closed")
        except Exception as e:
            logger.error(f"❌ Error closing Kimi client: {e}")
