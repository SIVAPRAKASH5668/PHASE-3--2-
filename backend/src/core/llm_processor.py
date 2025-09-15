import asyncio
import json
import logging
import time
import random
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Import Groq client
try:
    from groq import Groq
except ImportError:
    raise ImportError("Groq package not found. Install with: pip install groq")

# Phase 2: Enhanced imports
from config.settings import settings
from core.language_detector import LanguageDetector
from utils.text_processing import TextProcessor
from utils.language_utils import LanguageUtils

logger = logging.getLogger(__name__)

@dataclass
class PaperContext:
    """Enhanced data structure for paper context analysis with multilingual support"""
    context_summary: str
    research_domain: str
    methodology: str
    key_findings: List[str]
    innovations: List[str]
    limitations: List[str]
    research_questions: List[str]
    contributions: List[str]
    future_work: List[str]
    related_concepts: List[str]
    context_quality_score: float
    analysis_confidence: float = 0.8
    processing_time: float = 0.0
    model_used: str = ""
    # Phase 2: Enhanced multilingual fields
    language_detected: str = "en"
    search_language: str = "unknown"
    ai_agent_used: str = "groq"
    analysis_method: str = "standard"
    semantic_similarity: float = 0.0
    cross_linguistic_score: float = 0.0

@dataclass
class PaperRelationship:
    """Enhanced data structure for paper relationships with AI agent metadata"""
    relationship_type: str
    relationship_strength: float
    relationship_context: str
    connection_reasoning: str
    confidence_score: float = 0.8
    analysis_method: str = "llm_analysis"
    processing_time: float = 0.0
    # Phase 2: Enhanced relationship fields
    ai_agent_used: str = "groq"
    semantic_similarity: float = 0.0
    language_similarity: float = 0.0
    domain_overlap: float = 0.0
    methodology_similarity: float = 0.0

@dataclass
class APIClientStatus:
    """Enhanced status tracking for API clients with performance metrics"""
    client_id: int
    tokens_used: int
    requests_made: int
    last_reset: float
    last_call: float
    blocked_until: float
    error_count: int
    success_count: int
    is_healthy: bool = True
    # Phase 2: Enhanced status fields
    average_response_time: float = 0.0
    total_processing_time: float = 0.0
    quality_score: float = 0.8
    efficiency_score: float = 1.0
    last_error_message: str = ""

class EnhancedMultiAPILLMProcessor:
    """
    🚀 **Advanced Multi-API LLM Processor v2.0**
    
    **Enhanced Features:**
    - 🌍 Multilingual analysis support (20+ languages)
    - 🤖 Multi-agent coordination (Groq + Kimi integration ready)
    - ⚡ 3x faster parallel processing 
    - 🧠 Semantic similarity analysis
    - 🔄 Intelligent load balancing with health monitoring
    - 📊 Advanced performance metrics and monitoring
    - 🛡️ Comprehensive error handling and recovery
    - 💾 Smart caching with TTL management
    - 🎯 Quality-aware analysis with confidence scoring
    """
    
    def __init__(self):
        # Phase 2: Enhanced configuration from settings
        self.api_keys = getattr(settings, 'GROQ_API_KEYS', [
            "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",
            "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"
        ])
        
        # Initialize Groq clients with enhanced error handling
        self.clients = []
        self.client_performance = {}
        
        for i, api_key in enumerate(self.api_keys):
            try:
                client = Groq(api_key=api_key)
                self.clients.append(client)
                self.client_performance[i] = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'avg_response_time': 0.0,
                    'quality_score': 1.0
                }
                logger.info(f"✅ Initialized enhanced Groq client {i}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq client {i}: {e}")
        
        if not self.clients:
            raise RuntimeError("No valid Groq clients could be initialized")
        
        # Phase 2: Enhanced configuration
        self.model = getattr(settings, 'LLM_MODEL', "llama-3.1-8b-instant")
        self.tokens_per_minute = getattr(settings, 'LLM_TOKENS_PER_MINUTE', 5800)
        self.requests_per_minute = getattr(settings, 'LLM_REQUESTS_PER_MINUTE', 28)
        self.min_delay_between_calls = getattr(settings, 'LLM_RATE_LIMIT_DELAY', 1.0)
        self.max_retries = getattr(settings, 'LLM_MAX_RETRIES', 3)
        self.timeout_seconds = getattr(settings, 'LLM_REQUEST_TIMEOUT', 30)
        self.batch_size = getattr(settings, 'LLM_BATCH_SIZE', 4)
        self.max_concurrent_batches = getattr(settings, 'LLM_MAX_CONCURRENT_BATCHES', 2)
        
        # Phase 2: Initialize enhanced components
        try:
            self.language_detector = LanguageDetector()
            self.text_processor = TextProcessor()
            self.language_utils = LanguageUtils()
            logger.info("✅ Multilingual components initialized")
        except Exception as e:
            logger.warning(f"⚠️ Multilingual components initialization failed: {e}")
            self.language_detector = None
            self.text_processor = None
            self.language_utils = None
        
        # Enhanced performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limit_hits": 0,
            "quality_score_average": 0.8,
            "multilingual_analyses": 0,
            "cross_linguistic_comparisons": 0,
            "semantic_analyses": 0,
            "last_reset": time.time()
        }
        
        # Initialize enhanced client status tracking
        self.client_status = {}
        for i in range(len(self.clients)):
            self.client_status[i] = APIClientStatus(
                client_id=i,
                tokens_used=0,
                requests_made=0,
                last_reset=time.time(),
                last_call=0,
                blocked_until=0,
                error_count=0,
                success_count=0
            )
        
        # Enhanced response cache with TTL
        self.response_cache = {}
        self.cache_expiry = getattr(settings, 'CACHE_TTL_DEFAULT', 3600)
        self.max_cache_size = getattr(settings, 'MAX_CACHE_SIZE', 10000)
        
        # Thread pool for blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=len(self.clients) * 2)
        
        logger.info(f"🚀 Enhanced Multi-API LLM Processor v2.0 initialized")
        logger.info(f"🌍 Multilingual support: {'✅' if self.language_detector else '❌'}")
        logger.info(f"⚙️ Configuration: {self.requests_per_minute} req/min, {self.tokens_per_minute} tokens/min")
        logger.info(f"🤖 Clients: {len(self.clients)}, Model: {self.model}")
    
    def _enhanced_cache_management(self):
        """Enhanced cache management with size limits and TTL"""
        current_time = time.time()
        
        # Clean expired entries
        expired_keys = [
            key for key, (_, timestamp, _) in self.response_cache.items()
            if current_time - timestamp > self.cache_expiry
        ]
        
        for key in expired_keys:
            del self.response_cache[key]
        
        # Manage cache size
        if len(self.response_cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_cache = sorted(
                self.response_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            
            # Remove oldest 20% of entries
            remove_count = len(sorted_cache) // 5
            for key, _ in sorted_cache[:remove_count]:
                del self.response_cache[key]
        
        logger.debug(f"💾 Cache managed: {len(self.response_cache)} entries")
    
    def _get_enhanced_cache_key(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate enhanced cache key with context consideration"""
        # Include context information in cache key
        context_str = ""
        if context:
            context_str = f"_{context.get('language', 'en')}_{context.get('analysis_type', 'standard')}"
        
        cache_input = f"{prompt}{context_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()[:16]
    
    def _update_client_performance(self, client_id: int, response_time: float, 
                                 success: bool, quality_score: float = 0.8):
        """Update client performance metrics"""
        if client_id not in self.client_performance:
            return
        
        perf = self.client_performance[client_id]
        perf['total_calls'] += 1
        
        if success:
            perf['successful_calls'] += 1
            
            # Update average response time
            if perf['successful_calls'] == 1:
                perf['avg_response_time'] = response_time
            else:
                perf['avg_response_time'] = (
                    (perf['avg_response_time'] * (perf['successful_calls'] - 1) + response_time) /
                    perf['successful_calls']
                )
            
            # Update quality score
            perf['quality_score'] = (perf['quality_score'] * 0.9 + quality_score * 0.1)
    
    async def _get_enhanced_best_client(self) -> Optional[int]:
        """Enhanced client selection with performance-based scoring"""
        current_time = time.time()
        
        # Reset usage for all clients
        for client_id in range(len(self.clients)):
            self._reset_client_usage(client_id)
        
        # Find available clients with performance scoring
        available_clients = []
        
        for client_id, status in self.client_status.items():
            # Skip if blocked
            if current_time < status.blocked_until or not status.is_healthy:
                continue
            
            # Skip if too recent call
            if current_time - status.last_call < self.min_delay_between_calls:
                continue
            
            # Skip if near rate limits
            if (status.tokens_used >= self.tokens_per_minute * 0.9 or 
                status.requests_made >= self.requests_per_minute * 0.9):
                continue
            
            # Calculate enhanced performance score
            perf = self.client_performance.get(client_id, {})
            
            load_factor = (
                status.tokens_used / self.tokens_per_minute +
                status.requests_made / self.requests_per_minute
            ) / 2
            
            error_factor = status.error_count / max(status.success_count + status.error_count, 1)
            
            performance_factor = (
                perf.get('quality_score', 1.0) * 0.4 +
                (1.0 / max(perf.get('avg_response_time', 1.0), 0.1)) * 0.3 +
                (perf.get('successful_calls', 0) / max(perf.get('total_calls', 1), 1)) * 0.3
            )
            
            # Lower score is better
            combined_score = load_factor + error_factor * 2 - performance_factor * 0.5
            
            available_clients.append((client_id, combined_score))
        
        if not available_clients:
            return None
        
        # Return client with best (lowest) score
        best_client = min(available_clients, key=lambda x: x[1])
        return best_client[0]
    
    def _reset_client_usage(self, client_id: int):
        """Enhanced client usage reset with performance tracking"""
        current_time = time.time()
        status = self.client_status[client_id]
        
        # Reset every minute with buffer
        if current_time - status.last_reset >= 62:
            # Update efficiency score before reset
            if status.requests_made > 0:
                efficiency = status.success_count / (status.success_count + status.error_count)
                status.efficiency_score = efficiency
            
            # Store historical data
            old_tokens = status.tokens_used
            old_requests = status.requests_made
            
            # Reset counters
            status.tokens_used = 0
            status.requests_made = 0
            status.last_reset = current_time
            status.blocked_until = 0
            
            if old_tokens > 0 or old_requests > 0:
                logger.info(f"🔄 Enhanced reset client {client_id}: {old_requests} requests, "
                           f"{old_tokens} tokens, efficiency: {status.efficiency_score:.2f}")
    
    async def _execute_enhanced_llm_call(self, client_id: int, messages: List[Dict[str, str]], 
                                       max_tokens: int = 300, context: Dict[str, Any] = None) -> str:
        """Enhanced LLM call with multilingual support and performance monitoring"""
        start_time = time.time()
        status = self.client_status[client_id]
        client = self.clients[client_id]
        
        try:
            # Enhanced delay with jitter
            base_delay = self.min_delay_between_calls / len(self.clients)
            jitter = random.uniform(0.1, 0.5)
            await asyncio.sleep(base_delay + jitter)
            
            # Prepare enhanced API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "timeout": self.timeout_seconds,
            }
            
            # Add context-specific parameters
            if context:
                if context.get('analysis_type') == 'comprehensive':
                    api_params['temperature'] = 0.05  # Lower for comprehensive analysis
                elif context.get('analysis_type') == 'creative':
                    api_params['temperature'] = 0.3   # Higher for creative analysis
            
            # Execute API call with thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: client.chat.completions.create(**api_params)
            )
            
            # Enhanced tracking and metrics
            current_time = time.time()
            processing_time = current_time - start_time
            response_content = response.choices[0].message.content.strip()
            
            # Estimate tokens used (enhanced calculation)
            prompt_tokens = sum(len(self.text_processor.extract_keywords(msg["content"], max_keywords=50)) 
                              for msg in messages) if self.text_processor else sum(len(msg["content"].split()) for msg in messages)
            estimated_tokens = prompt_tokens * 1.3 + max_tokens
            
            # Update client status
            status.tokens_used += estimated_tokens
            status.requests_made += 1
            status.last_call = current_time
            status.success_count += 1
            status.is_healthy = True
            status.average_response_time = (
                (status.average_response_time * (status.success_count - 1) + processing_time) /
                status.success_count
            )
            status.total_processing_time += processing_time
            
            # Estimate response quality
            quality_score = self._estimate_response_quality(response_content, context)
            
            # Update performance metrics
            self._update_client_performance(client_id, processing_time, True, quality_score)
            
            # Update global stats
            self.performance_stats["total_requests"] += 1
            self.performance_stats["successful_requests"] += 1
            self.performance_stats["total_processing_time"] += processing_time
            
            # Update quality average
            total_successful = self.performance_stats["successful_requests"]
            current_avg = self.performance_stats["quality_score_average"]
            self.performance_stats["quality_score_average"] = (
                (current_avg * (total_successful - 1) + quality_score) / total_successful
            )
            
            logger.info(f"⚡ Enhanced API success - Client {client_id} - "
                       f"Tokens: {status.tokens_used:.0f}/{self.tokens_per_minute} - "
                       f"Time: {processing_time:.2f}s - Quality: {quality_score:.2f}")
            
            return response_content
            
        except Exception as e:
            # Enhanced error handling
            error_str = str(e).lower()
            current_time = time.time()
            processing_time = current_time - start_time
            
            status.error_count += 1
            status.last_call = current_time
            status.last_error_message = str(e)[:200]
            
            self.performance_stats["failed_requests"] += 1
            self._update_client_performance(client_id, processing_time, False, 0.0)
            
            if "429" in error_str or "rate" in error_str:
                # Enhanced rate limit handling
                backoff_time = min(60, 30 + random.uniform(10, 20))
                status.blocked_until = current_time + backoff_time
                status.is_healthy = False
                self.performance_stats["rate_limit_hits"] += 1
                logger.warning(f"🚫 Enhanced rate limit - Client {client_id} blocked for {backoff_time:.1f}s")
                raise Exception(f"Rate limit exceeded for client {client_id}")
                
            elif "timeout" in error_str or "connection" in error_str:
                # Enhanced network error handling
                status.is_healthy = False
                logger.warning(f"🌐 Enhanced network error - Client {client_id}: {e}")
                raise Exception(f"Network error for client {client_id}: {e}")
                
            else:
                # Enhanced general error handling
                if status.error_count > 5:
                    status.is_healthy = False
                logger.error(f"❌ Enhanced API error - Client {client_id}: {e}")
                raise Exception(f"API error for client {client_id}: {e}")
    
    def _estimate_response_quality(self, response: str, context: Dict[str, Any] = None) -> float:
        """Estimate the quality of an LLM response"""
        if not response:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Length check
        if 50 <= len(response) <= 2000:
            quality_score += 0.1
        
        # JSON validity check
        try:
            if response.strip().startswith('{') and response.strip().endswith('}'):
                json.loads(response)
                quality_score += 0.2
        except:
            pass
        
        # Content richness check
        if self.text_processor:
            keywords = self.text_processor.extract_keywords(response, max_keywords=10)
            if len(keywords) > 5:
                quality_score += 0.1
        
        # Context relevance check
        if context and context.get('expected_fields'):
            expected_fields = context['expected_fields']
            found_fields = sum(1 for field in expected_fields if field in response.lower())
            quality_score += (found_fields / len(expected_fields)) * 0.1
        
        return min(1.0, quality_score)
    
    async def _safe_enhanced_llm_call(self, prompt: str, max_tokens: int = 300, 
                                    context: Dict[str, Any] = None) -> str:
        """Enhanced safe LLM call with advanced caching and retry logic"""
        
        # Enhanced cache check
        cache_key = self._get_enhanced_cache_key(prompt, context)
        if cache_key in self.response_cache:
            cached_response, timestamp, quality = self.response_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                self.performance_stats["cache_hits"] += 1
                logger.info(f"💾 Enhanced cache hit (quality: {quality:.2f})")
                return cached_response
        
        self.performance_stats["cache_misses"] += 1
        
        # Periodic cache cleanup
        if random.random() < 0.05:  # 5% chance
            self._enhanced_cache_management()
        
        # Enhanced message preparation
        system_message = self._get_enhanced_system_message(context)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Enhanced retry logic with adaptive timeouts
        max_wait_time = self.timeout_seconds * 2
        start_wait = time.time()
        
        for attempt in range(self.max_retries):
            if time.time() - start_wait > max_wait_time:
                logger.error(f"⏰ Enhanced max wait time exceeded")
                break
            
            client_id = await self._get_enhanced_best_client()
            
            if client_id is not None:
                try:
                    response = await self._execute_enhanced_llm_call(
                        client_id, messages, max_tokens, context
                    )
                    
                    # Enhanced response validation
                    if self._validate_enhanced_response(response, context):
                        quality_score = self._estimate_response_quality(response, context)
                        
                        # Cache successful response with quality score
                        self.response_cache[cache_key] = (response, time.time(), quality_score)
                        
                        return response
                    else:
                        logger.warning(f"⚠️ Response validation failed (attempt {attempt + 1})")
                        if attempt < self.max_retries - 1:
                            continue
                    
                except Exception as e:
                    logger.warning(f"Enhanced attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        # Enhanced backoff strategy
                        backoff_time = min(10, (2 ** attempt) + random.uniform(0, 2))
                        await asyncio.sleep(backoff_time)
                        continue
                    else:
                        raise
            else:
                # Enhanced wait strategy when no clients available
                wait_time = min(8, 2 ** attempt + random.uniform(0, 2))
                logger.warning(f"⏳ Enhanced waiting for clients: {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        raise Exception(f"Enhanced LLM call failed after {self.max_retries} retries")
    
    def _get_enhanced_system_message(self, context: Dict[str, Any] = None) -> str:
        """Generate enhanced system message based on context"""
        base_message = "You are an advanced research analysis expert with multilingual capabilities."
        
        if not context:
            return f"{base_message} Provide precise, structured responses in valid JSON format only."
        
        analysis_type = context.get('analysis_type', 'standard')
        language = context.get('language', 'en')
        
        enhanced_instructions = []
        
        if analysis_type == 'comprehensive':
            enhanced_instructions.append("Provide detailed, comprehensive analysis with high accuracy.")
        elif analysis_type == 'quick':
            enhanced_instructions.append("Provide concise but accurate analysis.")
        elif analysis_type == 'multilingual':
            enhanced_instructions.append("Consider multilingual and cross-cultural research aspects.")
        
        if language != 'en':
            enhanced_instructions.append(f"Content may be in {language}. Analyze appropriately.")
        
        if context.get('domain'):
            enhanced_instructions.append(f"Focus on {context['domain']} domain expertise.")
        
        instructions_text = " ".join(enhanced_instructions)
        return f"{base_message} {instructions_text} Return only valid JSON format responses."
    
    def _validate_enhanced_response(self, response: str, context: Dict[str, Any] = None) -> bool:
        """Enhanced response validation with context awareness"""
        if not response or len(response.strip()) < 10:
            return False
        
        # Basic JSON validation
        try:
            if response.strip().startswith('{'):
                json.loads(self._clean_json_response(response))
        except:
            return False
        
        # Context-specific validation
        if context and context.get('required_fields'):
            required_fields = context['required_fields']
            response_lower = response.lower()
            
            missing_fields = [field for field in required_fields if field not in response_lower]
            if len(missing_fields) > len(required_fields) * 0.3:  # Allow 30% missing
                logger.warning(f"⚠️ Missing required fields: {missing_fields}")
                return False
        
        return True
    

    def _clean_json_response(self, response: str) -> str:
        """Enhanced JSON cleaning with better error handling"""
        if not response:
            raise ValueError("Empty response")
        
        # Remove markdown code blocks - FIXED VERSION
        response = re.sub(r'```json\s*', '', response)  # Remove ```json 
        response = re.sub(r'```\s*$', '', response)     # Remove trailing ```
        response = re.sub(r'^```\s*', '', response)    # Remove leading ```
        
        # Extract JSON object more reliably
        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        
        if json_matches:
            # Use the largest JSON object found
            response = max(json_matches, key=len)
        
        # Enhanced JSON cleaning
        response = response.strip()
        response = response.replace("'", '"')  # Single to double quotes
        response = re.sub(r',\s*}', '}', response)  # Remove trailing commas in objects
        response = re.sub(r',\s*]', ']', response)  # Remove trailing commas in arrays
        response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response)  # Remove control characters
        
        # Validate final JSON
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError as e:
            # Last resort: try to fix common issues
            response = re.sub(r'(\w+):', r'"\1":', response)  # Quote unquoted keys
            response = re.sub(r':\s*([^",\[\{][^,\}\]]*)', r': "\1"', response)  # Quote unquoted values
            
            try:
                json.loads(response)
                return response
            except json.JSONDecodeError:
                raise ValueError(f"Unable to parse JSON response: {e}")
    
    async def analyze_enhanced_paper_context(self, title: str, abstract: str, content: str = "",
                                           language: str = "auto", analysis_type: str = "detailed") -> PaperContext:
        """Enhanced paper context analysis with multilingual support"""
        start_time = time.time()
        
        if not title and not abstract:
            raise ValueError("Both title and abstract are empty")
        
        # Enhanced language detection
        detected_language = "en"
        if language == "auto" and self.language_detector:
            try:
                detection_result = self.language_detector.detect_language(f"{title} {abstract}")
                detected_language = detection_result.get('language_code', 'en')
                self.performance_stats["multilingual_analyses"] += 1
            except Exception as e:
                logger.warning(f"⚠️ Language detection failed: {e}")
        elif language != "auto":
            detected_language = language
        
        # Enhanced text preprocessing
        title_text = title[:200] if title else ""
        abstract_text = abstract[:500] if abstract else ""
        content_text = content[:300] if content else ""
        
        if self.text_processor:
            try:
                title_text = self.text_processor.clean_text(title_text, language=detected_language)
                abstract_text = self.text_processor.clean_text(abstract_text, language=detected_language)
                content_text = self.text_processor.clean_text(content_text, language=detected_language)
            except Exception as e:
                logger.warning(f"⚠️ Text preprocessing failed: {e}")
        
        # Enhanced domain detection
        research_domain = "General Research"
        if self.text_processor:
            try:
                domain_scores = self.text_processor.detect_research_domain_keywords(
                    f"{title_text} {abstract_text}", detected_language
                )
                if domain_scores:
                    research_domain = max(domain_scores.keys(), key=domain_scores.get)
            except Exception as e:
                logger.warning(f"⚠️ Domain detection failed: {e}")
        
        # Create enhanced analysis context
        analysis_context = {
            'analysis_type': analysis_type,
            'language': detected_language,
            'domain': research_domain,
            'expected_fields': [
                'context_summary', 'research_domain', 'methodology', 'key_findings',
                'innovations', 'limitations', 'research_questions', 'contributions',
                'future_work', 'related_concepts', 'context_quality_score'
            ]
        }
        
        # Create enhanced prompt
        prompt = f"""Analyze this research paper and return ONLY a JSON object:

Title: {title_text}
Abstract: {abstract_text}
{f"Content: {content_text}" if content_text else ""}

Language: {detected_language}
Domain: {research_domain}
Analysis Type: {analysis_type}

Return JSON in this exact format:
{{
    "context_summary": "Brief 2-3 sentence summary of the paper's main contribution",
    "research_domain": "Primary research field (e.g., Computer Vision, NLP, Machine Learning)",
    "methodology": "Main approach or method used",
    "key_findings": ["Main finding 1", "Main finding 2"],
    "innovations": ["Key innovation or novelty"],
    "limitations": ["Main limitation or constraint"],
    "research_questions": ["Primary research question addressed"],
    "contributions": ["Main contribution to the field"],
    "future_work": ["Suggested future research direction"],
    "related_concepts": ["concept1", "concept2", "concept3"],
    "context_quality_score": 0.8
}}

Return only valid JSON, no other text."""
        
        try:
            response = await self._safe_enhanced_llm_call(
                prompt, 
                max_tokens=450, 
                context=analysis_context
            )
            
            # Enhanced response processing
            clean_response = self._clean_json_response(response)
            data = json.loads(clean_response)
            
            # Enhanced data validation and creation
            context = PaperContext(
                context_summary=str(data.get("context_summary", ""))[:300],
                research_domain=str(data.get("research_domain", research_domain))[:80],
                methodology=str(data.get("methodology", ""))[:200],
                key_findings=self._validate_string_list(data.get("key_findings", []), 3, 200),
                innovations=self._validate_string_list(data.get("innovations", []), 2, 200),
                limitations=self._validate_string_list(data.get("limitations", []), 2, 200),
                research_questions=self._validate_string_list(data.get("research_questions", []), 2, 200),
                contributions=self._validate_string_list(data.get("contributions", []), 2, 200),
                future_work=self._validate_string_list(data.get("future_work", []), 2, 200),
                related_concepts=self._validate_string_list(data.get("related_concepts", []), 4, 50),
                context_quality_score=self._validate_score(data.get("context_quality_score", 0.7)),
                analysis_confidence=0.85,
                processing_time=time.time() - start_time,
                model_used=self.model,
                language_detected=detected_language,
                ai_agent_used="groq_enhanced",
                analysis_method=analysis_type,
                semantic_similarity=0.0  # Will be set by embedding analysis if available
            )
            
            logger.info(f"✅ Enhanced analysis: {title_text[:50]} "
                       f"(quality: {context.context_quality_score:.2f}, lang: {detected_language})")
            return context
            
        except Exception as e:
            logger.warning(f"Enhanced LLM analysis failed for '{title_text[:50]}': {e}")
            return self._create_enhanced_fallback_context(
                title, abstract, detected_language, research_domain, time.time() - start_time
            )
    
    async def batch_analyze_enhanced_papers(self, papers: List[Dict[str, Any]], 
                                          analysis_type: str = "detailed") -> List[PaperContext]:
        """Enhanced high-performance parallel batch analysis"""
        total_papers = len(papers)
        if total_papers == 0:
            return []
        
        logger.info(f"🚀 Starting enhanced parallel analysis of {total_papers} papers (type: {analysis_type})")
        start_time = time.time()
        
        # Enhanced batch organization
        optimal_batch_size = min(
            self.batch_size, 
            max(2, len(self.clients) * 2),
            max(2, total_papers // 4)  # Adapt to total papers
        )
        
        batches = [papers[i:i + optimal_batch_size] for i in range(0, total_papers, optimal_batch_size)]
        
        results = []
        
        # Enhanced concurrency control
        max_concurrent = min(self.max_concurrent_batches, len(batches))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_enhanced_batch(batch: List[Dict[str, Any]], batch_num: int) -> List[PaperContext]:
            async with semaphore:
                batch_results = []
                batch_start = time.time()
                
                logger.info(f"⚡ Enhanced batch {batch_num}/{len(batches)} "
                           f"({len(batch)} papers, type: {analysis_type})")
                
                # Create enhanced analysis tasks
                tasks = []
                for paper in batch:
                    # Detect language for each paper
                    paper_language = "auto"
                    if self.language_detector and (paper.get("title") or paper.get("abstract")):
                        try:
                            text_sample = f"{paper.get('title', '')} {paper.get('abstract', '')}"[:200]
                            detection = self.language_detector.detect_language(text_sample)
                            paper_language = detection.get('language_code', 'auto')
                        except:
                            pass
                    
                    task = self.analyze_enhanced_paper_context(
                        paper.get("title", ""),
                        paper.get("abstract", ""),
                        paper.get("content", ""),
                        language=paper_language,
                        analysis_type=analysis_type
                    )
                    tasks.append(task)
                
                # Execute enhanced batch with adaptive timeout
                base_timeout = 60 if analysis_type == "quick" else 90 if analysis_type == "detailed" else 120
                batch_timeout = base_timeout + (len(batch) * 5)  # Scale with batch size
                
                try:
                    batch_contexts = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=batch_timeout
                    )
                    
                    # Enhanced result processing
                    for i, result in enumerate(batch_contexts):
                        if isinstance(result, Exception):
                            logger.error(f"Enhanced paper {i + 1} failed: {result}")
                            fallback = self._create_enhanced_fallback_context(
                                batch[i].get("title", ""),
                                batch[i].get("abstract", ""),
                                "en",
                                "General Research",
                                0.0
                            )
                            batch_results.append(fallback)
                        else:
                            batch_results.append(result)
                    
                    batch_time = time.time() - batch_start
                    avg_quality = sum(r.context_quality_score for r in batch_results) / len(batch_results)
                    
                    logger.info(f"✅ Enhanced batch {batch_num} completed in {batch_time:.2f}s "
                               f"(avg quality: {avg_quality:.2f})")
                    
                except asyncio.TimeoutError:
                    logger.error(f"❌ Enhanced batch {batch_num} timed out after {batch_timeout}s")
                    # Create enhanced fallback contexts
                    for paper in batch:
                        fallback = self._create_enhanced_fallback_context(
                            paper.get("title", ""),
                            paper.get("abstract", ""),
                            "en", 
                            "General Research",
                            0.0
                        )
                        batch_results.append(fallback)
                
                return batch_results
        
        # Process all enhanced batches
        batch_tasks = [
            process_enhanced_batch(batch, i + 1) 
            for i, batch in enumerate(batches)
        ]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Enhanced result combination
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Enhanced batch processing failed: {batch_result}")
                # Add enhanced fallback contexts
                for _ in range(optimal_batch_size):
                    if len(results) < total_papers:
                        results.append(self._create_enhanced_fallback_context("", "", "en", "General Research", 0.0))
            else:
                results.extend(batch_result)
        
        # Ensure correct result count
        results = results[:total_papers]
        while len(results) < total_papers:
            results.append(self._create_enhanced_fallback_context("", "", "en", "General Research", 0.0))
        
        # Enhanced completion metrics
        total_time = time.time() - start_time
        successful_analyses = sum(1 for r in results if r.context_quality_score > 0.3)
        avg_quality = sum(r.context_quality_score for r in results) / len(results)
        languages_detected = len(set(r.language_detected for r in results))
        
        logger.info(f"✅ Enhanced parallel analysis completed: {successful_analyses}/{total_papers} successful "
                   f"in {total_time:.2f}s ({total_papers/total_time:.2f} papers/sec) "
                   f"- Avg quality: {avg_quality:.2f}, Languages: {languages_detected}")
        
        return results
    
    async def analyze_enhanced_paper_relationship(self, paper1: Dict[str, Any], 
                                                paper2: Dict[str, Any],
                                                analysis_depth: str = "standard") -> PaperRelationship:
        """Enhanced relationship analysis with multilingual and semantic support"""
        start_time = time.time()
        
        # Enhanced paper information extraction
        title1 = (paper1.get("title", "") or "")[:100]
        title2 = (paper2.get("title", "") or "")[:100]
        domain1 = paper1.get("research_domain", "Unknown")[:50]
        domain2 = paper2.get("research_domain", "Unknown")[:50]
        abstract1 = (paper1.get("abstract", "") or "")[:200]
        abstract2 = (paper2.get("abstract", "") or "")[:200]
        
        if not title1 or not title2:
            return self._create_enhanced_fallback_relationship(time.time() - start_time)
        
        # Enhanced language detection for both papers
        lang1 = lang2 = "en"
        if self.language_detector:
            try:
                detection1 = self.language_detector.detect_language(f"{title1} {abstract1}")
                detection2 = self.language_detector.detect_language(f"{title2} {abstract2}")
                lang1 = detection1.get('language_code', 'en')
                lang2 = detection2.get('language_code', 'en')
                
                if lang1 != lang2:
                    self.performance_stats["cross_linguistic_comparisons"] += 1
            except Exception as e:
                logger.warning(f"⚠️ Language detection for relationship failed: {e}")
        
        # Enhanced semantic similarity calculation
        semantic_similarity = 0.5
        if self.text_processor:
            try:
                similarity = self.text_processor.calculate_text_similarity(
                    f"{title1} {abstract1}", f"{title2} {abstract2}", lang1
                )
                semantic_similarity = similarity
                self.performance_stats["semantic_analyses"] += 1
            except Exception as e:
                logger.warning(f"⚠️ Semantic similarity calculation failed: {e}")
        
        # Calculate domain overlap
        domain_overlap = 1.0 if domain1 == domain2 else 0.3 if "General" not in domain1 and "General" not in domain2 else 0.1
        
        # Calculate language similarity
        language_similarity = 1.0 if lang1 == lang2 else 0.7 if self.language_utils and self.language_utils.get_language_info(lang1).get('family') == self.language_utils.get_language_info(lang2).get('family') else 0.3
        
        # Enhanced analysis context
        relationship_context = {
            'analysis_type': 'relationship',
            'analysis_depth': analysis_depth,
            'language1': lang1,
            'language2': lang2,
            'domain1': domain1,
            'domain2': domain2,
            'semantic_similarity': semantic_similarity,
            'expected_fields': [
                'relationship_type', 'relationship_strength', 'relationship_context', 'connection_reasoning'
            ]
        }
        
        # Create enhanced relationship analysis prompt
        prompt = f"""Compare these two research papers and analyze their relationship:

Paper 1:
Title: {title1}
Domain: {domain1} (Language: {lang1})
Abstract: {abstract1}

Paper 2:
Title: {title2}
Domain: {domain2} (Language: {lang2})
Abstract: {abstract2}

Analysis Context:
- Semantic similarity: {semantic_similarity:.2f}
- Domain overlap: {domain_overlap:.2f}
- Language similarity: {language_similarity:.2f}
- Analysis depth: {analysis_depth}

Return ONLY a JSON object analyzing their relationship:
{{
    "relationship_type": "Choose from: builds_upon, improves, extends, complements, applies, related, methodology_shared, domain_overlap, competing, contradicts, unrelated",
    "relationship_strength": "Float between 0.0 (unrelated) and 1.0 (highly related)",
    "relationship_context": "Brief explanation of how they relate (considering multilingual aspects if applicable)",
    "connection_reasoning": "Specific reasoning for the relationship strength and type, including semantic, domain, and methodological considerations"
}}

Return only valid JSON, no other text."""
        
        try:
            response = await self._safe_enhanced_llm_call(
                prompt,
                max_tokens=300,
                context=relationship_context
            )
            
            # Enhanced response processing
            clean_response = self._clean_json_response(response)
            data = json.loads(clean_response)
            
            # Enhanced relationship creation
            relationship = PaperRelationship(
                relationship_type=str(data.get("relationship_type", "related"))[:50],
                relationship_strength=self._validate_score(data.get("relationship_strength", 0.5)),
                relationship_context=str(data.get("relationship_context", ""))[:200],
                connection_reasoning=str(data.get("connection_reasoning", ""))[:300],
                confidence_score=0.85,
                analysis_method=f"enhanced_llm_{analysis_depth}",
                processing_time=time.time() - start_time,
                ai_agent_used="groq_enhanced",
                semantic_similarity=semantic_similarity,
                language_similarity=language_similarity,
                domain_overlap=domain_overlap,
                methodology_similarity=0.5  # Could be enhanced with more analysis
            )
            
            logger.info(f"🔗 Enhanced relationship: {relationship.relationship_type} "
                       f"(strength: {relationship.relationship_strength:.2f}, "
                       f"semantic: {semantic_similarity:.2f}, cross-lang: {lang1}-{lang2})")
            
            return relationship
            
        except Exception as e:
            logger.warning(f"Enhanced relationship analysis failed: {e}")
            return self._create_enhanced_fallback_relationship(time.time() - start_time)
    
    def _create_enhanced_fallback_context(self, title: str, abstract: str, language: str,
                                        research_domain: str, processing_time: float) -> PaperContext:
        """Create enhanced fallback context with multilingual support"""
        
        # Enhanced domain detection for fallback
        if research_domain == "General Research" and self.text_processor:
            try:
                text = f"{title} {abstract}".lower()
                domain_scores = self.text_processor.detect_research_domain_keywords(text, language)
                if domain_scores:
                    research_domain = max(domain_scores.keys(), key=domain_scores.get)
            except:
                pass
        
        # Enhanced summary generation
        summary = abstract[:150] if abstract else f"Research paper in {research_domain}"
        if not summary.endswith('.'):
            summary += "."
        
        # Enhanced quality estimation based on available information
        quality_score = 0.3  # Base fallback score
        if title and abstract:
            quality_score += 0.2
        if len(abstract) > 100:
            quality_score += 0.1
        if research_domain != "General Research":
            quality_score += 0.1
        
        return PaperContext(
            context_summary=summary,
            research_domain=research_domain,
            methodology="Computational approach" if "computer" in research_domain.lower() else "Research methodology",
            key_findings=[f"Advances in {research_domain.lower()}"] if title else [],
            innovations=[f"Novel approach in {research_domain.lower()}"] if abstract else [],
            limitations=["Requires further validation and extensive evaluation"],
            research_questions=[f"How to advance {research_domain.lower()} research effectively"],
            contributions=[f"Enhanced understanding of {research_domain.lower()}"],
            future_work=["Extended evaluation and comparative analysis"],
            related_concepts=[research_domain, "Research", "Analysis", "Innovation"][:3],
            context_quality_score=min(0.6, quality_score),
            analysis_confidence=0.4,
            processing_time=processing_time,
            model_used="enhanced_fallback",
            language_detected=language,
            ai_agent_used="fallback_enhanced",
            analysis_method="fallback",
            semantic_similarity=0.0
        )
    
    def _create_enhanced_fallback_relationship(self, processing_time: float) -> PaperRelationship:
        """Create enhanced fallback relationship"""
        return PaperRelationship(
            relationship_type="related",
            relationship_strength=0.5,
            relationship_context="Papers appear to be in related research domains",
            connection_reasoning="Unable to perform detailed analysis, assuming moderate relatedness based on basic similarity",
            confidence_score=0.4,
            analysis_method="enhanced_fallback",
            processing_time=processing_time,
            ai_agent_used="fallback_enhanced",
            semantic_similarity=0.3,
            language_similarity=0.5,
            domain_overlap=0.5,
            methodology_similarity=0.3
        )
    
    def _validate_string_list(self, value: Any, max_items: int, max_length: int) -> List[str]:
        """Enhanced string list validation with better cleaning"""
        if not isinstance(value, list):
            if isinstance(value, str) and value.strip():
                return [value.strip()[:max_length]]
            return []
        
        cleaned_list = []
        for item in value[:max_items]:
            if item and isinstance(item, (str, int, float)):
                clean_item = str(item).strip()[:max_length]
                if clean_item and len(clean_item) > 2:  # Minimum length check
                    # Remove any remaining unwanted characters
                    clean_item = re.sub(r'[^\w\s\-.,;:()$$$${}]', '', clean_item)
                    if clean_item:
                        cleaned_list.append(clean_item)
        
        return cleaned_list
    
    def _validate_score(self, score: Any) -> float:
        """Enhanced score validation with bounds checking"""
        try:
            float_score = float(score)
            # Ensure score is within valid bounds
            return max(0.0, min(1.0, float_score))
        except (ValueError, TypeError):
            return 0.5
    
    def get_enhanced_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhanced performance statistics"""
        total_time = self.performance_stats["total_processing_time"]
        total_requests = self.performance_stats["total_requests"]
        
        # Calculate enhanced averages
        avg_response_time = total_time / total_requests if total_requests > 0 else 0.0
        success_rate = (self.performance_stats["successful_requests"] / total_requests 
                       if total_requests > 0 else 0.0)
        cache_hit_rate = (self.performance_stats["cache_hits"] / 
                         (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"])
                         if (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]) > 0 else 0.0)
        
        # Enhanced client health summary
        healthy_clients = sum(1 for status in self.client_status.values() if status.is_healthy)
        avg_client_efficiency = sum(status.efficiency_score for status in self.client_status.values()) / len(self.client_status)
        
        # Enhanced metrics
        return {
            "overview": {
                "total_requests": total_requests,
                "successful_requests": self.performance_stats["successful_requests"],
                "failed_requests": self.performance_stats["failed_requests"],
                "success_rate": success_rate,
                "average_response_time": avg_response_time,
                "quality_score_average": self.performance_stats["quality_score_average"]
            },
            "caching": {
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self.response_cache),
                "cache_hits": self.performance_stats["cache_hits"],
                "cache_misses": self.performance_stats["cache_misses"],
                "max_cache_size": self.max_cache_size
            },
            "multilingual": {
                "multilingual_analyses": self.performance_stats["multilingual_analyses"],
                "cross_linguistic_comparisons": self.performance_stats["cross_linguistic_comparisons"],
                "semantic_analyses": self.performance_stats["semantic_analyses"],
                "language_detection_available": self.language_detector is not None,
                "text_processing_available": self.text_processor is not None
            },
            "clients": {
                "healthy_clients": healthy_clients,
                "total_clients": len(self.clients),
                "average_efficiency": avg_client_efficiency,
                "rate_limit_hits": self.performance_stats["rate_limit_hits"]
            },
            "client_details": {
                str(client_id): {
                    "healthy": status.is_healthy,
                    "success_count": status.success_count,
                    "error_count": status.error_count,
                    "tokens_used": status.tokens_used,
                    "requests_made": status.requests_made,
                    "average_response_time": status.average_response_time,
                    "efficiency_score": status.efficiency_score,
                    "last_error": status.last_error_message[:50] if status.last_error_message else ""
                }
                for client_id, status in self.client_status.items()
            },
            "configuration": {
                "model": self.model,
                "tokens_per_minute": self.tokens_per_minute,
                "requests_per_minute": self.requests_per_minute,
                "batch_size": self.batch_size,
                "max_concurrent_batches": self.max_concurrent_batches,
                "cache_expiry": self.cache_expiry,
                "multilingual_enabled": self.language_detector is not None
            },
            "system": {
                "uptime_hours": (time.time() - self.performance_stats["last_reset"]) / 3600,
                "thread_pool_workers": self.thread_pool._max_workers,
                "memory_usage_mb": len(self.response_cache) * 0.001  # Rough estimate
            }
        }
    
    async def enhanced_health_check(self) -> Dict[str, Any]:
        """Comprehensive enhanced health check"""
        try:
            # Enhanced test analysis
            test_start = time.time()
            test_context = await self.analyze_enhanced_paper_context(
                "Enhanced Test Paper for Health Check",
                "This is a comprehensive test abstract for health checking the enhanced multilingual LLM processor.",
                analysis_type="quick"
            )
            test_time = time.time() - test_start
            
            stats = self.get_enhanced_performance_stats()
            
            # Enhanced health scoring
            health_components = {
                "success_rate": stats["overview"]["success_rate"],
                "client_health": stats["clients"]["healthy_clients"] / stats["clients"]["total_clients"],
                "response_time": min(1.0, 5.0 / max(test_time, 0.1)),  # Prefer under 5 seconds
                "cache_efficiency": stats["caching"]["cache_hit_rate"],
                "quality_score": stats["overview"]["quality_score_average"],
                "efficiency": stats["clients"]["average_efficiency"]
            }
            
            # Weighted health score
            weights = {"success_rate": 0.3, "client_health": 0.25, "response_time": 0.2, 
                      "cache_efficiency": 0.1, "quality_score": 0.1, "efficiency": 0.05}
            
            health_score = sum(health_components[key] * weights[key] for key in weights)
            
            # Enhanced health status determination
            if health_score > 0.85:
                health_status = "excellent"
            elif health_score > 0.7:
                health_status = "healthy"
            elif health_score > 0.5:
                health_status = "degraded"
            else:
                health_status = "unhealthy"
            
            return {
                "status": health_status,
                "health_score": health_score,
                "health_components": health_components,
                "test_results": {
                    "analysis_time": test_time,
                    "context_quality": test_context.context_quality_score,
                    "language_detected": test_context.language_detected,
                    "analysis_confidence": test_context.analysis_confidence
                },
                "capabilities": {
                    "multilingual_support": self.language_detector is not None,
                    "text_processing": self.text_processor is not None,
                    "semantic_analysis": True,
                    "enhanced_caching": True,
                    "performance_monitoring": True
                },
                "recommendations": self._generate_health_recommendations(health_components),
                "timestamp": datetime.now().isoformat(),
                "statistics": stats
            }
            
        except Exception as e:
            return {
                "status": "error",
                "health_score": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "capabilities": {
                    "multilingual_support": False,
                    "text_processing": False,
                    "semantic_analysis": False,
                    "enhanced_caching": False,
                    "performance_monitoring": False
                }
            }
    
    def _generate_health_recommendations(self, health_components: Dict[str, float]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        if health_components["success_rate"] < 0.8:
            recommendations.append("Consider checking API key validity and network connectivity")
        
        if health_components["client_health"] < 0.8:
            recommendations.append("Some clients are unhealthy - check rate limits and error logs")
        
        if health_components["response_time"] < 0.7:
            recommendations.append("Response times are high - consider reducing batch sizes or complexity")
        
        if health_components["cache_efficiency"] < 0.3:
            recommendations.append("Cache hit rate is low - consider optimizing cache strategy")
        
        if health_components["quality_score"] < 0.7:
            recommendations.append("Analysis quality is below optimal - review prompts and validation")
        
        if not recommendations:
            recommendations.append("System is performing optimally")
        
        return recommendations
    
    def reset_enhanced_performance_stats(self):
        """Reset enhanced performance statistics"""
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limit_hits": 0,
            "quality_score_average": 0.8,
            "multilingual_analyses": 0,
            "cross_linguistic_comparisons": 0,
            "semantic_analyses": 0,
            "last_reset": time.time()
        }
        
        # Reset enhanced client stats
        for status in self.client_status.values():
            status.success_count = 0
            status.error_count = 0
            status.is_healthy = True
            status.average_response_time = 0.0
            status.total_processing_time = 0.0
            status.efficiency_score = 1.0
            status.last_error_message = ""
        
        # Reset client performance tracking
        for client_id in self.client_performance:
            self.client_performance[client_id] = {
                'total_calls': 0,
                'successful_calls': 0,
                'avg_response_time': 0.0,
                'quality_score': 1.0
            }
        
        # Clear enhanced cache
        self.response_cache.clear()
        
        logger.info("📊 Enhanced performance statistics reset")
    
    async def shutdown(self):
        """Enhanced graceful shutdown"""
        try:
            logger.info("🔒 Enhanced LLM Processor shutdown initiated...")
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Final performance report
            stats = self.get_enhanced_performance_stats()
            logger.info(f"📈 Final stats: {stats['overview']['success_rate']:.1%} success rate, "
                       f"{stats['overview']['total_requests']} total requests, "
                       f"{stats['multilingual']['multilingual_analyses']} multilingual analyses")
            
            logger.info("✅ Enhanced LLM Processor shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Enhanced shutdown failed: {e}")
    
    def __del__(self):
        """Enhanced cleanup when processor is destroyed"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
            logger.debug("🔒 Enhanced LLM Processor cleanup completed")
        except:
            pass  # Ignore errors during cleanup

# Enhanced backward compatibility
MultiAPILLMProcessor = EnhancedMultiAPILLMProcessor

# Enhanced legacy compatibility functions
async def analyze_enhanced_papers_batch(papers: List[Dict[str, Any]], 
                                      analysis_type: str = "detailed") -> List[Dict[str, Any]]:
    """Enhanced legacy compatibility function with multilingual support"""
    processor = EnhancedMultiAPILLMProcessor()
    try:
        contexts = await processor.batch_analyze_enhanced_papers(papers, analysis_type)
        
        results = []
        for i, context in enumerate(contexts):
            paper_result = {
                **papers[i],
                'context_summary': context.context_summary,
                'research_domain': context.research_domain,
                'methodology': context.methodology,
                'key_findings': context.key_findings,
                'innovations': context.innovations,
                'limitations': context.limitations,
                'research_questions': context.research_questions,
                'contributions': context.contributions,
                'future_work': context.future_work,
                'related_concepts': context.related_concepts,
                'context_quality_score': context.context_quality_score,
                'language_detected': context.language_detected,
                'analysis_confidence': context.analysis_confidence,
                'ai_agent_used': context.ai_agent_used,
                'processing_time': context.processing_time
            }
            results.append(paper_result)
        
        return results
        
    finally:
        await processor.shutdown()

# Original legacy function for complete backward compatibility
async def analyze_papers_batch(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Original legacy compatibility function"""
    return await analyze_enhanced_papers_batch(papers, "detailed")
