import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx
from dataclasses import dataclass
import time
import hashlib
import concurrent.futures
import random
import json

logger = logging.getLogger(__name__)

@dataclass
class TranslationResult:
    """Translation result with metadata"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    translation_service: str
    processing_time: float

class ImprovedGoogleTranslator:
    """Improved Google Translator with retry logic and rate limiting"""
    
    def __init__(self):
        self.last_request_time = 0
        self.min_delay = 1.0  # Minimum 1 second between requests
        self.session = None
        self.initialize_session()
    
    def initialize_session(self):
        """Initialize HTTP session for Google Translate"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        self.session = httpx.Client(
            headers=headers,
            timeout=30.0,
            follow_redirects=True
        )
    
    def _respect_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last + random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def translate(self, text: str, dest: str, src: str = 'auto') -> Optional[Dict]:
        """
        Translate using HTTP requests to Google Translate
        More reliable than googletrans library
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Respect rate limits
                self._respect_rate_limit()
                
                # Google Translate API endpoint
                url = "https://translate.googleapis.com/translate_a/single"
                
                params = {
                    'client': 'gtx',
                    'sl': src,
                    'tl': dest,
                    'dt': 't',
                    'q': text[:5000]  # Limit text length
                }
                
                logger.debug(f"🔄 Attempting translation {attempt + 1}/{max_retries}: {src} → {dest}")
                
                response = self.session.get(url, params=params)
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Parse Google's response format
                    if result_data and isinstance(result_data, list) and len(result_data) > 0:
                        translated_text = ""
                        
                        # Extract translated text from nested arrays
                        if isinstance(result_data[0], list):
                            for item in result_data[0]:
                                if isinstance(item, list) and len(item) > 0 and item[0]:
                                    translated_text += str(item[0])
                        
                        # Extract detected source language
                        detected_lang = src
                        if len(result_data) > 2 and result_data[2]:
                            detected_lang = result_data[2]
                        
                        if translated_text and translated_text.strip():
                            logger.info(f"✅ HTTP Google Translate success: {detected_lang} → {dest}")
                            return {
                                'text': translated_text.strip(),
                                'src': detected_lang,
                                'confidence': 0.95  # High confidence for successful HTTP translation
                            }
                
                logger.warning(f"⚠️ Translation attempt {attempt + 1} failed: HTTP {response.status_code}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    sleep_time = (2 ** attempt) + random.uniform(1, 3)
                    logger.info(f"⏰ Rate limited, waiting {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    continue
                
                # For other errors, try with different session
                if attempt < max_retries - 1:
                    logger.info("🔄 Reinitializing session for retry...")
                    self.initialize_session()
                    time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.warning(f"⚠️ Translation attempt {attempt + 1} error: {e}")
                
                if attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(sleep_time)
                    # Reinitialize session on error
                    try:
                        self.initialize_session()
                    except Exception as init_error:
                        logger.warning(f"Session reinit failed: {init_error}")
        
        logger.error(f"❌ All translation attempts failed for: {text[:50]}...")
        return None
    
    def close(self):
        """Close the HTTP session"""
        if self.session:
            self.session.close()

class TranslationService:
    """Advanced multilingual translation service with multiple backends"""
    
    # Target languages for research expansion
    TARGET_RESEARCH_LANGUAGES = [
        'en',  # English
        'zh',  # Chinese
        'de',  # German  
        'fr',  # French
        'ja',  # Japanese
        'ko',  # Korean
        'es',  # Spanish
        'ru',  # Russian
        'it',  # Italian
        'pt',  # Portuguese
        'ar',  # Arabic
        'hi'   # Hindi
    ]
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        # ✅ FIXED: Initialize improved translator
        try:
            self.google_translator = ImprovedGoogleTranslator()
            self.translation_available = True
            logger.info("✅ Improved Google Translator initialized successfully")
            
            # Test with a simple translation
            test_result = self.google_translator.translate('test', 'zh', 'en')
            if test_result and test_result.get('text'):
                logger.info("✅ Translation test successful")
            else:
                logger.warning("⚠️ Translation test failed, but service will attempt translations")
                
        except Exception as e:
            logger.warning(f"⚠️ Google Translator initialization failed: {e}")
            self.google_translator = None
            self.translation_available = False
        
        self.translation_cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = 10000
        
        # Performance tracking
        self.translation_stats = {
            'total_translations': 0,
            'cache_hits': 0,
            'google_translations': 0,
            'failed_translations': 0,
            'successful_translations': 0,
            'http_translations': 0  # New stat for HTTP method
        }
        
        logger.info("🌐 Enhanced Translation service initialized")
    
    def _get_cache_key(self, text: str, target_lang: str) -> str:
        """Generate cache key for translation"""
        key_string = f"v2_{text[:100]}_{target_lang}"  # v2 for new version
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached translation is still valid"""
        return (time.time() - timestamp) < self.cache_ttl
    
    def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.translation_cache.items()
            if not self._is_cache_valid(timestamp)
        ]
        
        for key in expired_keys:
            del self.translation_cache[key]
        
        # Limit cache size
        if len(self.translation_cache) > self.max_cache_size:
            sorted_items = sorted(
                self.translation_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            self.translation_cache = dict(sorted_items[-self.max_cache_size//2:])
    
    async def translate_text(self, text: str, target_language: str, 
                           source_language: str = 'auto') -> Optional[TranslationResult]:
        """
        Translate text to target language with comprehensive error handling
        ✅ FIXED: Now uses improved HTTP-based Google Translate
        """
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                logger.warning("❌ Empty text provided for translation")
                return None
            
            # ✅ FIXED: Check if translation is available
            if not self.translation_available or not self.google_translator:
                logger.warning("⚠️ Translation service unavailable - returning fallback")
                return self._create_fallback_result(text, target_language, start_time)
            
            # Skip translation if source and target are the same
            if source_language == target_language and source_language != 'auto':
                logger.info(f"🔄 Same language detected ({source_language}), skipping translation")
                return self._create_same_language_result(text, target_language, start_time)
            
            # Check cache first
            cache_key = self._get_cache_key(text, target_language)
            if cache_key in self.translation_cache:
                cached_result, timestamp = self.translation_cache[cache_key]
                if self._is_cache_valid(timestamp):
                    self.translation_stats['cache_hits'] += 1
                    logger.info(f"💾 Cache hit for translation to {target_language}")
                    return cached_result
            
            # Clean cache periodically
            if len(self.translation_cache) > self.max_cache_size * 1.2:
                self._clean_cache()
            
            # ✅ FIXED: Use ThreadPoolExecutor with improved translator
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(self._improved_google_translate_sync, text, target_language, source_language)
                    translated = future.result(timeout=20.0)  # Increased timeout for HTTP requests
                except concurrent.futures.TimeoutError:
                    logger.error(f"⏰ Translation timed out for: {text[:50]}...")
                    self.translation_stats['failed_translations'] += 1
                    return self._create_fallback_result(text, target_language, start_time)
                except Exception as e:
                    logger.error(f"❌ Translation executor failed: {e}")
                    self.translation_stats['failed_translations'] += 1
                    return self._create_fallback_result(text, target_language, start_time)
            
            # ✅ FIXED: Process translation result with proper validation
            if translated and isinstance(translated, dict) and 'text' in translated:
                processing_time = time.time() - start_time
                
                # Validate that we actually got a translation (not just original text)
                translated_text = str(translated['text']).strip()
                original_text = text.strip()
                
                # Check if translation is meaningfully different or same language
                if translated_text and (translated_text != original_text or source_language == target_language):
                    result = TranslationResult(
                        original_text=text[:500],
                        translated_text=translated_text,
                        source_language=str(translated.get('src', source_language)),
                        target_language=target_language,
                        confidence=float(translated.get('confidence', 0.95)),
                        translation_service='google_http',  # Updated service name
                        processing_time=processing_time
                    )
                    
                    # Cache successful translation
                    self.translation_cache[cache_key] = (result, time.time())
                    
                    self.translation_stats['total_translations'] += 1
                    self.translation_stats['google_translations'] += 1
                    self.translation_stats['successful_translations'] += 1
                    self.translation_stats['http_translations'] += 1
                    
                    logger.info(f"✅ Translated {translated.get('src', 'unknown')} → {target_language} "
                               f"({processing_time:.2f}s): '{original_text[:30]}...' → '{translated_text[:30]}...'")
                    
                    return result
                else:
                    logger.warning(f"⚠️ Translation returned same text: {translated_text}")
                    self.translation_stats['failed_translations'] += 1
                    return self._create_fallback_result(text, target_language, start_time)
            else:
                logger.error("❌ Invalid translation result format")
                self.translation_stats['failed_translations'] += 1
                return self._create_fallback_result(text, target_language, start_time)
            
        except Exception as e:
            logger.error(f"❌ Translation failed: {e}")
            self.translation_stats['failed_translations'] += 1
            return self._create_fallback_result(text, target_language, start_time)
    
    def _improved_google_translate_sync(self, text: str, target_lang: str, source_lang: str = 'auto') -> Optional[Dict]:
        """
        ✅ FIXED: Uses improved HTTP-based Google Translate
        Much more reliable than googletrans library
        """
        try:
            if not self.google_translator:
                logger.error("❌ No translator available")
                return None
            
            if not text or len(text.strip()) == 0:
                logger.error("❌ Empty text for translation")
                return None
            
            # Use improved translator
            result = self.google_translator.translate(text, target_lang, source_lang)
            
            if result and isinstance(result, dict):
                logger.debug(f"🌐 HTTP Google Translate success: {source_lang} → {target_lang}")
                return result
            else:
                logger.error("❌ HTTP Google Translate returned no result")
                return None
                
        except Exception as e:
            logger.error(f"❌ HTTP Google Translate sync failed: {e}")
            return None
    
    def _create_fallback_result(self, text: str, target_language: str, start_time: float) -> TranslationResult:
        """Create fallback translation result"""
        processing_time = time.time() - start_time
        
        return TranslationResult(
            original_text=text[:500],
            translated_text=text,  # Return original text as fallback
            source_language='auto',
            target_language=target_language,
            confidence=0.1,  # Low confidence for fallback
            translation_service='fallback',
            processing_time=processing_time
        )
    
    def _create_same_language_result(self, text: str, language: str, start_time: float) -> TranslationResult:
        """Create result for same source/target language"""
        processing_time = time.time() - start_time
        
        return TranslationResult(
            original_text=text[:500],
            translated_text=text,
            source_language=language,
            target_language=language,
            confidence=1.0,  # High confidence - no translation needed
            translation_service='same_language',
            processing_time=processing_time
        )
    
    async def translate_to_multiple_languages(self, text: str, 
                                            target_languages: List[str] = None) -> List[TranslationResult]:
        """
        Translate text to multiple target languages with improved batching
        ✅ FIXED: Better error handling and success validation
        """
        if target_languages is None:
            target_languages = self.TARGET_RESEARCH_LANGUAGES
        
        try:
            logger.info(f"🔄 Starting batch translation to {len(target_languages)} languages")
            
            # Reduce concurrent translations to avoid overwhelming
            max_concurrent = 1  # Even more conservative for HTTP requests
            successful_translations = []
            
            for i in range(0, len(target_languages), max_concurrent):
                batch = target_languages[i:i + max_concurrent]
                batch_num = i // max_concurrent + 1
                
                logger.info(f"🔄 Processing batch {batch_num}: {batch}")
                
                # Create translation tasks for this batch
                translation_tasks = []
                for lang in batch:
                    task = self.translate_text(text, lang)
                    translation_tasks.append(task)
                
                try:
                    # Execute batch with longer timeout for HTTP requests
                    results = await asyncio.wait_for(
                        asyncio.gather(*translation_tasks, return_exceptions=True),
                        timeout=60.0  # Longer timeout for HTTP-based translation
                    )
                    
                    # Process results
                    for idx, result in enumerate(results):
                        if isinstance(result, TranslationResult):
                            # Only add successful translations (not fallbacks)
                            if (result.translation_service in ['google_http', 'same_language'] and 
                                result.confidence > 0.7):
                                successful_translations.append(result)
                                logger.info(f"✅ Batch {batch_num} - {result.target_language}: SUCCESS")
                            else:
                                logger.warning(f"⚠️ Batch {batch_num} - {batch[idx] if idx < len(batch) else 'unknown'}: LOW QUALITY")
                        elif isinstance(result, Exception):
                            logger.warning(f"❌ Batch {batch_num} - Translation failed: {result}")
                        else:
                            logger.warning(f"⚠️ Batch {batch_num} - Unexpected result type: {type(result)}")
                
                except asyncio.TimeoutError:
                    logger.error(f"⏰ Translation batch {batch_num} timed out")
                    continue
                except Exception as e:
                    logger.error(f"❌ Translation batch {batch_num} failed: {e}")
                    continue
                
                # Longer delay between batches for HTTP rate limiting
                if i + max_concurrent < len(target_languages):
                    await asyncio.sleep(2.0)  # 2 second delay
            
            success_count = len(successful_translations)
            total_count = len(target_languages)
            
            logger.info(f"✅ Completed {success_count}/{total_count} translations")
            return successful_translations
            
        except Exception as e:
            logger.error(f"❌ Batch translation failed: {e}")
            return []
    
    async def generate_multilingual_keywords(self, query: str) -> Dict[str, str]:
        """
        Generate multilingual keywords for research query
        ✅ FIXED: Improved error handling and validation
        """
        try:
            logger.info(f"🔤 Generating multilingual keywords for: '{query}'")
            
            # Validate input
            if not query or not query.strip():
                logger.warning("❌ Empty query provided for keyword generation")
                return {'original': query or ''}
            
            # Get translations
            translations = await self.translate_to_multiple_languages(query)
            
            multilingual_keywords = {}
            
            # Add original query
            multilingual_keywords['original'] = query
            
            # Add successful translations
            for translation in translations:
                if not translation or not translation.translated_text:
                    continue
                
                lang_code = translation.target_language
                translated_query = translation.translated_text.strip()
                
                # Validate translation quality
                if (translated_query and 
                    translated_query != query and 
                    translation.confidence > 0.5 and  # Higher confidence threshold
                    len(translated_query) > 0):
                    
                    multilingual_keywords[lang_code] = translated_query
                    logger.debug(f"✅ Added keyword for {lang_code}: {translated_query}")
            
            logger.info(f"✅ Generated keywords in {len(multilingual_keywords)} languages")
            return multilingual_keywords
            
        except Exception as e:
            logger.error(f"❌ Multilingual keyword generation failed: {e}")
            return {'original': query}
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get comprehensive translation service statistics"""
        total_attempts = max(self.translation_stats['total_translations'], 1)
        cache_efficiency = (self.translation_stats['cache_hits'] / total_attempts) * 100
        
        success_rate = 0
        if self.translation_stats['total_translations'] > 0:
            success_rate = (
                self.translation_stats['successful_translations'] / 
                self.translation_stats['total_translations'] * 100
            )
        
        return {
            **self.translation_stats,
            'cache_size': len(self.translation_cache),
            'cache_efficiency_percent': round(cache_efficiency, 2),
            'success_rate_percent': round(success_rate, 2),
            'supported_languages': len(self.TARGET_RESEARCH_LANGUAGES),
            'translation_available': self.translation_available,
            'service_status': 'healthy' if self.translation_available else 'degraded',
            'service_type': 'google_http_improved'
        }
    
    def clear_cache(self):
        """Clear translation cache"""
        cache_size = len(self.translation_cache)
        self.translation_cache.clear()
        logger.info(f"🧹 Translation cache cleared ({cache_size} entries)")
    
    async def detect_and_translate_if_needed(self, text: str, target_lang: str = 'en') -> str:
        """
        Detect language and translate to target if needed
        ✅ FIXED: Improved language detection integration
        """
        try:
            if not text or not text.strip():
                return text
            
            # Try to import language detector
            try:
                from core.language_detector import LanguageDetector
                detector = LanguageDetector()
                detection_result = detector.detect_language(text)
                detected_lang = detection_result.get('language_code', 'unknown')
            except ImportError:
                logger.warning("⚠️ Language detector not available, assuming auto-detection")
                detected_lang = 'auto'
            except Exception as e:
                logger.warning(f"⚠️ Language detection failed: {e}")
                detected_lang = 'auto'
            
            # If already in target language, return original
            if detected_lang == target_lang:
                logger.info(f"🔄 Text already in target language ({target_lang})")
                return text
            
            # Translate to target language
            translation = await self.translate_text(text, target_lang, detected_lang)
            
            if (translation and 
                translation.translation_service in ['google_http', 'same_language'] and 
                translation.confidence > 0.7):
                
                logger.info(f"🔄 Auto-translated {detected_lang} → {target_lang}")
                return translation.translated_text
            else:
                logger.warning(f"⚠️ Auto-translation failed or low quality, using original")
                return text
                
        except Exception as e:
            logger.error(f"❌ Auto-translation failed: {e}")
            return text
    
    def reset_stats(self):
        """Reset translation statistics"""
        self.translation_stats = {
            'total_translations': 0,
            'cache_hits': 0,
            'google_translations': 0,
            'failed_translations': 0,
            'successful_translations': 0,
            'http_translations': 0
        }
        logger.info("📊 Translation statistics reset")
    
    def is_healthy(self) -> bool:
        """Check if translation service is healthy"""
        return self.translation_available and self.google_translator is not None
    
    def __del__(self):
        """Cleanup when service is destroyed"""
        try:
            if hasattr(self, 'google_translator') and self.google_translator:
                self.google_translator.close()
        except Exception as e:
            logger.warning(f"⚠️ Error during translator cleanup: {e}")
