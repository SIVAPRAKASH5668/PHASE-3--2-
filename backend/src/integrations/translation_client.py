import asyncio
import logging
from typing import Dict, Any, List, Optional
import httpx
from googletrans import Translator
import time

from core.translation_service import TranslationService

logger = logging.getLogger(__name__)

class TranslationClient:
    """
    Dedicated translation client for external integrations
    Wrapper around TranslationService for external API access
    """
    
    def __init__(self):
        self.translation_service = TranslationService()
        self.external_apis = {
            'google_translate': True,
            'deepl': False,  # Can be enabled with API key
            'azure': False   # Can be enabled with API key
        }
        
        logger.info("🌐 Translation client initialized")
    
    async def translate_query(self, query: str, target_languages: List[str] = None) -> Dict[str, str]:
        """
        Translate query to multiple languages for research
        
        Args:
            query: Original query
            target_languages: List of target language codes
            
        Returns:
            Dictionary mapping language codes to translated queries
        """
        try:
            if not target_languages:
                target_languages = self.translation_service.TARGET_RESEARCH_LANGUAGES
            
            multilingual_keywords = await self.translation_service.generate_multilingual_keywords(query)
            
            # Filter to requested languages
            filtered_keywords = {
                lang: translation for lang, translation in multilingual_keywords.items()
                if lang in target_languages or lang == 'original'
            }
            
            return filtered_keywords
            
        except Exception as e:
            logger.error(f"❌ Query translation failed: {e}")
            return {'original': query}
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of input text
        
        Args:
            text: Input text
            
        Returns:
            Language detection result
        """
        try:
            from core.language_detector import LanguageDetector
            detector = LanguageDetector()
            return detector.detect_language(text, include_probabilities=True)
            
        except Exception as e:
            logger.error(f"❌ Language detection failed: {e}")
            return {
                'language_code': 'en',
                'language_name': 'English',
                'confidence': 0.5,
                'supported': True
            }
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Get list of supported translation languages"""
        try:
            from core.language_detector import LanguageDetector
            detector = LanguageDetector()
            return detector.get_supported_languages()
            
        except Exception as e:
            logger.error(f"❌ Failed to get supported languages: {e}")
            return []
