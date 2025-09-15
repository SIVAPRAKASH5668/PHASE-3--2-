from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException
import logging
from typing import Dict, List, Optional
import re

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Advanced multilingual language detection with confidence scoring"""
    
    # Comprehensive language mapping
    LANGUAGE_MAPPING = {
        'en': 'English',
        'zh': 'Chinese',
        'de': 'German',
        'fr': 'French',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'es': 'Spanish',
        'ru': 'Russian',
        'it': 'Italian',
        'pt': 'Portuguese',
        'hi': 'Hindi',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'da': 'Danish',
        'fi': 'Finnish',
        'tr': 'Turkish',
        'pl': 'Polish',
        'cs': 'Czech',
        'hu': 'Hungarian',
        'ro': 'Romanian',
        'bg': 'Bulgarian',
        'hr': 'Croatian',
        'sr': 'Serbian',
        'sk': 'Slovak',
        'sl': 'Slovenian',
        'et': 'Estonian',
        'lv': 'Latvian',
        'lt': 'Lithuanian',
        'uk': 'Ukrainian',
        'be': 'Belarusian',
        'mk': 'Macedonian',
        'sq': 'Albanian',
        'ca': 'Catalan',
        'eu': 'Basque',
        'gl': 'Galician',
        'cy': 'Welsh',
        'ga': 'Irish',
        'mt': 'Maltese',
        'is': 'Icelandic',
        'fo': 'Faroese',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'lo': 'Lao',
        'my': 'Burmese',
        'km': 'Khmer',
        'ne': 'Nepali',
        'bn': 'Bengali',
        'ur': 'Urdu',
        'fa': 'Persian',
        'he': 'Hebrew',
        'id': 'Indonesian',
        'ms': 'Malay',
        'tl': 'Filipino',
        'sw': 'Swahili',
        'am': 'Amharic',
        'yo': 'Yoruba',
        'zu': 'Zulu',
        'af': 'Afrikaans'
    }
    
    # High-priority languages for research
    RESEARCH_PRIORITY_LANGUAGES = [
        'en', 'zh', 'de', 'fr', 'ja', 'ko', 'es', 'ru', 'it', 'pt', 'ar', 'hi'
    ]
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.supported_languages = list(self.LANGUAGE_MAPPING.keys())
        logger.info(f"✅ Language detector initialized with {len(self.supported_languages)} languages")
    
    def detect_language(self, text: str, include_probabilities: bool = False) -> Dict:
        """
        Advanced language detection with confidence scoring
        
        Args:
            text: Input text
            include_probabilities: Include probability scores for multiple languages
            
        Returns:
            dict: Language detection result with confidence
        """
        try:
            if not text or len(text.strip()) < 5:
                return {
                    'language_code': 'en',
                    'language_name': 'English',
                    'confidence': 0.3,
                    'supported': True,
                    'is_research_priority': True,
                    'probabilities': []
                }
            
            # Clean text for better detection
            cleaned_text = self._clean_text_for_detection(text)
            
            if include_probabilities:
                # Get multiple language probabilities
                lang_probs = detect_langs(cleaned_text)
                probabilities = [
                    {
                        'language_code': prob.lang,
                        'language_name': self.LANGUAGE_MAPPING.get(prob.lang, 'Unknown'),
                        'probability': prob.prob
                    }
                    for prob in lang_probs[:5]  # Top 5 languages
                ]
                
                # Get the most likely language
                detected_lang = lang_probs[0].lang
                confidence = lang_probs[0].prob
            else:
                # Simple detection
                detected_lang = detect(cleaned_text)
                confidence = 0.9  # langdetect doesn't provide confidence by default
                probabilities = []
            
            # Get language information
            language_name = self.LANGUAGE_MAPPING.get(detected_lang, 'Unknown')
            is_supported = detected_lang in self.supported_languages
            is_research_priority = detected_lang in self.RESEARCH_PRIORITY_LANGUAGES
            
            result = {
                'language_code': detected_lang,
                'language_name': language_name,
                'confidence': confidence,
                'supported': is_supported,
                'is_research_priority': is_research_priority,
                'probabilities': probabilities
            }
            
            logger.info(f"🌍 Detected language: {language_name} ({detected_lang}) - Confidence: {confidence:.2f}")
            return result
            
        except LangDetectException as e:
            logger.warning(f"⚠️ Language detection failed: {e}")
            return self._create_fallback_result(include_probabilities)
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in language detection: {e}")
            return self._create_fallback_result(include_probabilities)
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text to improve language detection accuracy"""
        try:
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep essential punctuation
            text = re.sub(r'[^\w\s\.\,\!\?\:\;\-]', '', text)
            
            return text.strip()
        except Exception:
            return text
    
    def _create_fallback_result(self, include_probabilities: bool = False) -> Dict:
        """Create fallback result when detection fails"""
        return {
            'language_code': 'en',
            'language_name': 'English',
            'confidence': 0.1,
            'supported': True,
            'is_research_priority': True,
            'probabilities': [] if not include_probabilities else [
                {
                    'language_code': 'en',
                    'language_name': 'English',
                    'probability': 0.1
                }
            ]
        }
    
    def detect_multiple_languages(self, text: str, threshold: float = 0.1) -> List[Dict]:
        """
        Detect multiple languages in text
        
        Args:
            text: Input text
            threshold: Minimum probability threshold
            
        Returns:
            List of detected languages above threshold
        """
        try:
            result = self.detect_language(text, include_probabilities=True)
            
            if result['probabilities']:
                return [
                    lang for lang in result['probabilities']
                    if lang['probability'] >= threshold
                ]
            else:
                return [
                    {
                        'language_code': result['language_code'],
                        'language_name': result['language_name'],
                        'probability': result['confidence']
                    }
                ]
        except Exception as e:
            logger.error(f"❌ Multiple language detection failed: {e}")
            return []
    
    def get_research_priority_languages(self) -> List[Dict]:
        """Get list of high-priority languages for research"""
        return [
            {
                'language_code': lang,
                'language_name': self.LANGUAGE_MAPPING[lang],
                'is_priority': True
            }
            for lang in self.RESEARCH_PRIORITY_LANGUAGES
        ]
    
    def is_research_language(self, language_code: str) -> bool:
        """Check if language is high-priority for research"""
        return language_code in self.RESEARCH_PRIORITY_LANGUAGES
    
    def get_supported_languages(self) -> List[Dict]:
        """Get all supported languages"""
        return [
            {
                'language_code': code,
                'language_name': name,
                'is_research_priority': code in self.RESEARCH_PRIORITY_LANGUAGES
            }
            for code, name in self.LANGUAGE_MAPPING.items()
        ]
