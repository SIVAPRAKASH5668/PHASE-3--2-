import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.translation_service import TranslationService
from core.language_detector import LanguageDetector

logger = logging.getLogger(__name__)

router = APIRouter()

# Request Models
class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    target_language: str = Field(..., min_length=2, max_length=10)
    source_language: Optional[str] = Field("auto")
    domain: Optional[str] = Field("academic")

class LanguageDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class BulkTranslationRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    target_language: str = Field(..., min_length=2, max_length=10)
    source_language: Optional[str] = Field("auto")
    domain: Optional[str] = Field("academic")

@router.post("/translate")
async def translate_text(request: TranslationRequest):
    """
    Translate text with academic domain optimization
    """
    try:
        start_time = time.time()
        
        # Initialize translation service
        translation_service = TranslationService()
        
        # Perform translation
        result = await translation_service.translate_text(
            text=request.text,
            target_language=request.target_language,
            source_language=request.source_language,
            domain=request.domain
        )
        
        return {
            "success": True,
            "translation": {
                "original_text": request.text,
                "translated_text": result.get("translated_text", ""),
                "source_language": result.get("detected_language", request.source_language),
                "target_language": request.target_language,
                "confidence_score": result.get("confidence", 0.8),
                "domain": request.domain
            },
            "metadata": {
                "processing_time": time.time() - start_time,
                "service_used": result.get("service", "unknown"),
                "quality_score": result.get("quality_score", 0.8)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Translation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "translation": None
        }

@router.post("/detect")
async def detect_language(request: LanguageDetectionRequest):
    """
    Detect language of input text
    """
    try:
        start_time = time.time()
        
        # Initialize language detector
        language_detector = LanguageDetector()
        
        # Detect language
        result = language_detector.detect_language(request.text)
        
        return {
            "success": True,
            "detection": {
                "text": request.text,
                "language_code": result.get("language_code", "unknown"),
                "language_name": result.get("language_name", "Unknown"),
                "confidence": result.get("confidence", 0.0),
                "is_academic": result.get("is_academic", False),
                "domain_indicators": result.get("domain_indicators", [])
            },
            "metadata": {
                "processing_time": time.time() - start_time,
                "detection_method": result.get("method", "statistical")
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Language detection failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "detection": None
        }

@router.post("/bulk")
async def bulk_translate(request: BulkTranslationRequest):
    """
    Bulk translate multiple texts efficiently
    """
    try:
        start_time = time.time()
        
        # Initialize translation service
        translation_service = TranslationService()
        
        # Perform bulk translation
        results = await translation_service.bulk_translate(
            texts=request.texts,
            target_language=request.target_language,
            source_language=request.source_language,
            domain=request.domain
        )
        
        return {
            "success": True,
            "translations": [
                {
                    "original_text": text,
                    "translated_text": result.get("translated_text", ""),
                    "source_language": result.get("detected_language", request.source_language),
                    "confidence_score": result.get("confidence", 0.8)
                }
                for text, result in zip(request.texts, results)
            ],
            "metadata": {
                "processing_time": time.time() - start_time,
                "texts_processed": len(request.texts),
                "success_rate": len([r for r in results if r.get("success", False)]) / len(results),
                "average_confidence": sum(r.get("confidence", 0) for r in results) / len(results)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Bulk translation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "translations": []
        }

@router.get("/languages")
async def get_supported_languages():
    """
    Get supported languages for translation
    """
    return {
        "success": True,
        "languages": {
            "en": {"name": "English", "native_name": "English", "academic_support": True},
            "es": {"name": "Spanish", "native_name": "Español", "academic_support": True},
            "fr": {"name": "French", "native_name": "Français", "academic_support": True},
            "de": {"name": "German", "native_name": "Deutsch", "academic_support": True},
            "it": {"name": "Italian", "native_name": "Italiano", "academic_support": True},
            "pt": {"name": "Portuguese", "native_name": "Português", "academic_support": True},
            "ru": {"name": "Russian", "native_name": "Русский", "academic_support": True},
            "zh": {"name": "Chinese", "native_name": "中文", "academic_support": True},
            "ja": {"name": "Japanese", "native_name": "日本語", "academic_support": True},
            "ko": {"name": "Korean", "native_name": "한국어", "academic_support": True},
            "ar": {"name": "Arabic", "native_name": "العربية", "academic_support": True},
            "hi": {"name": "Hindi", "native_name": "हिन्दी", "academic_support": True},
            "nl": {"name": "Dutch", "native_name": "Nederlands", "academic_support": True},
            "sv": {"name": "Swedish", "native_name": "Svenska", "academic_support": True},
            "da": {"name": "Danish", "native_name": "Dansk", "academic_support": True},
            "no": {"name": "Norwegian", "native_name": "Norsk", "academic_support": True},
            "fi": {"name": "Finnish", "native_name": "Suomi", "academic_support": True},
            "pl": {"name": "Polish", "native_name": "Polski", "academic_support": True},
            "tr": {"name": "Turkish", "native_name": "Türkçe", "academic_support": True},
            "th": {"name": "Thai", "native_name": "ไทย", "academic_support": True}
        },
        "total_supported": 20,
        "detection_supported": True,
        "academic_optimization": True
    }

@router.get("/domains")
async def get_translation_domains():
    """
    Get supported translation domains
    """
    return {
        "success": True,
        "domains": {
            "academic": {
                "name": "Academic",
                "description": "Scientific and research content",
                "optimization": "Technical terminology preservation"
            },
            "medical": {
                "name": "Medical", 
                "description": "Medical and healthcare content",
                "optimization": "Medical terminology accuracy"
            },
            "technical": {
                "name": "Technical",
                "description": "Technical and engineering content", 
                "optimization": "Technical term consistency"
            },
            "general": {
                "name": "General",
                "description": "General purpose translation",
                "optimization": "Natural language flow"
            }
        }
    }

@router.get("/health")
async def translation_health_check():
    """
    Health check for translation service
    """
    try:
        # Test translation service
        translation_service = TranslationService()
        language_detector = LanguageDetector()
        
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "components": {
                "translation_service": "healthy",
                "language_detector": "healthy", 
                "multilingual_support": "active"
            },
            "capabilities": [
                "text_translation",
                "language_detection",
                "bulk_processing",
                "domain_optimization"
            ],
            "supported_languages": 20
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
