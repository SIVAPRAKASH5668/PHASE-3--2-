
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import time
from datetime import datetime

# ✅ ADD THIS MISSING IMPORT:
from concurrent.futures import ThreadPoolExecutor


# Enhanced imports for Phase 2
from core.llm_processor import EnhancedMultiAPILLMProcessor, PaperContext
from core.translation_service import TranslationService
from core.language_detector import LanguageDetector
from core.embedding_generator import EmbeddingGenerator
from utils.text_processing import TextProcessor
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Enhanced analysis result with comprehensive metadata"""
    paper_id: Optional[int]
    title: str
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
    analysis_confidence: float
    processing_time: float
    language_detected: str
    ai_agent_used: str
    analysis_method: str
    error_message: Optional[str] = None
    
    # Phase 2: Enhanced metadata
    embedding_generated: bool = False
    embedding_dimension: Optional[int] = None
    translation_data: Optional[Dict[str, Any]] = None
    multilingual_keywords: Optional[List[str]] = None
    cross_linguistic_analysis: Optional[Dict[str, Any]] = None

@dataclass
class BatchAnalysisResult:
    """Results from batch paper analysis"""
    successful_analyses: List[AnalysisResult]
    failed_analyses: List[Dict[str, Any]]
    total_processed: int
    success_rate: float
    average_processing_time: float
    total_processing_time: float
    batch_statistics: Dict[str, Any]

class PaperAnalyzerAgent:
    """
    🚀 **Enhanced Paper Analyzer Agent v2.0**
    
    **New Features:**
    - 🌍 Multilingual paper analysis
    - 🧠 AI-powered context extraction
    - ⚡ Parallel batch processing 
    - 🎯 Quality-aware analysis
    - 📊 Advanced performance metrics
    - 🔗 Embedding generation
    - 🌐 Cross-linguistic analysis
    """
    
    def __init__(self):
        # Core components
        self.llm_processor = EnhancedMultiAPILLMProcessor()
        
        # Phase 2: Enhanced components
        try:
            self.translation_service = TranslationService()
            self.language_detector = LanguageDetector()
            self.embedding_generator = EmbeddingGenerator()
            self.text_processor = TextProcessor()
            self.multilingual_support = True
            logger.info("🌍 Enhanced Paper Analyzer with multilingual support initialized")
        except Exception as e:
            logger.warning(f"⚠️ Multilingual components failed: {e}")
            self.translation_service = None
            self.language_detector = None
            self.embedding_generator = None
            self.text_processor = None
            self.multilingual_support = False
        
        # Performance tracking
        self.analysis_stats = {
            "total_papers_analyzed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
            "multilingual_analyses": 0,
            "embeddings_generated": 0,
            "last_reset": time.time()
        }
        
        # Configuration
        self.max_concurrent_analyses = settings.PAPER_ANALYSIS_MAX_CONCURRENT
        self.analysis_timeout = settings.AGENT_TASK_TIMEOUT
        self.enable_embeddings = settings.ENABLE_VECTOR_SEARCH
        self.quality_threshold = settings.GRAPH_QUALITY_THRESHOLD
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
    
        logger.info(f"📊 Paper Analyzer Agent initialized (max concurrent: {self.max_concurrent_analyses})")

    async def initialize(self):
        """Initialize the Paper Analyzer Agent and all components"""
        try:
            logger.info("🚀 Initializing Paper Analyzer Agent...")
            
            # Test LLM processor initialization
            if self.llm_processor:
                try:
                    health_check = await self.llm_processor.enhanced_health_check()
                    if health_check.get('status') not in ['healthy', 'excellent']:
                        logger.warning("⚠️ LLM processor health check failed")
                except Exception as e:
                    logger.warning(f"⚠️ LLM processor health check error: {e}")
            
            # Test multilingual components if available
            if self.multilingual_support:
                try:
                    # Test language detector
                    if self.language_detector:
                        test_result = self.language_detector.detect_language("Cette recherche explore les techniques avancées")
                        logger.info("✅ Language detector test passed")
                    
                    # Test translation service
                    if self.translation_service:
                        # Simple test without actual API call
                        logger.info("✅ Translation service ready")
                    
                    # ✅ FIXED: Test embedding generator
                    if self.embedding_generator and self.enable_embeddings:
                        try:
                            # Test with simple text
                            test_embedding = await self.embedding_generator.generate_embedding_async("test")
                            
                            # ✅ CORRECT: Check if embedding is valid
                            if test_embedding is not None and hasattr(test_embedding, 'shape') and test_embedding.size > 0:
                                logger.info("✅ Embedding generator test passed")
                            elif test_embedding is not None and isinstance(test_embedding, (list, tuple)) and len(test_embedding) > 0:
                                logger.info("✅ Embedding generator test passed")
                            else:
                                logger.warning("⚠️ Embedding generator returned invalid result")
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Embedding generator test failed: {e}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Multilingual component test failed: {e}")
            
            logger.info("✅ Paper Analyzer Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Paper Analyzer Agent initialization failed: {e}")
            raise



    
    async def analyze_paper(self, paper: Dict[str, Any], 
                          analysis_type: str = "standard",
                          enable_multilingual: bool = True,
                          generate_embedding: bool = True) -> AnalysisResult:
        """
        🌟 Enhanced single paper analysis
        
        Args:
            paper: Paper data dictionary
            analysis_type: "quick", "standard", "detailed"
            enable_multilingual: Enable multilingual processing
            generate_embedding: Generate embeddings for vector search
            
        Returns:
            Enhanced analysis result
        """
        start_time = time.time()
        
        try:
            # Extract basic information
            title = paper.get('title', '').strip()
            abstract = paper.get('abstract', '').strip()
            
            if not title:
                return self._create_error_result(paper, "Missing paper title", start_time)
            
            # Phase 2: Enhanced language detection
            detected_language = 'en'
            translation_data = None
            
            if self.multilingual_support and enable_multilingual:
                # Detect language of paper content
                text_for_detection = f"{title} {abstract}"[:1000]  # Limit for detection
                
                if self.language_detector:
                    detection_result = self.language_detector.detect_language(text_for_detection)
                    detected_language = detection_result.get('language_code', 'en')
                    
                    # Translate if non-English
                    if detected_language != 'en' and self.translation_service:
                        try:
                            translation_result = await self.translation_service.translate_text(
                                abstract, 'en'
                            )
                            if translation_result.translated_text:
                                translation_data = {
                                    'original_language': detected_language,
                                    'translated_abstract': translation_result.translated_text,
                                    'translation_confidence': translation_result.confidence
                                }
                                # Use translated text for analysis
                                abstract = translation_result.translated_text
                                
                                self.analysis_stats['multilingual_analyses'] += 1
                        except Exception as e:
                            logger.warning(f"Translation failed: {e}")
            
            # Perform AI-powered analysis
            paper_context = await self._perform_ai_analysis(
                title, abstract, paper, analysis_type
            )
            
            # Phase 2: Generate embeddings if requested
            embedding_generated = False
            embedding_dimension = None
            
            if self.enable_embeddings and generate_embedding and self.embedding_generator:
                try:
                    text_for_embedding = f"{title} {paper_context.context_summary}"
                    embedding = await self.embedding_generator.generate_embedding_async(text_for_embedding)
                    
                    if embedding:
                        # Store embedding in paper context (would be saved to database)
                        embedding_generated = True
                        embedding_dimension = len(embedding)
                        self.analysis_stats['embeddings_generated'] += 1
                        
                except Exception as e:
                    logger.warning(f"Embedding generation failed: {e}")
            
            # Generate multilingual keywords
            multilingual_keywords = None
            if self.multilingual_support and self.text_processor:
                try:
                    keywords = self.text_processor.extract_keywords(
                        f"{title} {abstract}", language=detected_language
                    )
                    multilingual_keywords = keywords[:10]  # Top 10 keywords
                except Exception as e:
                    logger.warning(f"Keyword extraction failed: {e}")
            
            # Cross-linguistic analysis
            cross_linguistic_analysis = None
            if detected_language != 'en' and translation_data:
                cross_linguistic_analysis = {
                    'original_language': detected_language,
                    'has_translation': True,
                    'translation_quality': translation_data.get('translation_confidence', 0.8),
                    'language_family': self._get_language_family(detected_language)
                }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create enhanced result
            result = AnalysisResult(
                paper_id=paper.get('id'),
                title=title,
                context_summary=paper_context.context_summary,
                research_domain=paper_context.research_domain,
                methodology=paper_context.methodology,
                key_findings=paper_context.key_findings,
                innovations=paper_context.innovations,
                limitations=paper_context.limitations,
                research_questions=paper_context.research_questions,
                contributions=paper_context.contributions,
                future_work=paper_context.future_work,
                related_concepts=paper_context.related_concepts,
                context_quality_score=paper_context.context_quality_score,
                analysis_confidence=paper_context.analysis_confidence,
                processing_time=processing_time,
                language_detected=detected_language,
                ai_agent_used=paper_context.model_used,
                analysis_method=analysis_type,
                
                # Phase 2 enhancements
                embedding_generated=embedding_generated,
                embedding_dimension=embedding_dimension,
                translation_data=translation_data,
                multilingual_keywords=multilingual_keywords,
                cross_linguistic_analysis=cross_linguistic_analysis
            )
            
            # Update statistics
            self._update_analysis_stats(processing_time, True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_analysis_stats(processing_time, False)
            logger.error(f"❌ Paper analysis failed: {e}")
            return self._create_error_result(paper, str(e), start_time)
    
    async def batch_analyze_papers(self, papers: List[Dict[str, Any]],
                                 analysis_type: str = "standard",
                                 enable_multilingual: bool = True) -> BatchAnalysisResult:
        """
        🚀 Enhanced batch paper analysis with parallel processing
        
        Args:
            papers: List of paper dictionaries
            analysis_type: Analysis depth
            enable_multilingual: Enable multilingual processing
            
        Returns:
            Batch analysis results
        """
        start_time = time.time()
        
        if not papers:
            return BatchAnalysisResult(
                successful_analyses=[],
                failed_analyses=[],
                total_processed=0,
                success_rate=0.0,
                average_processing_time=0.0,
                total_processing_time=0.0,
                batch_statistics={}
            )
        
        logger.info(f"📊 Starting batch analysis of {len(papers)} papers")
        
        # Process papers in parallel batches
        successful_analyses = []
        failed_analyses = []
        
        # Create semaphore for concurrent control
        semaphore = asyncio.Semaphore(self.max_concurrent_analyses)
        
        async def analyze_single_with_semaphore(paper, index):
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        self.analyze_paper(
                            paper, 
                            analysis_type=analysis_type,
                            enable_multilingual=enable_multilingual
                        ),
                        timeout=self.analysis_timeout
                    )
                    return ('success', result, index)
                except asyncio.TimeoutError:
                    error_info = {
                        'paper_index': index,
                        'paper_title': paper.get('title', 'Unknown'),
                        'error': 'Analysis timeout',
                        'timeout_seconds': self.analysis_timeout
                    }
                    return ('failure', error_info, index)
                except Exception as e:
                    error_info = {
                        'paper_index': index,
                        'paper_title': paper.get('title', 'Unknown'),
                        'error': str(e)
                    }
                    return ('failure', error_info, index)
        
        # Create all tasks
        tasks = [
            analyze_single_with_semaphore(paper, i) 
            for i, paper in enumerate(papers)
        ]
        
        # Execute all tasks and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                failed_analyses.append({
                    'error': str(result),
                    'paper_title': 'Unknown'
                })
            else:
                status, data, index = result
                if status == 'success':
                    successful_analyses.append(data)
                else:
                    failed_analyses.append(data)
        
        # Calculate statistics
        total_time = time.time() - start_time
        total_processed = len(papers)
        success_rate = len(successful_analyses) / total_processed if total_processed > 0 else 0.0
        average_time = total_time / total_processed if total_processed > 0 else 0.0
        
        # Enhanced batch statistics
        batch_statistics = {
            "analysis_type": analysis_type,
            "multilingual_enabled": enable_multilingual,
            "concurrent_analyses": self.max_concurrent_analyses,
            "languages_detected": len(set(
                result.language_detected for result in successful_analyses
            )),
            "embeddings_generated": sum(
                1 for result in successful_analyses if result.embedding_generated
            ),
            "multilingual_papers": sum(
                1 for result in successful_analyses 
                if result.language_detected != 'en'
            ),
            "quality_distribution": self._calculate_quality_distribution(successful_analyses)
        }
        
        result = BatchAnalysisResult(
            successful_analyses=successful_analyses,
            failed_analyses=failed_analyses,
            total_processed=total_processed,
            success_rate=success_rate,
            average_processing_time=average_time,
            total_processing_time=total_time,
            batch_statistics=batch_statistics
        )
        
        logger.info(f"✅ Batch analysis completed: {len(successful_analyses)}/{total_processed} successful "
                   f"({success_rate:.1%} success rate, {total_time:.2f}s)")
        
        return result
    
    async def _perform_ai_analysis(self, title: str, abstract: str, 
                                 paper: Dict[str, Any], analysis_type: str) -> PaperContext:
        """Perform AI-powered paper analysis"""
        try:
            # Prepare paper data for LLM processing
            paper_data = {
                'title': title,
                'abstract': abstract,
                **paper
            }
            
            # Use enhanced LLM processor
            if analysis_type == "detailed":
                context = await self.llm_processor.analyze_paper_context_detailed(paper_data)
            elif analysis_type == "quick":
                context = await self.llm_processor.analyze_paper_context_quick(paper_data)
            else:
                context = await self.llm_processor.analyze_paper_context(paper_data)
            
            return context
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            # Return fallback context
            return PaperContext(
                context_summary=f"Analysis failed: {str(e)}",
                research_domain="General Research",
                methodology="Unknown",
                key_findings=["Analysis could not be completed"],
                innovations=[],
                limitations=["Analysis limitations unknown"],
                research_questions=[],
                contributions=[],
                future_work=[],
                related_concepts=[],
                context_quality_score=0.2,
                analysis_confidence=0.1,
                model_used="fallback"
            )
    
    def _create_error_result(self, paper: Dict[str, Any], error: str, start_time: float) -> AnalysisResult:
        """Create error result for failed analysis"""
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            paper_id=paper.get('id'),
            title=paper.get('title', 'Unknown'),
            context_summary=f"Analysis failed: {error}",
            research_domain="Unknown",
            methodology="Unknown",
            key_findings=[],
            innovations=[],
            limitations=[],
            research_questions=[],
            contributions=[],
            future_work=[],
            related_concepts=[],
            context_quality_score=0.0,
            analysis_confidence=0.0,
            processing_time=processing_time,
            language_detected="unknown",
            ai_agent_used="none",
            analysis_method="failed",
            error_message=error
        )
    
    def _update_analysis_stats(self, processing_time: float, success: bool):
        """Update analysis performance statistics"""
        self.analysis_stats['total_papers_analyzed'] += 1
        
        if success:
            self.analysis_stats['successful_analyses'] += 1
            
            # Update average processing time
            successful = self.analysis_stats['successful_analyses']
            current_avg = self.analysis_stats['average_processing_time']
            self.analysis_stats['average_processing_time'] = (
                (current_avg * (successful - 1) + processing_time) / successful
            )
        else:
            self.analysis_stats['failed_analyses'] += 1
        
        self.analysis_stats['total_processing_time'] += processing_time
    
    def _calculate_quality_distribution(self, analyses: List[AnalysisResult]) -> Dict[str, Any]:
        """Calculate quality distribution of analyses"""
        if not analyses:
            return {}
        
        quality_scores = [result.context_quality_score for result in analyses]
        
        return {
            "high_quality_count": len([s for s in quality_scores if s >= 0.8]),
            "medium_quality_count": len([s for s in quality_scores if 0.5 <= s < 0.8]),
            "low_quality_count": len([s for s in quality_scores if s < 0.5]),
            "average_quality": sum(quality_scores) / len(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores)
        }
    
    def _get_language_family(self, language_code: str) -> str:
        """Get language family for linguistic analysis"""
        language_families = {
            'en': 'Germanic',
            'de': 'Germanic',
            'nl': 'Germanic',
            'sv': 'Germanic',
            'no': 'Germanic',
            'da': 'Germanic',
            'fr': 'Romance',
            'es': 'Romance',
            'it': 'Romance',
            'pt': 'Romance',
            'zh': 'Sino-Tibetan',
            'ja': 'Japonic',
            'ko': 'Koreanic',
            'ar': 'Semitic',
            'hi': 'Indo-European',
            'ru': 'Slavic',
            'pl': 'Slavic',
            'cs': 'Slavic',
            'tr': 'Turkic',
            'fi': 'Uralic',
            'hu': 'Uralic'
        }
        
        return language_families.get(language_code, 'Unknown')
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics"""
        total_analyses = self.analysis_stats['total_papers_analyzed']
        success_rate = (
            self.analysis_stats['successful_analyses'] / total_analyses 
            if total_analyses > 0 else 0.0
        )
        
        return {
            **self.analysis_stats,
            'success_rate': success_rate,
            'multilingual_support': self.multilingual_support,
            'embedding_generation_enabled': self.enable_embeddings,
            'max_concurrent_analyses': self.max_concurrent_analyses,
            'uptime_hours': (time.time() - self.analysis_stats['last_reset']) / 3600
        }
    
    def reset_statistics(self):
        """Reset analysis statistics"""
        self.analysis_stats = {
            "total_papers_analyzed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
            "multilingual_analyses": 0,
            "embeddings_generated": 0,
            "last_reset": time.time()
        }
        logger.info("📊 Paper Analyzer Agent statistics reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Test LLM processor
            llm_health = await self.llm_processor.enhanced_health_check()
            
            health_status = {
                "status": "healthy",
                "agent": "PaperAnalyzerAgent",
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "llm_processor": llm_health.get('status', 'unknown'),
                    "multilingual_support": self.multilingual_support,
                    "embedding_generation": self.enable_embeddings,
                    "translation_service": self.translation_service is not None,
                    "language_detector": self.language_detector is not None
                },
                "configuration": {
                    "max_concurrent_analyses": self.max_concurrent_analyses,
                    "analysis_timeout": self.analysis_timeout,
                    "quality_threshold": self.quality_threshold
                },
                "statistics": self.get_analysis_statistics()
            }
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "agent": "PaperAnalyzerAgent",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            logger.info("🔒 Shutting down Paper Analyzer Agent...")
            
            # Shutdown LLM processor
            if self.llm_processor:
                await self.llm_processor.shutdown()
            
            # ✅ ADD: Shutdown thread pool
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # Clear caches
            if self.embedding_generator:
                # Clear any cached embeddings
                pass
            
            logger.info("🔒 Paper Analyzer Agent shut down gracefully")
            
        except Exception as e:
            logger.error(f"Paper Analyzer Agent shutdown error: {e}")

# Backward compatibility alias
PaperAnalysisAgent = PaperAnalyzerAgent
