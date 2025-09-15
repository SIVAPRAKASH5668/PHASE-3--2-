import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from workflow.langgraph_workflow import ContextAwareResearchWorkflow
from core.translation_service import TranslationService
from core.language_detector import LanguageDetector
from core.embedding_generator import EmbeddingGenerator
from core.multi_ai_agent import MultiAIAgentOrchestrator, AgentTask, AgentType
from database.vector_operations import VectorOperations
from database.master_client import MasterSearchClient
    
logger = logging.getLogger(__name__)

@dataclass
class MultilingualWorkflowState:
    """Enhanced workflow state with multilingual capabilities"""
    original_query: str
    detected_language: Dict[str, Any]
    multilingual_keywords: Dict[str, str]
    papers_found: List[Dict[str, Any]]
    papers_analyzed: List[Dict[str, Any]]
    relationships_found: List[Dict[str, Any]]
    graph_data: Dict[str, Any]
    embeddings_generated: Dict[str, Any]
    vector_search_results: List[Dict[str, Any]]
    status: str
    processing_stats: Dict[str, Any]
    multilingual_stats: Dict[str, Any]

class MultilingualResearchWorkflow(ContextAwareResearchWorkflow):
    """
    Enhanced research workflow with multilingual capabilities and multi-AI agents
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize multilingual components with error handling
        try:
            self.translation_service = TranslationService()
            self.language_detector = LanguageDetector()
            self.embedding_generator = EmbeddingGenerator()
            self.multi_ai_orchestrator = MultiAIAgentOrchestrator()
            self.master_search_client = MasterSearchClient()
            
            # ✅ FIXED: Safe vector operations initialization
            try:
                self.vector_ops = VectorOperations(self.db_client)
            except Exception as e:
                logger.warning(f"⚠️ Vector operations initialization failed: {e}")
                self.vector_ops = None
            
            logger.info("🌍 Multilingual Research Workflow initialized")
            
        except Exception as e:
            logger.error(f"❌ Multilingual workflow initialization failed: {e}")
            raise
    
    async def initialize(self):
        """Initialize all workflow components"""
        try:
            await super().initialize()
            
            # ✅ FIXED: Safe multi-AI orchestrator initialization
            if hasattr(self.multi_ai_orchestrator, 'initialize_agents'):
                await self.multi_ai_orchestrator.initialize_agents()
            else:
                logger.warning("⚠️ Multi-AI orchestrator doesn't have initialize_agents method")
            
            logger.info("✅ Multilingual workflow fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Multilingual workflow initialization failed: {e}")
            raise
    
    async def process_multilingual_research_query(self, query: str, 
                                                max_papers: int = 30) -> MultilingualWorkflowState:
        """
        Main multilingual workflow with AI agents and vector search
        
        Args:
            query: Research query in any language
            max_papers: Maximum papers to process
            
        Returns:
            MultilingualWorkflowState with complete results
        """
        start_time = datetime.now()
        
        # Initialize enhanced state
        state = MultilingualWorkflowState(
            original_query=query,
            detected_language={},
            multilingual_keywords={},
            papers_found=[],
            papers_analyzed=[],
            relationships_found=[],
            graph_data={},
            embeddings_generated={},
            vector_search_results=[],
            status="starting",
            processing_stats={},
            multilingual_stats={}
        )
        
        try:
            # Step 1: Language Detection and Translation
            logger.info(f"🔍 Step 1: Processing multilingual query: '{query}'")
            state.status = "language_processing"
            
            # Detect original language
            state.detected_language = self.language_detector.detect_language(query, include_probabilities=True)
            logger.info(f"🌍 Detected language: {state.detected_language['language_name']}")
            
            # Generate multilingual keywords
            state.multilingual_keywords = await self.translation_service.generate_multilingual_keywords(query)
            logger.info(f"🔤 Generated keywords in {len(state.multilingual_keywords)} languages")
            
            # Step 2: Enhanced Multilingual Paper Search
            logger.info("📚 Step 2: Multilingual paper search")
            state.status = "searching_papers"
            
            state.papers_found = await self._multilingual_paper_search(
                state.multilingual_keywords, max_papers
            )
            
            if not state.papers_found:
                state.status = "completed"
                state.multilingual_stats = {"error": "No papers found"}
                return state
            
            logger.info(f"📄 Found {len(state.papers_found)} papers across languages")
            
            # Step 3: Generate Embeddings for Vector Search
            logger.info("🧮 Step 3: Generating embeddings")
            state.status = "generating_embeddings"
            
            state.embeddings_generated = await self._generate_query_embeddings(
                state.multilingual_keywords
            )
            
            # Step 4: Vector Similarity Search
            logger.info("🔍 Step 4: Vector similarity search")
            state.status = "vector_search"
            
            state.vector_search_results = await self._perform_vector_search(
                state.embeddings_generated, max_papers//2
            )
            
            # Combine traditional and vector search results
            combined_papers = self._combine_search_results(
                state.papers_found, state.vector_search_results
            )
            
            # Step 5: Multi-AI Agent Analysis
            logger.info(f"🤖 Step 5: Multi-AI agent analysis of {len(combined_papers)} papers")
            state.status = "ai_agent_analysis"
            
            state.papers_analyzed = await self._multi_ai_paper_analysis(combined_papers)
            
            # Step 6: Enhanced Relationship Discovery
            logger.info("🔗 Step 6: AI-powered relationship discovery")
            state.status = "relationship_discovery"
            
            state.relationships_found = await self._multi_ai_relationship_analysis(
                state.papers_analyzed
            )
            
            # Step 7: Build Enhanced Graph
            logger.info("📊 Step 7: Building multilingual graph")
            state.status = "building_graph"
            
            state.graph_data = await self._build_multilingual_graph(
                state.papers_analyzed,
                state.relationships_found,
                state.multilingual_keywords
            )
            
            # Step 8: Calculate Statistics
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            state.processing_stats = {
                "total_processing_time": total_duration,
                "papers_found": len(state.papers_found),
                "papers_analyzed": len(state.papers_analyzed),
                "relationships_found": len(state.relationships_found),
                "vector_search_results": len(state.vector_search_results),
                "languages_processed": len(state.multilingual_keywords),
                "average_quality": self._calculate_average_quality(state.papers_analyzed)
            }
            
            state.multilingual_stats = {
                "original_language": state.detected_language,
                "keywords_generated": len(state.multilingual_keywords),
                "embedding_dimensions": getattr(self.embedding_generator, 'embedding_dimension', 384),
                "ai_agents_used": len(getattr(self.multi_ai_orchestrator, 'active_agents', {})),
                "translation_stats": self.translation_service.get_translation_stats(),
                "vector_search_coverage": len(state.vector_search_results)
            }
            
            state.status = "completed"
            logger.info(f"✅ Multilingual workflow completed in {total_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Multilingual workflow failed: {e}")
            state.status = "error"
            state.multilingual_stats = {"error": str(e)}
        
        return state
    
    async def _multilingual_paper_search(self, multilingual_keywords: Dict[str, str], max_papers: int) -> List[Dict[str, Any]]:
        all_papers = []
        
        for lang_code, query in multilingual_keywords.items():
            # ✅ PARALLEL SEARCH: ArXiv + Felo AI + Others
            search_tasks = [
                self.arxiv_client.search_papers_async(query, max_papers // len(multilingual_keywords) // 3),
                self.felo_ai_client.search_papers_async(query, max_papers // len(multilingual_keywords) // 3, language=lang_code),
                self.semantic_scholar_client.search_papers_async(query, max_papers // len(multilingual_keywords) // 3)
            ]
            
            # Wait for all sources
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine results from all sources
            for result in search_results:
                if isinstance(result, list):
                    all_papers.extend(result)
            
            logger.info(f"🔍 Found {len(all_papers)} papers for {lang_code} from multiple sources")
        
        # Enhanced deduplication across sources
        unique_papers = self._cross_source_deduplication(all_papers)
        return unique_papers[:max_papers]

    async def _search_papers_safe(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """✅ FIXED: Safe wrapper for paper search with proper async handling"""
        try:
            # ✅ FIXED: Use correct parent method with await
            if hasattr(super(), 'search_papers'):
                return await super().search_papers(query, limit)
            elif hasattr(self, '_search_papers'):
                return await self._search_papers(query, limit)
            else:
                # Fallback implementation
                return await self._fallback_paper_search(query, limit)
        except Exception as e:
            logger.error(f"❌ Safe paper search failed: {e}")
            return []

    async def _fallback_paper_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """✅ FIXED: Fallback paper search with proper async handling"""
        try:
            # ✅ FIXED: Use await instead of asyncio.run()
            if hasattr(self, 'arxiv_client') and self.arxiv_client:
                # Properly await the async method
                papers = await self.arxiv_client.search_papers_async(query, limit)
                return papers if papers else []
            
            logger.warning("⚠️ No paper search method available")
            return []
            
        except Exception as e:
            logger.error(f"❌ Fallback paper search failed: {e}")
            return []

    def _remove_duplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity"""
        if not papers:
            return papers
        
        try:
            seen_titles = set()
            unique_papers = []
            
            for paper in papers:
                title = paper.get('title', '').strip().lower()
                if not title:
                    continue
                    
                # Simple deduplication by title
                title_key = ''.join(title.split())[:100]  # Normalized title
                
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_papers.append(paper)
            
            removed_count = len(papers) - len(unique_papers)
            if removed_count > 0:
                logger.info(f"🗑️ Removed {removed_count} duplicate papers")
            
            return unique_papers
            
        except Exception as e:
            logger.error(f"❌ Deduplication failed: {e}")
            return papers
    
    async def _generate_query_embeddings(self, multilingual_keywords: Dict[str, str]) -> Dict[str, Any]:
        """Generate embeddings for multilingual queries"""
        try:
            embedding_tasks = {}
            
            # Create embedding tasks for each language
            for lang_code, query in multilingual_keywords.items():
                if lang_code != 'original' and query:
                    task = self.embedding_generator.generate_embedding_async(query)
                    embedding_tasks[lang_code] = task
            
            # Execute embedding generation in parallel
            if not embedding_tasks:
                return {}
            
            results = await asyncio.gather(
                *embedding_tasks.values(), 
                return_exceptions=True
            )
            
            # Process results
            embeddings = {}
            for lang_code, result in zip(embedding_tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Embedding failed for {lang_code}: {result}")
                    continue
                
                if result is not None:  # ✅ FIXED: Check for valid embedding
                    embeddings[lang_code] = result
            
            logger.info(f"🧮 Generated embeddings for {len(embeddings)} languages")
            return {
                'embeddings': embeddings,
                'dimension': getattr(self.embedding_generator, 'embedding_dimension', 384),
                'model': getattr(self.embedding_generator, 'model_name', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"❌ Query embedding generation failed: {e}")
            return {}
    
    async def _perform_vector_search(self, embeddings_data: Dict[str, Any], 
                                   limit: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search using multilingual embeddings"""
        try:
            if not embeddings_data.get('embeddings') or not self.vector_ops:
                logger.warning("⚠️ Vector search unavailable - no embeddings or vector operations")
                return []
            
            query_embeddings = embeddings_data['embeddings']
            
            # ✅ FIXED: Check if vector ops method exists
            if hasattr(self.vector_ops, 'search_similar_multilingual'):
                similar_papers = await self.vector_ops.search_similar_multilingual(
                    query_embeddings, 
                    limit=limit,
                    similarity_threshold=0.6
                )
            else:
                # Fallback to basic similarity search
                logger.warning("⚠️ Using fallback vector search")
                similar_papers = []
                
                # Use first available embedding for basic search
                if query_embeddings:
                    first_embedding = next(iter(query_embeddings.values()))
                    if hasattr(self.vector_ops, 'search_similar'):
                        similar_papers = await self.vector_ops.search_similar(
                            first_embedding, limit, 0.6
                        )
            
            logger.info(f"🎯 Vector search found {len(similar_papers)} similar papers")
            return similar_papers
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []
    
    def _calculate_average_quality(self, papers: List[Dict[str, Any]]) -> float:
        """Calculate average quality score of papers"""
        try:
            if not papers:
                return 0.0
            
            quality_scores = [
                paper.get('context_quality_score', 0.5) 
                for paper in papers 
                if paper.get('context_quality_score') is not None
            ]
            
            if not quality_scores:
                return 0.5  # Default quality
            
            return sum(quality_scores) / len(quality_scores)
            
        except Exception as e:
            logger.error(f"❌ Average quality calculation failed: {e}")
            return 0.5

    def _combine_search_results(self, traditional_papers: List[Dict[str, Any]], 
                              vector_papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine traditional and vector search results intelligently"""
        try:
            # Mark paper sources
            for paper in traditional_papers:
                paper['search_method'] = 'traditional'
                paper['discovery_priority'] = 1
            
            for paper in vector_papers:
                paper['search_method'] = 'vector_similarity'
                paper['discovery_priority'] = 2
            
            # Combine and deduplicate
            all_papers = traditional_papers + vector_papers
            unique_papers = self._remove_duplicate_papers(all_papers)
            
            # Sort by quality and relevance
            sorted_papers = sorted(
                unique_papers,
                key=lambda p: (
                    p.get('similarity_score', 0.5),
                    p.get('context_quality_score', 0.5),
                    -p.get('discovery_priority', 3)
                ),
                reverse=True
            )
            
            logger.info(f"🔀 Combined search results: {len(sorted_papers)} unique papers")
            return sorted_papers
            
        except Exception as e:
            logger.error(f"❌ Search result combination failed: {e}")
            return traditional_papers  # Fallback to traditional results

    async def _multi_ai_paper_analysis(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze papers using multi-AI agent system"""
        try:
            if not papers:
                return []
            
            # ✅ FIXED: Check if multi-AI orchestrator has required method
            if hasattr(self.multi_ai_orchestrator, 'process_paper_analysis'):
                agent_results = await self.multi_ai_orchestrator.process_paper_analysis(papers)
            else:
                logger.warning("⚠️ Multi-AI orchestrator unavailable - using fallback analysis")
                return await self._fallback_paper_analysis(papers)
            
            # Process agent results and combine with original papers
            analyzed_papers = []
            
            for i, paper in enumerate(papers):
                # Find corresponding agent result
                agent_result = None
                for result in agent_results:
                    if result.task_id == f"paper_analysis_{i}" and result.success:
                        agent_result = result
                        break
                
                if agent_result:
                    # Extract analysis from agent result
                    if 'context' in agent_result.result:
                        # Groq analysis result
                        context = agent_result.result['context']
                        enhanced_paper = {
                            **paper,
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
                            'ai_agent_used': agent_result.agent_type.value,
                            'analysis_confidence': context.analysis_confidence,
                            'processing_time': agent_result.processing_time
                        }
                    
                    elif 'analysis' in agent_result.result:
                        # Kimi analysis result
                        analysis = agent_result.result['analysis']
                        enhanced_paper = {
                            **paper,
                            'detailed_analysis': analysis.content,
                            'ai_agent_used': agent_result.agent_type.value,
                            'analysis_confidence': analysis.confidence_score,
                            'processing_time': agent_result.processing_time,
                            'context_quality_score': analysis.confidence_score
                        }
                    
                    else:
                        # Fallback
                        enhanced_paper = {**paper, 'ai_agent_used': 'fallback'}
                    
                    analyzed_papers.append(enhanced_paper)
                
                else:
                    # No agent result, use fallback
                    fallback_paper = {
                        **paper,
                        'context_summary': 'Analysis failed - using fallback',
                        'context_quality_score': 0.3,
                        'ai_agent_used': 'fallback'
                    }
                    analyzed_papers.append(fallback_paper)
            
            logger.info(f"🤖 Multi-AI analysis completed: {len(analyzed_papers)} papers")
            return analyzed_papers
            
        except Exception as e:
            logger.error(f"❌ Multi-AI paper analysis failed: {e}")
            return papers  # Return original papers as fallback

    async def _fallback_paper_analysis(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback paper analysis when multi-AI is unavailable"""
        try:
            analyzed_papers = []
            
            for paper in papers:
                # Basic analysis using available information
                enhanced_paper = {
                    **paper,
                    'context_summary': paper.get('abstract', 'No summary available')[:500],
                    'research_domain': 'General Research',
                    'context_quality_score': 0.5,
                    'ai_agent_used': 'fallback',
                    'analysis_confidence': 0.3
                }
                analyzed_papers.append(enhanced_paper)
            
            logger.info(f"🔄 Fallback analysis completed for {len(analyzed_papers)} papers")
            return analyzed_papers
            
        except Exception as e:
            logger.error(f"❌ Fallback analysis failed: {e}")
            return papers
    
    async def _multi_ai_relationship_analysis(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze relationships using multi-AI agents"""
        try:
            if len(papers) < 2:
                return []
            
            # Create strategic paper pairs for relationship analysis
            paper_pairs = []
            limited_papers = papers[:min(10, len(papers))]  # Limit for performance
            
            for i in range(len(limited_papers)):
                for j in range(i + 1, min(i + 4, len(limited_papers))):
                    paper_pairs.append((limited_papers[i], limited_papers[j]))
            
            # Limit total pairs
            paper_pairs = paper_pairs[:15]
            
            if not paper_pairs:
                return []
            
            # ✅ FIXED: Check if method exists
            if hasattr(self.multi_ai_orchestrator, 'process_relationship_analysis'):
                agent_results = await self.multi_ai_orchestrator.process_relationship_analysis(paper_pairs)
            else:
                logger.warning("⚠️ Multi-AI relationship analysis unavailable")
                return []
            
            # Process agent results
            relationships = []
            
            for i, (paper1, paper2) in enumerate(paper_pairs):
                # Find corresponding agent result
                agent_result = None
                for result in agent_results:
                    if result.task_id == f"relationship_analysis_{i}" and result.success:
                        agent_result = result
                        break
                
                if agent_result:
                    if 'relationship' in agent_result.result:
                        # Standard relationship result
                        relationship = agent_result.result['relationship']
                        rel_data = {
                            'paper1_id': paper1.get('id'),
                            'paper2_id': paper2.get('id'),
                            'relationship_type': relationship.relationship_type,
                            'relationship_strength': relationship.relationship_strength,
                            'relationship_context': relationship.relationship_context,
                            'connection_reasoning': relationship.connection_reasoning,
                            'ai_agent_used': agent_result.agent_type.value,
                            'confidence_score': relationship.confidence_score,
                            'processing_time': agent_result.processing_time
                        }
                        
                        # Only keep strong relationships
                        if relationship.relationship_strength >= 0.4:
                            relationships.append(rel_data)
            
            logger.info(f"🔗 Multi-AI relationship analysis: {len(relationships)} strong relationships found")
            return relationships
            
        except Exception as e:
            logger.error(f"❌ Multi-AI relationship analysis failed: {e}")
            return []
    
    async def _build_multilingual_graph(self, papers: List[Dict[str, Any]],
                                      relationships: List[Dict[str, Any]],
                                      multilingual_keywords: Dict[str, str]) -> Dict[str, Any]:
        """Build enhanced graph with multilingual metadata"""
        try:
            # ✅ FIXED: Check if parent method exists
            if hasattr(super(), '_build_contextual_graph'):
                base_graph = await super()._build_contextual_graph(
                    papers, relationships, multilingual_keywords.get('original', '')
                )
            else:
                # Fallback graph structure
                base_graph = {
                    'nodes': [{'id': str(i), 'title': p.get('title', 'Unknown')} for i, p in enumerate(papers)],
                    'edges': [],
                    'metadata': {}
                }
            
            # Add multilingual enhancements
            if 'metadata' not in base_graph:
                base_graph['metadata'] = {}
            
            # Enhanced multilingual metadata
            base_graph['metadata'].update({
                'multilingual_query': multilingual_keywords,
                'languages_searched': list(multilingual_keywords.keys()),
                'ai_agents_used': list(getattr(self.multi_ai_orchestrator, 'active_agents', {}).keys()),
                'vector_search_enabled': self.vector_ops is not None,
                'embedding_model': getattr(self.embedding_generator, 'model_name', 'unknown'),
                'translation_service_stats': self.translation_service.get_translation_stats(),
                'workflow_type': 'multilingual_ai_enhanced'
            })
            
            # Add language distribution to nodes
            if 'nodes' in base_graph:
                for node in base_graph['nodes']:
                    paper_id = node.get('id')
                    # Find corresponding paper
                    paper = next((p for p in papers if str(p.get('id', '')) == str(paper_id)), None)
                    if paper:
                        node.update({
                            'search_language': paper.get('search_language', 'unknown'),
                            'ai_agent_used': paper.get('ai_agent_used', 'unknown'),
                            'search_method': paper.get('search_method', 'traditional'),
                            'analysis_confidence': paper.get('analysis_confidence', 0.5)
                        })
            
            # Add AI agent metadata to edges
            if 'edges' in base_graph:
                for edge in base_graph['edges']:
                    # Find corresponding relationship
                    rel = next((r for r in relationships 
                              if str(r.get('paper1_id', '')) == str(edge.get('source', '')) and 
                                 str(r.get('paper2_id', '')) == str(edge.get('target', ''))), None)
                    if rel:
                        edge.update({
                            'ai_agent_used': rel.get('ai_agent_used', 'unknown'),
                            'confidence_score': rel.get('confidence_score', 0.5),
                            'analysis_type': 'ai_enhanced'
                        })
            
            logger.info("📊 Built enhanced multilingual graph with AI metadata")
            return base_graph
            
        except Exception as e:
            logger.error(f"❌ Multilingual graph building failed: {e}")
            # Return basic graph as fallback
            return {
                'nodes': [{'id': str(i), 'title': p.get('title', 'Unknown')} for i, p in enumerate(papers)],
                'edges': [],
                'metadata': {'error': str(e), 'workflow_type': 'multilingual_ai_enhanced'}
            }
    
    async def get_workflow_health_extended(self) -> Dict[str, Any]:
        """Get extended health status including multilingual components"""
        try:
            # Get base health
            base_health = await self.get_workflow_health()
            
            # Add multilingual component health
            multilingual_health = {
                'translation_service': {
                    'status': 'healthy',
                    'stats': self.translation_service.get_translation_stats()
                },
                'embedding_generator': {
                    'status': 'healthy',
                    'stats': getattr(self.embedding_generator, 'get_embedding_stats', lambda: {})()
                },
                'multi_ai_orchestrator': {
                    'status': 'healthy',
                    'stats': getattr(self.multi_ai_orchestrator, 'get_agent_statistics', lambda: {})()
                },
                'language_detector': {
                    'status': 'healthy',
                    'supported_languages': len(getattr(self.language_detector, 'supported_languages', []))
                },
                'vector_operations': {
                    'status': 'healthy' if self.vector_ops else 'unavailable',
                    'stats': await self.vector_ops.get_vector_statistics() if self.vector_ops else {}
                }
            }
            
            # Combine health data
            extended_health = {
                **base_health,
                'multilingual_components': multilingual_health,
                'workflow_type': 'multilingual_ai_enhanced',
                'capabilities': [
                    'multilingual_search',
                    'vector_similarity',
                    'multi_ai_analysis',
                    'relationship_discovery',
                    'graph_visualization'
                ]
            }
            
            return extended_health
            
        except Exception as e:
            logger.error(f"❌ Extended health check failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def shutdown(self):
        """Shutdown multilingual workflow components"""
        try:
            if hasattr(self.multi_ai_orchestrator, 'shutdown'):
                await self.multi_ai_orchestrator.shutdown()
            
            self.translation_service.clear_cache()
            
            if hasattr(self.embedding_generator, 'clear_cache'):
                self.embedding_generator.clear_cache()
            
            logger.info("🔒 Multilingual workflow shutdown completed")
        except Exception as e:
            logger.error(f"❌ Shutdown failed: {e}")
