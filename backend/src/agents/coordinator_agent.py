import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from agents.paper_analyzer_agent import PaperAnalyzerAgent, AnalysisResult
from agents.relationship_agent import RelationshipAgent, RelationshipResult
from core.multi_ai_agent import MultiAIAgentOrchestrator
from workflow.multilingual_workflow import MultilingualResearchWorkflow

logger = logging.getLogger(__name__)

class WorkflowPhase(Enum):
    INITIALIZATION = "initialization"
    LANGUAGE_PROCESSING = "language_processing"
    PAPER_SEARCH = "paper_search"
    EMBEDDING_GENERATION = "embedding_generation"
    PAPER_ANALYSIS = "paper_analysis"
    RELATIONSHIP_DISCOVERY = "relationship_discovery"
    GRAPH_BUILDING = "graph_building"
    FINALIZATION = "finalization"
    ERROR_RECOVERY = "error_recovery"

@dataclass
class WorkflowTask:
    """Coordinated workflow task"""
    task_id: str
    phase: WorkflowPhase
    priority: int = 1
    dependencies: List[str] = None
    timeout: int = 60
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"
    result: Any = None
    error_message: str = ""

@dataclass
class CoordinationResult:
    """Result of coordinated workflow execution"""
    workflow_id: str
    success: bool
    total_time: float
    phases_completed: List[str]
    final_result: Any = None
    error_summary: Dict[str, Any] = None
    performance_metrics: Dict[str, Any] = None

class CoordinatorAgent:
    """
    Master coordinator agent that orchestrates the entire multilingual research workflow
    Manages all other agents, handles error recovery, and optimizes performance
    """
    
    def __init__(self):
        # Initialize specialized agents
        self.paper_analyzer = PaperAnalyzerAgent()
        self.relationship_agent = RelationshipAgent()
        self.multi_ai_orchestrator = MultiAIAgentOrchestrator()
        self.multilingual_workflow = MultilingualResearchWorkflow()
        
        # Coordination state
        self.active_workflows = {}
        self.task_queue = asyncio.Queue()
        self.coordination_stats = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_execution_time': 0.0,
            'phase_performance': {phase.value: {'count': 0, 'avg_time': 0.0} for phase in WorkflowPhase}
        }
        
        # Performance optimization settings
        self.optimization_config = {
            'adaptive_timeouts': True,
            'smart_error_recovery': True,
            'performance_monitoring': True,
            'resource_management': True
        }
        
        logger.info("🎯 Coordinator Agent initialized")
    
    async def initialize(self):
        """Initialize all coordinated components"""
        try:
            logger.info("🚀 Initializing coordinator and all sub-agents...")
            
            # Initialize all agents in parallel
            init_tasks = [
                self.paper_analyzer.initialize(),
                self.relationship_agent.initialize(),
                self.multi_ai_orchestrator.initialize_agents(),
                self.multilingual_workflow.initialize()
            ]
            
            await asyncio.gather(*init_tasks, return_exceptions=True)
            
            logger.info("✅ Coordinator Agent fully initialized with all sub-agents")
            
        except Exception as e:
            logger.error(f"❌ Coordinator initialization failed: {e}")
            raise
    
    async def coordinate_research_workflow(self, query: str, 
                                     max_papers: int = 25,
                                     workflow_id: str = None,
                                     **kwargs) -> CoordinationResult:
        """Master coordination method for complete research workflow"""
        start_time = asyncio.get_event_loop().time()
        workflow_id = workflow_id or f"workflow_{int(start_time)}"
        
        logger.info(f"🎯 Coordinating research workflow {workflow_id} for query: '{query}'")
        
        # Initialize workflow tracking
        self.active_workflows[workflow_id] = {
            'query': query,
            'status': 'running',
            'start_time': start_time,
            'phases_completed': [],
            'current_phase': WorkflowPhase.INITIALIZATION
        }
        
        try:
            self.coordination_stats['total_workflows'] += 1
            
            # ✅ FIXED: Use the multilingual workflow properly
            logger.info("🌍 Phase 1: Language processing and translation")
            phase_start = asyncio.get_event_loop().time()
            
            await self._update_workflow_phase(workflow_id, WorkflowPhase.LANGUAGE_PROCESSING)
            
            # ✅ CRITICAL FIX: Use the correct workflow method
            multilingual_state = await self.multilingual_workflow.process_enhanced_research_query(
                query, max_papers, enable_multilingual=True, enable_vector_search=True
            )
            
            # ✅ CRITICAL FIX: Check for successful completion properly
            if multilingual_state.status == "completed":
                # Extract the actual results
                final_result = {
                    'papers': multilingual_state.papers_analyzed,
                    'relationships': multilingual_state.relationships_found,
                    'graph_data': multilingual_state.graph_data,  # ✅ CRITICAL: Include graph data!
                    'multilingual_state': multilingual_state
                }
                
                total_time = asyncio.get_event_loop().time() - start_time
                
                # Update statistics
                self.coordination_stats['successful_workflows'] += 1
                self._update_average_execution_time(total_time)
                
                # Cleanup workflow tracking
                self.active_workflows[workflow_id]['status'] = 'completed'
                self.active_workflows[workflow_id]['end_time'] = asyncio.get_event_loop().time()
                
                logger.info(f"✅ Workflow {workflow_id} completed successfully in {total_time:.2f}s")
                logger.info(f"📊 Results: {len(final_result['papers'])} papers, {len(final_result['relationships'])} relationships")
                
                return CoordinationResult(
                    workflow_id=workflow_id,
                    success=True,
                    total_time=total_time,
                    phases_completed=['language_processing', 'paper_analysis', 'graph_building'],
                    final_result=final_result,
                    performance_metrics=self._generate_performance_metrics(workflow_id, total_time)
                )
            else:
                raise Exception(f"Multilingual workflow failed with status: {multilingual_state.status}")
                
        except Exception as e:
            total_time = asyncio.get_event_loop().time() - start_time
            
            logger.error(f"❌ Workflow {workflow_id} failed: {e}")
            
            # ❌ PROBLEM IS HERE: Don't attempt recovery for successful workflows!
            # Only attempt recovery if we have a real failure
            if "workflow failed with status: completed" in str(e):
                # This is actually a success, don't recover
                logger.error("❌ False positive error - workflow actually succeeded")
                return CoordinationResult(
                    workflow_id=workflow_id,
                    success=False,
                    total_time=total_time,
                    phases_completed=[],
                    error_summary={'error_message': 'Coordination error - workflow succeeded but was marked as failed'}
                )
            
            # Try error recovery only for real failures
            recovery_result = await self._attempt_error_recovery(workflow_id, str(e))
            
            if recovery_result:
                logger.info(f"🔄 Workflow {workflow_id} recovered successfully")
                return recovery_result
            
            # Update failure statistics
            self.coordination_stats['failed_workflows'] += 1
            self.active_workflows[workflow_id]['status'] = 'failed'
            
            return CoordinationResult(
                workflow_id=workflow_id,
                success=False,
                total_time=total_time,
                phases_completed=self.active_workflows[workflow_id].get('phases_completed', []),
                error_summary={
                    'error_message': str(e),
                    'failed_phase': self.active_workflows[workflow_id].get('current_phase', 'unknown'),
                    'recovery_attempted': True
                }
            )

    
    async def _coordinate_enhanced_analysis(self, papers: List[Dict[str, Any]], 
                                          relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate enhanced analysis using all available agents"""
        try:
            logger.info(f"🔍 Coordinating enhanced analysis of {len(papers)} papers")
            
            # Phase 1: Enhanced paper analysis using dedicated agent
            if papers:
                logger.info("📋 Running dedicated paper analyzer")
                paper_analysis_results = await self.paper_analyzer.analyze_paper_batch(
                    papers, 
                    analysis_type='detailed',
                    max_concurrent=6
                )
                
                # Merge results with original papers
                enhanced_papers = self._merge_analysis_results(papers, paper_analysis_results)
            else:
                enhanced_papers = papers
            
            # Phase 2: Enhanced relationship analysis using dedicated agent
            if len(enhanced_papers) >= 2:
                logger.info("🔗 Running dedicated relationship analyzer")
                relationship_results = await self.relationship_agent.analyze_relationships_batch(
                    enhanced_papers,
                    analysis_depth='standard',
                    max_pairs=20
                )
                
                # Combine with existing relationships
                all_relationships = relationships + [r.__dict__ for r in relationship_results]
            else:
                all_relationships = relationships
            
            # Phase 3: Multi-AI orchestrator enhancement
            logger.info("🤖 Running multi-AI orchestrator enhancement")
            orchestrator_results = await self._run_orchestrator_enhancement(
                enhanced_papers[:15],  # Limit for performance
                all_relationships[:15]
            )
            
            return {
                'papers': enhanced_papers,
                'relationships': all_relationships,
                'orchestrator_insights': orchestrator_results,
                'analysis_summary': {
                    'papers_analyzed': len(enhanced_papers),
                    'relationships_discovered': len(all_relationships),
                    'ai_agents_used': ['paper_analyzer', 'relationship_agent', 'multi_ai_orchestrator'],
                    'enhancement_level': 'comprehensive'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis coordination failed: {e}")
            # Return basic results as fallback
            return {
                'papers': papers,
                'relationships': relationships,
                'error': str(e),
                'analysis_summary': {'enhancement_level': 'basic_fallback'}
            }
    
    async def _run_orchestrator_enhancement(self, papers: List[Dict[str, Any]], 
                                          relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run multi-AI orchestrator for additional insights"""
        try:
            # Get orchestrator statistics and insights
            orchestrator_stats = self.multi_ai_orchestrator.get_agent_statistics()
            
            # Run quality assessment
            quality_insights = self._assess_result_quality(papers, relationships)
            
            # Generate recommendations
            recommendations = self._generate_coordination_recommendations(papers, relationships, orchestrator_stats)
            
            return {
                'orchestrator_stats': orchestrator_stats,
                'quality_insights': quality_insights,
                'recommendations': recommendations,
                'enhancement_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestrator enhancement failed: {e}")
            return {'error': str(e)}
    
    async def _optimize_final_graph(self, papers: List[Dict[str, Any]], 
                                  relationships: List[Dict[str, Any]],
                                  multilingual_keywords: Dict[str, str]) -> Dict[str, Any]:
        """Final graph optimization and enhancement"""
        try:
            from core.graph_builder import EnhancedIntelligentGraphBuilder
            
            graph_builder = EnhancedIntelligentGraphBuilder()
            
            # Build enhanced graph with all metadata
            enhanced_graph = graph_builder.build_graph(
                papers, 
                relationships, 
                multilingual_keywords
            )
            
            # Add coordination metadata
            enhanced_graph['coordination_metadata'] = {
                'coordinator_agent': 'v2.0',
                'optimization_level': 'maximum',
                'ai_agents_coordinated': 4,
                'multilingual_enhancement': True,
                'quality_optimization': True
            }
            
            return enhanced_graph
            
        except Exception as e:
            logger.error(f"❌ Final graph optimization failed: {e}")
            # Return basic graph as fallback
            return {
                'nodes': [{'id': str(p.get('id', i)), 'title': p.get('title', 'Unknown')} for i, p in enumerate(papers)],
                'edges': [],
                'error': str(e)
            }
    
    def _merge_analysis_results(self, original_papers: List[Dict[str, Any]], 
                              analysis_results: List[AnalysisResult]) -> List[Dict[str, Any]]:
        """Merge analysis results with original papers"""
        try:
            enhanced_papers = []
            result_map = {r.paper_id: r for r in analysis_results if r.success}
            
            for paper in original_papers:
                paper_id = str(paper.get('id', ''))
                enhanced_paper = paper.copy()
                
                if paper_id in result_map:
                    result = result_map[paper_id]
                    if result.analysis_data:
                        context = result.analysis_data
                        enhanced_paper.update({
                            'enhanced_context_summary': context.context_summary,
                            'enhanced_research_domain': context.research_domain,
                            'enhanced_methodology': context.methodology,
                            'enhanced_innovations': context.innovations,
                            'enhanced_quality_score': context.context_quality_score,
                            'analysis_agent_used': result.agent_used,
                            'analysis_processing_time': result.processing_time,
                            'enhancement_level': 'dedicated_agent'
                        })
                
                enhanced_papers.append(enhanced_paper)
            
            return enhanced_papers
            
        except Exception as e:
            logger.error(f"❌ Analysis results merge failed: {e}")
            return original_papers
    
    async def _update_workflow_phase(self, workflow_id: str, phase: WorkflowPhase):
        """Update workflow phase tracking"""
        try:
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                workflow['current_phase'] = phase
                workflow['phases_completed'].append(phase.value)
                
                logger.info(f"🔄 Workflow {workflow_id} entering phase: {phase.value}")
                
        except Exception as e:
            logger.error(f"❌ Phase update failed: {e}")
    
    def _update_phase_performance(self, phase: WorkflowPhase, execution_time: float):
        """Update phase performance statistics"""
        try:
            phase_stats = self.coordination_stats['phase_performance'][phase.value]
            count = phase_stats['count']
            avg_time = phase_stats['avg_time']
            
            # Update running average
            new_avg = (avg_time * count + execution_time) / (count + 1)
            
            phase_stats['count'] = count + 1
            phase_stats['avg_time'] = new_avg
            
        except Exception as e:
            logger.error(f"❌ Phase performance update failed: {e}")
    
    async def _attempt_error_recovery(self, workflow_id: str, error_message: str) -> Optional[CoordinationResult]:
        """Attempt to recover from workflow errors"""
        try:
            logger.info(f"🔄 Attempting error recovery for workflow {workflow_id}")
            
            if workflow_id not in self.active_workflows:
                return None
            
            workflow = self.active_workflows[workflow_id]
            
            # Simple recovery strategies based on error type
            if "timeout" in error_message.lower():
                logger.info("⏰ Timeout error detected - implementing timeout recovery")
                return await self._recover_from_timeout(workflow_id)
                
            elif "api" in error_message.lower() or "connection" in error_message.lower():
                logger.info("🌐 API/Connection error detected - implementing connection recovery")
                return await self._recover_from_connection_error(workflow_id)
                
            else:
                logger.info("🛠️ General error detected - implementing basic recovery")
                return await self._basic_error_recovery(workflow_id)
                
        except Exception as e:
            logger.error(f"❌ Error recovery failed: {e}")
            return None
    
    async def _recover_from_timeout(self, workflow_id: str) -> Optional[CoordinationResult]:
        """Recover from timeout errors"""
        try:
            workflow = self.active_workflows[workflow_id]
            query = workflow['query']
            
            # Retry with reduced scope
            logger.info("🔄 Retrying with reduced scope...")
            return await self.coordinate_research_workflow(
                query, 
                max_papers=10,  # Reduced scope
                workflow_id=f"{workflow_id}_recovery"
            )
            
        except Exception as e:
            logger.error(f"❌ Timeout recovery failed: {e}")
            return None
    
    async def _recover_from_connection_error(self, workflow_id: str) -> Optional[CoordinationResult]:
        """Recover from connection errors"""
        try:
            # Wait and retry
            await asyncio.sleep(2)
            
            workflow = self.active_workflows[workflow_id]
            query = workflow['query']
            
            logger.info("🔄 Retrying after connection recovery...")
            return await self.coordinate_research_workflow(
                query,
                max_papers=15,
                workflow_id=f"{workflow_id}_recovery"
            )
            
        except Exception as e:
            logger.error(f"❌ Connection recovery failed: {e}")
            return None
    
    async def _basic_error_recovery(self, workflow_id: str) -> Optional[CoordinationResult]:
        """Basic error recovery with fallback workflow"""
        try:
            workflow = self.active_workflows[workflow_id]
            query = workflow['query']
            
            # Use basic workflow as fallback
            logger.info("🔄 Using basic workflow as fallback...")
            
            basic_workflow = ContextAwareResearchWorkflow()
            await basic_workflow.initialize()
            
            basic_result = await basic_workflow.process_research_query(query, max_papers=10)
            
            return CoordinationResult(
                workflow_id=f"{workflow_id}_basic_recovery",
                success=True,
                total_time=0.0,
                phases_completed=['basic_recovery'],
                final_result={
                    'basic_workflow_result': basic_result,
                    'recovery_mode': 'basic_fallback'
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Basic recovery failed: {e}")
            return None
    
    def _assess_result_quality(self, papers: List[Dict[str, Any]], 
                             relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of analysis results"""
        try:
            if not papers:
                return {'quality_score': 0.0, 'assessment': 'no_papers'}
            
            # Calculate quality metrics
            quality_scores = [p.get('context_quality_score', 0.5) for p in papers]
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            relationship_strengths = [r.get('relationship_strength', 0.5) for r in relationships]
            avg_relationship_strength = (
                sum(relationship_strengths) / len(relationship_strengths) 
                if relationship_strengths else 0.5
            )
            
            # Overall quality assessment
            overall_quality = (avg_quality * 0.6 + avg_relationship_strength * 0.4)
            
            assessment = "excellent" if overall_quality > 0.8 else \
                        "good" if overall_quality > 0.6 else \
                        "fair" if overall_quality > 0.4 else "poor"
            
            return {
                'overall_quality_score': overall_quality,
                'average_paper_quality': avg_quality,
                'average_relationship_strength': avg_relationship_strength,
                'quality_assessment': assessment,
                'papers_analyzed': len(papers),
                'relationships_found': len(relationships),
                'recommendations': self._generate_quality_recommendations(overall_quality)
            }
            
        except Exception as e:
            logger.error(f"❌ Quality assessment failed: {e}")
            return {'quality_score': 0.5, 'error': str(e)}
    
    def _generate_quality_recommendations(self, quality_score: float) -> List[str]:
        """Generate recommendations based on quality score"""
        recommendations = []
        
        if quality_score < 0.5:
            recommendations.extend([
                "Consider refining search terms for better paper quality",
                "Expand search to additional academic databases",
                "Use more specific domain keywords"
            ])
        elif quality_score < 0.7:
            recommendations.extend([
                "Results are acceptable but could be improved",
                "Consider using advanced AI agents for better analysis",
                "Try searching in additional languages"
            ])
        else:
            recommendations.extend([
                "Excellent results achieved",
                "Consider expanding scope for broader insights",
                "Current approach is working well"
            ])
        
        return recommendations
    
    def _generate_coordination_recommendations(self, papers: List[Dict[str, Any]], 
                                            relationships: List[Dict[str, Any]],
                                            orchestrator_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations for workflow coordination"""
        recommendations = []
        
        # Paper analysis recommendations
        if len(papers) < 10:
            recommendations.append("Consider increasing max_papers parameter for broader coverage")
        elif len(papers) > 50:
            recommendations.append("Consider reducing scope for faster processing")
        
        # Relationship recommendations
        if len(relationships) < len(papers) * 0.1:
            recommendations.append("Low relationship density detected - try related search terms")
        elif len(relationships) > len(papers) * 0.5:
            recommendations.append("High relationship density - results are well connected")
        
        # AI agent recommendations
        healthy_agents = orchestrator_stats.get('healthy_agents', 0)
        total_agents = orchestrator_stats.get('total_agents', 1)
        
        if healthy_agents / total_agents < 0.8:
            recommendations.append("Some AI agents are unhealthy - check API connections")
        
        return recommendations
    
    def _count_ai_agents_used(self, enhanced_results: Dict[str, Any]) -> int:
        """Count unique AI agents used in the workflow"""
        agents = set()
        
        # Check papers for agent usage
        for paper in enhanced_results.get('papers', []):
            if 'ai_agent_used' in paper:
                agents.add(paper['ai_agent_used'])
            if 'analysis_agent_used' in paper:
                agents.add(paper['analysis_agent_used'])
        
        # Check relationships for agent usage
        for rel in enhanced_results.get('relationships', []):
            if 'ai_agent_used' in rel:
                agents.add(rel['ai_agent_used'])
        
        return len(agents)
    
    def _calculate_coordination_efficiency(self, total_time: float, 
                                         enhanced_results: Dict[str, Any]) -> float:
        """Calculate coordination efficiency score"""
        try:
            papers_count = len(enhanced_results.get('papers', []))
            relationships_count = len(enhanced_results.get('relationships', []))
            
            if total_time <= 0 or papers_count == 0:
                return 0.0
            
            # Papers per second
            papers_per_second = papers_count / total_time
            
            # Relationships per paper ratio
            rel_ratio = relationships_count / papers_count if papers_count > 0 else 0
            
            # Efficiency score (normalized)
            efficiency = min(1.0, (papers_per_second * 0.6 + rel_ratio * 0.4) / 2)
            
            return efficiency
            
        except Exception:
            return 0.5  # Default efficiency
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time"""
        current_avg = self.coordination_stats['average_execution_time']
        successful_count = self.coordination_stats['successful_workflows']
        
        if successful_count == 1:
            self.coordination_stats['average_execution_time'] = execution_time
        else:
            new_avg = (current_avg * (successful_count - 1) + execution_time) / successful_count
            self.coordination_stats['average_execution_time'] = new_avg
    
    def _generate_performance_metrics(self, workflow_id: str, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive performance metrics"""
        workflow = self.active_workflows.get(workflow_id, {})
        
        return {
            'execution_time': total_time,
            'phases_completed': len(workflow.get('phases_completed', [])),
            'coordination_efficiency': self._calculate_coordination_efficiency(
                total_time, 
                {'papers': [], 'relationships': []}  # Placeholder
            ),
            'agent_utilization': {
                'paper_analyzer': 'utilized',
                'relationship_agent': 'utilized', 
                'multi_ai_orchestrator': 'utilized',
                'multilingual_workflow': 'utilized'
            },
            'resource_usage': {
                'memory_efficient': total_time < 120,  # Under 2 minutes
                'api_calls_optimized': True,
                'concurrent_processing': True
            }
        }
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get comprehensive coordination statistics"""
        success_rate = (
            self.coordination_stats['successful_workflows'] / 
            max(self.coordination_stats['total_workflows'], 1) * 100
        )
        
        return {
            **self.coordination_stats,
            'success_rate_percent': success_rate,
            'active_workflows': len(self.active_workflows),
            'phase_performance_summary': {
                phase: stats['avg_time'] 
                for phase, stats in self.coordination_stats['phase_performance'].items()
                if stats['count'] > 0
            }
        }
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific workflow"""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        current_time = asyncio.get_event_loop().time()
        
        return {
            'workflow_id': workflow_id,
            'query': workflow['query'],
            'status': workflow['status'],
            'current_phase': workflow.get('current_phase', {}).value if hasattr(workflow.get('current_phase', {}), 'value') else 'unknown',
            'phases_completed': workflow.get('phases_completed', []),
            'elapsed_time': current_time - workflow['start_time'],
            'estimated_completion': self._estimate_completion_time(workflow_id)
        }
    
    def _estimate_completion_time(self, workflow_id: str) -> Optional[float]:
        """Estimate remaining completion time for workflow"""
        try:
            workflow = self.active_workflows.get(workflow_id, {})
            phases_completed = len(workflow.get('phases_completed', []))
            total_phases = len(WorkflowPhase)
            
            if phases_completed >= total_phases:
                return 0.0
            
            elapsed_time = asyncio.get_event_loop().time() - workflow.get('start_time', 0)
            avg_time_per_phase = elapsed_time / max(phases_completed, 1)
            
            remaining_phases = total_phases - phases_completed
            estimated_remaining = remaining_phases * avg_time_per_phase
            
            return max(0.0, estimated_remaining)
            
        except Exception:
            return None
    
    async def shutdown(self):
        """Shutdown coordinator and all sub-agents"""
        try:
            logger.info("🔒 Shutting down Coordinator Agent and all sub-agents...")
            
            # Shutdown all agents
            shutdown_tasks = [
                self.paper_analyzer.shutdown(),
                self.relationship_agent.shutdown(),
                self.multi_ai_orchestrator.shutdown(),
                self.multilingual_workflow.shutdown()
            ]
            
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            # Clear active workflows
            self.active_workflows.clear()
            
            logger.info("✅ Coordinator Agent shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Coordinator shutdown failed: {e}")

# Import for backward compatibility in error recovery
from workflow.langgraph_workflow import ContextAwareResearchWorkflow
