import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time

from core.llm_processor import MultiAPILLMProcessor
from integrations.kimi_client import KimiClient

logger = logging.getLogger(__name__)

class AgentType(Enum):
    GROQ_FAST = "groq_fast"
    GROQ_DETAILED = "groq_detailed" 
    KIMI_CONTEXT = "kimi_context"
    KIMI_ANALYSIS = "kimi_analysis"

@dataclass
class AgentTask:
    task_id: str
    agent_type: AgentType
    task_type: str  # 'paper_analysis', 'relationship_analysis', 'context_extraction'
    input_data: Dict[str, Any]
    priority: int = 1  # 1=high, 2=medium, 3=low
    timeout: int = 30
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class AgentResult:
    task_id: str
    agent_type: AgentType
    success: bool
    result: Any
    processing_time: float
    error_message: Optional[str] = None
    agent_id: str = ""

class MultiAIAgentOrchestrator:
    """
    Advanced multi-AI agent orchestrator that coordinates between:
    - Groq agents (fast processing)  
    - Kimi agents (detailed analysis)
    - Load balancing and task distribution
    """
    
    def __init__(self):
        # Initialize AI clients
        self.groq_processor = MultiAPILLMProcessor()
        self.kimi_client = KimiClient()
        
        # Agent pools
        self.active_agents = {
            AgentType.GROQ_FAST: [],
            AgentType.GROQ_DETAILED: [],
            AgentType.KIMI_CONTEXT: [],
            AgentType.KIMI_ANALYSIS: []
        }
        
        # Task queue and management
        self.task_queue = asyncio.Queue()
        self.results_cache = {}
        self.agent_stats = {agent_type: {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_processing_time': 0.0,
            'current_load': 0
        } for agent_type in AgentType}
        
        # Configuration
        self.max_concurrent_tasks = 12
        self.task_timeout = 60
        
        logger.info("🤖 Multi-AI Agent Orchestrator initialized")
        logger.info(f"⚙️ Max concurrent tasks: {self.max_concurrent_tasks}")
    
    async def initialize_agents(self):
        """Initialize agent pools"""
        try:
            # Initialize Groq agents (using existing processor)
            self.active_agents[AgentType.GROQ_FAST] = ['groq_fast_1', 'groq_fast_2']
            self.active_agents[AgentType.GROQ_DETAILED] = ['groq_detailed_1', 'groq_detailed_2']
            
            # Initialize Kimi agents
            await self.kimi_client.initialize()
            self.active_agents[AgentType.KIMI_CONTEXT] = ['kimi_context_1']
            self.active_agents[AgentType.KIMI_ANALYSIS] = ['kimi_analysis_1']
            
            logger.info(f"✅ Initialized {sum(len(agents) for agents in self.active_agents.values())} agents")
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            raise
    
    async def submit_task(self, task: AgentTask) -> str:
        """Submit task to the orchestrator"""
        try:
            await self.task_queue.put(task)
            logger.info(f"📥 Task {task.task_id} submitted to {task.agent_type.value}")
            return task.task_id
        except Exception as e:
            logger.error(f"❌ Task submission failed: {e}")
            raise
    
    async def submit_batch_tasks(self, tasks: List[AgentTask]) -> List[str]:
        """Submit multiple tasks for batch processing"""
        task_ids = []
        try:
            for task in tasks:
                task_id = await self.submit_task(task)
                task_ids.append(task_id)
            
            logger.info(f"📥 Submitted batch of {len(tasks)} tasks")
            return task_ids
            
        except Exception as e:
            logger.error(f"❌ Batch task submission failed: {e}")
            return task_ids
    
    async def process_paper_analysis(self, papers: List[Dict[str, Any]]) -> List[AgentResult]:
        """Process paper analysis using optimal agent allocation"""
        try:
            # Create tasks with intelligent agent allocation
            tasks = []
            
            for i, paper in enumerate(papers):
                # Allocate based on paper complexity and available agents
                if len(paper.get('abstract', '')) > 1000 or paper.get('content', ''):
                    # Complex papers go to detailed agents
                    agent_type = AgentType.KIMI_ANALYSIS if i % 3 == 0 else AgentType.GROQ_DETAILED
                else:
                    # Simple papers go to fast agents
                    agent_type = AgentType.GROQ_FAST
                
                task = AgentTask(
                    task_id=f"paper_analysis_{i}",
                    agent_type=agent_type,
                    task_type="paper_analysis",
                    input_data={
                        'paper': paper,
                        'analysis_depth': 'detailed' if agent_type in [AgentType.KIMI_ANALYSIS, AgentType.GROQ_DETAILED] else 'standard'
                    },
                    priority=1,
                    timeout=45 if agent_type in [AgentType.KIMI_ANALYSIS] else 30
                )
                tasks.append(task)
            
            # Submit and process tasks
            task_ids = await self.submit_batch_tasks(tasks)
            results = await self.process_task_batch(tasks)
            
            logger.info(f"✅ Completed paper analysis for {len(papers)} papers")
            return results
            
        except Exception as e:
            logger.error(f"❌ Paper analysis processing failed: {e}")
            return []
    
    async def process_relationship_analysis(self, paper_pairs: List[tuple]) -> List[AgentResult]:
        """Process relationship analysis between paper pairs"""
        try:
            tasks = []
            
            for i, (paper1, paper2) in enumerate(paper_pairs):
                # Alternate between agent types for load balancing
                agent_type = AgentType.KIMI_CONTEXT if i % 4 == 0 else AgentType.GROQ_FAST
                
                task = AgentTask(
                    task_id=f"relationship_analysis_{i}",
                    agent_type=agent_type,
                    task_type="relationship_analysis",
                    input_data={
                        'paper1': paper1,
                        'paper2': paper2,
                        'analysis_type': 'semantic_similarity'
                    },
                    priority=2,
                    timeout=25
                )
                tasks.append(task)
            
            results = await self.process_task_batch(tasks)
            
            logger.info(f"✅ Completed relationship analysis for {len(paper_pairs)} pairs")
            return results
            
        except Exception as e:
            logger.error(f"❌ Relationship analysis processing failed: {e}")
            return []
    
    async def process_task_batch(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Process a batch of tasks with concurrency control"""
        try:
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            
            async def process_single_task(task: AgentTask) -> AgentResult:
                async with semaphore:
                    return await self.execute_task(task)
            
            # Create task coroutines
            task_coroutines = [process_single_task(task) for task in tasks]
            
            # Execute with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*task_coroutines, return_exceptions=True),
                timeout=self.task_timeout * 2  # Allow extra time for batch
            )
            
            # Filter successful results
            successful_results = []
            for result in results:
                if isinstance(result, AgentResult):
                    successful_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
            
            logger.info(f"✅ Processed {len(successful_results)}/{len(tasks)} tasks successfully")
            return successful_results
            
        except asyncio.TimeoutError:
            logger.error("⏰ Task batch processing timed out")
            return []
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return []
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute individual task with appropriate agent"""
        start_time = time.time()
        
        try:
            # Update agent load
            self.agent_stats[task.agent_type]['current_load'] += 1
            
            # Route task to appropriate agent
            if task.agent_type in [AgentType.GROQ_FAST, AgentType.GROQ_DETAILED]:
                result = await self._execute_groq_task(task)
            elif task.agent_type in [AgentType.KIMI_CONTEXT, AgentType.KIMI_ANALYSIS]:
                result = await self._execute_kimi_task(task)
            else:
                raise ValueError(f"Unknown agent type: {task.agent_type}")
            
            processing_time = time.time() - start_time
            
            # Update stats
            stats = self.agent_stats[task.agent_type]
            stats['tasks_completed'] += 1
            stats['average_processing_time'] = (
                (stats['average_processing_time'] * (stats['tasks_completed'] - 1) + processing_time) / 
                stats['tasks_completed']
            )
            stats['current_load'] -= 1
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                success=True,
                result=result,
                processing_time=processing_time,
                agent_id=f"{task.agent_type.value}_1"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update error stats
            self.agent_stats[task.agent_type]['tasks_failed'] += 1
            self.agent_stats[task.agent_type]['current_load'] -= 1
            
            logger.error(f"❌ Task {task.task_id} failed: {e}")
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                success=False,
                result=None,
                processing_time=processing_time,
                error_message=str(e),
                agent_id=f"{task.agent_type.value}_1"
            )
    
    async def _execute_groq_task(self, task: AgentTask) -> Any:
        """Execute task using Groq agents"""
        try:
            if task.task_type == "paper_analysis":
                paper = task.input_data['paper']
                context = await self.groq_processor.analyze_paper_context(
                    paper.get('title', ''),
                    paper.get('abstract', ''),
                    paper.get('content', '')
                )
                return {
                    'context': context,
                    'analysis_type': 'groq_analysis'
                }
                
            elif task.task_type == "relationship_analysis":
                paper1 = task.input_data['paper1']
                paper2 = task.input_data['paper2']
                relationship = await self.groq_processor.analyze_paper_relationship(paper1, paper2)
                return {
                    'relationship': relationship,
                    'analysis_type': 'groq_relationship'
                }
            
            else:
                raise ValueError(f"Unsupported task type for Groq: {task.task_type}")
                
        except Exception as e:
            logger.error(f"❌ Groq task execution failed: {e}")
            raise
    
    async def _execute_kimi_task(self, task: AgentTask) -> Any:
        """Execute task using Kimi agents"""
        try:
            if task.task_type == "paper_analysis":
                paper = task.input_data['paper']
                # Kimi provides more detailed analysis
                analysis = await self.kimi_client.analyze_paper_detailed(
                    paper.get('title', ''),
                    paper.get('abstract', ''),
                    paper.get('content', ''),
                    analysis_depth='deep'
                )
                return {
                    'analysis': analysis,
                    'analysis_type': 'kimi_detailed'
                }
                
            elif task.task_type == "relationship_analysis":
                paper1 = task.input_data['paper1']
                paper2 = task.input_data['paper2']
                relationship = await self.kimi_client.analyze_relationship_detailed(paper1, paper2)
                return {
                    'relationship': relationship,
                    'analysis_type': 'kimi_relationship'
                }
            
            else:
                raise ValueError(f"Unsupported task type for Kimi: {task.task_type}")
                
        except Exception as e:
            logger.error(f"❌ Kimi task execution failed: {e}")
            raise
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            'agent_stats': dict(self.agent_stats),
            'active_agents': {
                agent_type.value: len(agents) 
                for agent_type, agents in self.active_agents.items()
            },
            'queue_size': self.task_queue.qsize(),
            'cache_size': len(self.results_cache),
            'configuration': {
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'task_timeout': self.task_timeout
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_optimal_agent_for_task(self, task_type: str, complexity: str = 'medium') -> AgentType:
        """Get optimal agent type for a task based on current load and complexity"""
        try:
            if task_type == 'paper_analysis':
                if complexity == 'high':
                    return AgentType.KIMI_ANALYSIS
                elif complexity == 'medium':
                    return AgentType.GROQ_DETAILED
                else:
                    return AgentType.GROQ_FAST
            
            elif task_type == 'relationship_analysis':
                if complexity == 'high':
                    return AgentType.KIMI_CONTEXT
                else:
                    return AgentType.GROQ_FAST
            
            else:
                return AgentType.GROQ_FAST
                
        except Exception:
            return AgentType.GROQ_FAST
    
    async def shutdown(self):
        """Shutdown all agents gracefully"""
        try:
            # Cancel remaining tasks
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Shutdown Kimi client
            await self.kimi_client.close()
            
            logger.info("🔒 Multi-AI Agent Orchestrator shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Agent shutdown failed: {e}")
