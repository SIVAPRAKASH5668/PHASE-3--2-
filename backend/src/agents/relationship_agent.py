import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import itertools

from core.llm_processor import MultiAPILLMProcessor
from integrations.kimi_client import KimiClient

logger = logging.getLogger(__name__)

@dataclass
class RelationshipTask:
    """Relationship analysis task"""
    pair_id: str
    paper1: Dict[str, Any]
    paper2: Dict[str, Any]
    analysis_depth: str = 'standard'  # 'quick', 'standard', 'deep'
    timeout: int = 20

@dataclass
class RelationshipResult:
    """Relationship analysis result"""
    pair_id: str
    paper1_id: str
    paper2_id: str
    success: bool
    relationship_type: str = "unrelated"
    relationship_strength: float = 0.0
    relationship_context: str = ""
    connection_reasoning: str = ""
    confidence_score: float = 0.5
    processing_time: float = 0.0
    agent_used: str = "relationship_agent"
    error_message: Optional[str] = None

class RelationshipAgent:
    """
    Specialized agent for discovering and analyzing relationships between papers
    """
    
    def __init__(self):
        self.groq_processor = MultiAPILLMProcessor()
        self.kimi_client = KimiClient()
        
        # Relationship analysis strategies
        self.analysis_strategies = {
            'quick': {
                'timeout': 10, 
                'use_kimi': False, 
                'semantic_analysis': False,
                'max_concurrent': 8
            },
            'standard': {
                'timeout': 20, 
                'use_kimi': False, 
                'semantic_analysis': True,
                'max_concurrent': 6
            },
            'deep': {
                'timeout': 35, 
                'use_kimi': True, 
                'kimi_probability': 0.4,
                'semantic_analysis': True,
                'max_concurrent': 4
            }
        }
        
        # Relationship strength thresholds
        self.strength_thresholds = {
            'weak': 0.3,
            'moderate': 0.5,
            'strong': 0.7,
            'very_strong': 0.85
        }
        
        # Performance tracking
        self.stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'strong_relationships_found': 0,
            'weak_relationships_filtered': 0,
            'groq_analyses': 0,
            'kimi_analyses': 0,
            'failed_analyses': 0,
            'average_processing_time': 0.0
        }
        
        logger.info("🔗 Relationship Agent initialized")
    
    async def initialize(self):
        """Initialize agent components"""
        try:
            await self.kimi_client.initialize()
            logger.info("✅ Relationship Agent fully initialized")
        except Exception as e:
            logger.error(f"❌ Relationship Agent initialization failed: {e}")
    
    async def analyze_relationship(self, paper1: Dict[str, Any], paper2: Dict[str, Any],
                                 analysis_depth: str = 'standard') -> RelationshipResult:
        """
        Analyze relationship between two papers
        
        Args:
            paper1: First paper data
            paper2: Second paper data  
            analysis_depth: Analysis depth ('quick', 'standard', 'deep')
            
        Returns:
            RelationshipResult with analysis
        """
        start_time = asyncio.get_event_loop().time()
        paper1_id = str(paper1.get('id', 'unknown'))
        paper2_id = str(paper2.get('id', 'unknown'))
        pair_id = f"{paper1_id}_{paper2_id}"
        
        try:
            self.stats['total_analyses'] += 1
            
            # Quick similarity check to avoid unnecessary processing
            if not self._should_analyze_pair(paper1, paper2):
                logger.info(f"⏭️ Skipping pair {pair_id} - insufficient similarity")
                return self._create_weak_relationship_result(pair_id, paper1_id, paper2_id)
            
            # Get analysis strategy
            strategy = self.analysis_strategies.get(analysis_depth, self.analysis_strategies['standard'])
            
            # Determine AI backend
            use_kimi = strategy.get('use_kimi', False)
            if use_kimi:
                kimi_prob = strategy.get('kimi_probability', 0.3)
                use_kimi = asyncio.get_event_loop().time() % 1.0 < kimi_prob
            
            # Perform relationship analysis
            if use_kimi:
                result = await self._analyze_with_kimi(paper1, paper2, strategy)
                self.stats['kimi_analyses'] += 1
                agent_used = "kimi_relationship"
            else:
                result = await self._analyze_with_groq(paper1, paper2, strategy)
                self.stats['groq_analyses'] += 1
                agent_used = "groq_relationship"
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Enhance result with metadata
            result.pair_id = pair_id
            result.paper1_id = paper1_id
            result.paper2_id = paper2_id
            result.processing_time = processing_time
            result.agent_used = agent_used
            
            # Update statistics
            self.stats['successful_analyses'] += 1
            if result.relationship_strength >= self.strength_thresholds['strong']:
                self.stats['strong_relationships_found'] += 1
            elif result.relationship_strength < self.strength_thresholds['weak']:
                self.stats['weak_relationships_filtered'] += 1
            
            self._update_average_processing_time(processing_time)
            
            logger.info(f"✅ Relationship {pair_id} analyzed: {result.relationship_type} "
                       f"({result.relationship_strength:.2f}) using {agent_used}")
            
            return result
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            self.stats['failed_analyses'] += 1
            
            logger.error(f"❌ Relationship analysis failed for {pair_id}: {e}")
            
            return RelationshipResult(
                pair_id=pair_id,
                paper1_id=paper1_id,
                paper2_id=paper2_id,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def analyze_relationships_batch(self, papers: List[Dict[str, Any]],
                                        analysis_depth: str = 'standard',
                                        max_pairs: int = 20) -> List[RelationshipResult]:
        """
        Analyze relationships between multiple papers efficiently
        
        Args:
            papers: List of paper data
            analysis_depth: Analysis depth
            max_pairs: Maximum number of pairs to analyze
            
        Returns:
            List of relationship results
        """
        try:
            if len(papers) < 2:
                return []
            
            # Generate strategic paper pairs
            paper_pairs = self._generate_strategic_pairs(papers, max_pairs)
            
            if not paper_pairs:
                return []
            
            # Get concurrency limit from strategy
            strategy = self.analysis_strategies.get(analysis_depth, self.analysis_strategies['standard'])
            max_concurrent = strategy.get('max_concurrent', 6)
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def analyze_pair(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> RelationshipResult:
                async with semaphore:
                    return await self.analyze_relationship(pair[0], pair[1], analysis_depth)
            
            # Execute analyses
            logger.info(f"🔗 Starting batch relationship analysis: {len(paper_pairs)} pairs, "
                       f"max_concurrent={max_concurrent}")
            
            analysis_tasks = [analyze_pair(pair) for pair in paper_pairs]
            
            # Execute with timeout
            timeout = strategy['timeout'] * 2 * len(paper_pairs) / max_concurrent
            results = await asyncio.wait_for(
                asyncio.gather(*analysis_tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Process results
            valid_results = []
            for result in results:
                if isinstance(result, RelationshipResult):
                    # Only keep relationships above threshold
                    if result.success and result.relationship_strength >= self.strength_thresholds['weak']:
                        valid_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Batch relationship analysis error: {result}")
            
            successful = len(valid_results)
            strong_relationships = sum(1 for r in valid_results 
                                     if r.relationship_strength >= self.strength_thresholds['strong'])
            
            logger.info(f"📊 Batch relationship analysis completed: {successful}/{len(paper_pairs)} successful, "
                       f"{strong_relationships} strong relationships")
            
            # Sort by relationship strength
            valid_results.sort(key=lambda x: x.relationship_strength, reverse=True)
            
            return valid_results
            
        except asyncio.TimeoutError:
            logger.error("⏰ Batch relationship analysis timed out")
            return []
        except Exception as e:
            logger.error(f"❌ Batch relationship analysis failed: {e}")
            return []
    
    def _should_analyze_pair(self, paper1: Dict[str, Any], paper2: Dict[str, Any]) -> bool:
        """Quick check if pair is worth analyzing"""
        try:
            # Check if papers are in same domain
            domain1 = paper1.get('research_domain', '').lower()
            domain2 = paper2.get('research_domain', '').lower()
            
            # Always analyze if domains match
            if domain1 == domain2 and domain1:
                return True
            
            # Check for keyword overlap in titles
            title1_words = set(paper1.get('title', '').lower().split())
            title2_words = set(paper2.get('title', '').lower().split())
            
            # Filter out common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            title1_words = title1_words - common_words
            title2_words = title2_words - common_words
            
            # Check for word overlap
            if title1_words and title2_words:
                overlap = title1_words.intersection(title2_words)
                overlap_ratio = len(overlap) / min(len(title1_words), len(title2_words))
                
                if overlap_ratio >= 0.2:  # 20% word overlap
                    return True
            
            # Check abstract similarity (basic)
            abstract1 = paper1.get('abstract', '').lower()
            abstract2 = paper2.get('abstract', '').lower()
            
            if abstract1 and abstract2:
                # Simple keyword matching
                keywords1 = set(abstract1.split()[:20])  # First 20 words
                keywords2 = set(abstract2.split()[:20])
                
                overlap = keywords1.intersection(keywords2) - common_words
                if len(overlap) >= 3:  # At least 3 common keywords
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Pair filtering failed: {e}")
            return True  # Analyze by default if filtering fails
    
    def _generate_strategic_pairs(self, papers: List[Dict[str, Any]], 
                                max_pairs: int) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Generate strategic paper pairs for analysis"""
        try:
            # Sort papers by quality score to prioritize high-quality pairs
            sorted_papers = sorted(
                papers, 
                key=lambda p: p.get('context_quality_score', 0.5), 
                reverse=True
            )
            
            # Limit papers to manage computational complexity
            limited_papers = sorted_papers[:min(15, len(sorted_papers))]
            
            pairs = []
            
            # Strategy 1: High-quality papers with each other
            high_quality_papers = [p for p in limited_papers 
                                 if p.get('context_quality_score', 0.5) > 0.7]
            
            if len(high_quality_papers) >= 2:
                for pair in itertools.combinations(high_quality_papers[:8], 2):
                    pairs.append(pair)
            
            # Strategy 2: Same domain papers
            domain_groups = {}
            for paper in limited_papers:
                domain = paper.get('research_domain', 'Unknown')
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(paper)
            
            for domain, domain_papers in domain_groups.items():
                if len(domain_papers) >= 2:
                    # Add pairs within domain
                    domain_pairs = list(itertools.combinations(domain_papers[:5], 2))
                    pairs.extend(domain_pairs[:3])  # Limit to 3 pairs per domain
            
            # Strategy 3: Cross-domain high-potential pairs
            if len(limited_papers) > len(high_quality_papers):
                remaining_papers = [p for p in limited_papers if p not in high_quality_papers]
                for hq_paper in high_quality_papers[:3]:
                    for other_paper in remaining_papers[:3]:
                        pairs.append((hq_paper, other_paper))
            
            # Remove duplicates and limit
            unique_pairs = []
            seen_pairs = set()
            
            for pair in pairs:
                paper1_id = str(pair[0].get('id', ''))
                paper2_id = str(pair[1].get('id', ''))
                
                # Normalize pair order
                pair_key = tuple(sorted([paper1_id, paper2_id]))
                
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    unique_pairs.append(pair)
                    
                    if len(unique_pairs) >= max_pairs:
                        break
            
            logger.info(f"🎯 Generated {len(unique_pairs)} strategic paper pairs")
            return unique_pairs
            
        except Exception as e:
            logger.error(f"❌ Strategic pair generation failed: {e}")
            # Fallback to simple pairing
            return list(itertools.combinations(papers[:10], 2))[:max_pairs]
    
    async def _analyze_with_groq(self, paper1: Dict[str, Any], paper2: Dict[str, Any],
                               strategy: Dict[str, Any]) -> RelationshipResult:
        """Analyze relationship using Groq processor"""
        try:
            groq_result = await asyncio.wait_for(
                self.groq_processor.analyze_paper_relationship(paper1, paper2),
                timeout=strategy['timeout']
            )
            
            return RelationshipResult(
                pair_id="",  # Will be set by caller
                paper1_id="",  # Will be set by caller
                paper2_id="",  # Will be set by caller
                success=True,
                relationship_type=groq_result.relationship_type,
                relationship_strength=groq_result.relationship_strength,
                relationship_context=groq_result.relationship_context,
                connection_reasoning=groq_result.connection_reasoning,
                confidence_score=groq_result.confidence_score,
                agent_used="groq_relationship"
            )
            
        except asyncio.TimeoutError:
            logger.warning("Groq relationship analysis timed out")
            raise
        except Exception as e:
            logger.error(f"Groq relationship analysis failed: {e}")
            raise
    
    async def _analyze_with_kimi(self, paper1: Dict[str, Any], paper2: Dict[str, Any],
                               strategy: Dict[str, Any]) -> RelationshipResult:
        """Analyze relationship using Kimi client"""
        try:
            kimi_result = await asyncio.wait_for(
                self.kimi_client.analyze_relationship_detailed(paper1, paper2),
                timeout=strategy['timeout']
            )
            
            # Extract relationship data from Kimi result
            rel_analysis = kimi_result.content.get('relationship_analysis', {})
            
            # Map Kimi analysis to standard format
            relationship_type = self._map_kimi_relationship_type(rel_analysis)
            relationship_strength = float(rel_analysis.get('semantic_similarity', 0.5))
            
            return RelationshipResult(
                pair_id="",  # Will be set by caller
                paper1_id="",  # Will be set by caller
                paper2_id="",  # Will be set by caller
                success=True,
                relationship_type=relationship_type,
                relationship_strength=relationship_strength,
                relationship_context=rel_analysis.get('thematic_connection', '')[:200],
                connection_reasoning=rel_analysis.get('citation_potential', '')[:300],
                confidence_score=kimi_result.confidence_score,
                agent_used="kimi_relationship"
            )
            
        except asyncio.TimeoutError:
            logger.warning("Kimi relationship analysis timed out")
            raise
        except Exception as e:
            logger.error(f"Kimi relationship analysis failed: {e}")
            raise
    
    def _map_kimi_relationship_type(self, rel_analysis: Dict[str, Any]) -> str:
        """Map Kimi relationship analysis to standard relationship types"""
        try:
            # Analyze the relationship context to determine type
            context = ' '.join(str(v) for v in rel_analysis.values()).lower()
            
            if 'build' in context or 'extend' in context:
                return 'builds_upon'
            elif 'improve' in context or 'enhance' in context:
                return 'improves'
            elif 'complement' in context or 'support' in context:
                return 'complements'
            elif 'similar' in context or 'related' in context:
                return 'related'
            elif 'different' in context or 'contrast' in context:
                return 'contradicts'
            else:
                return 'related'
                
        except Exception:
            return 'related'
    
    def _create_weak_relationship_result(self, pair_id: str, paper1_id: str, 
                                       paper2_id: str) -> RelationshipResult:
        """Create result for weak/filtered relationships"""
        return RelationshipResult(
            pair_id=pair_id,
            paper1_id=paper1_id,
            paper2_id=paper2_id,
            success=True,
            relationship_type="unrelated",
            relationship_strength=0.1,
            relationship_context="Papers appear to be in different research areas",
            connection_reasoning="Filtered out due to low similarity",
            confidence_score=0.8,
            agent_used="filter"
        )
    
    def _update_average_processing_time(self, new_time: float):
        """Update average processing time"""
        if self.stats['successful_analyses'] == 1:
            self.stats['average_processing_time'] = new_time
        else:
            current_avg = self.stats['average_processing_time']
            count = self.stats['successful_analyses']
            self.stats['average_processing_time'] = (current_avg * (count - 1) + new_time) / count
    
    def get_relationship_insights(self, results: List[RelationshipResult]) -> Dict[str, Any]:
        """Get insights from relationship analysis results"""
        try:
            if not results:
                return {}
            
            # Relationship type distribution
            type_counts = {}
            strength_distribution = {'strong': 0, 'moderate': 0, 'weak': 0}
            
            for result in results:
                if result.success:
                    rel_type = result.relationship_type
                    type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
                    
                    strength = result.relationship_strength
                    if strength >= self.strength_thresholds['strong']:
                        strength_distribution['strong'] += 1
                    elif strength >= self.strength_thresholds['moderate']:
                        strength_distribution['moderate'] += 1
                    else:
                        strength_distribution['weak'] += 1
            
            # Find strongest relationships
            strongest_relationships = sorted(
                [r for r in results if r.success],
                key=lambda x: x.relationship_strength,
                reverse=True
            )[:5]
            
            return {
                'relationship_type_distribution': type_counts,
                'strength_distribution': strength_distribution,
                'strongest_relationships': [
                    {
                        'papers': f"{r.paper1_id} - {r.paper2_id}",
                        'type': r.relationship_type,
                        'strength': r.relationship_strength,
                        'reasoning': r.connection_reasoning[:100]
                    }
                    for r in strongest_relationships
                ],
                'average_relationship_strength': sum(r.relationship_strength for r in results if r.success) / len([r for r in results if r.success]) if results else 0,
                'total_relationships_analyzed': len(results),
                'successful_analyses': len([r for r in results if r.success])
            }
            
        except Exception as e:
            logger.error(f"❌ Relationship insights generation failed: {e}")
            return {}
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        try:
            success_rate = (
                self.stats['successful_analyses'] / max(self.stats['total_analyses'], 1) * 100
            )
            
            return {
                **self.stats,
                'success_rate_percent': success_rate,
                'strong_relationship_rate': (
                    self.stats['strong_relationships_found'] / 
                    max(self.stats['successful_analyses'], 1) * 100
                ),
                'filtering_effectiveness': (
                    self.stats['weak_relationships_filtered'] / 
                    max(self.stats['total_analyses'], 1) * 100
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Stats calculation failed: {e}")
            return self.stats
    
    async def shutdown(self):
        """Shutdown agent gracefully"""
        try:
            await self.kimi_client.close()
            logger.info("🔒 Relationship Agent shutdown completed")
        except Exception as e:
            logger.error(f"❌ Relationship Agent shutdown failed: {e}")
