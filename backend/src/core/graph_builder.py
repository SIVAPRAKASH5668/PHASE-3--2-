import logging
from typing import List, Dict, Any, Set, Tuple, Optional
import networkx as nx
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import colorsys

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Enhanced graph node representing a paper with multilingual support"""
    id: str
    label: str
    title: str
    research_domain: str
    context_summary: str
    methodology: str
    innovations: List[str]
    contributions: List[str]
    quality_score: float
    size: int
    color: str
    # Enhanced multilingual fields
    language: str = "en"
    search_language: str = "unknown"
    ai_agent_used: str = "unknown"
    search_method: str = "traditional"
    analysis_confidence: float = 0.8
    embedding_similarity: float = 0.0
    citation_potential: float = 0.5

@dataclass
class GraphEdge:
    """Enhanced graph edge with AI agent information"""
    source: str
    target: str
    relationship_type: str
    strength: float
    context: str
    reasoning: str
    weight: float
    # Enhanced AI fields
    ai_agent_used: str = "unknown"
    confidence_score: float = 0.7
    analysis_type: str = "standard"
    semantic_similarity: float = 0.5

class EnhancedIntelligentGraphBuilder:
    """
    Enhanced intelligent graph builder with multilingual and AI agent support
    ✅ FIXED: All method signatures and error handling corrected
    """
    
    def __init__(self):
        # Enhanced domain colors with gradients
        self.domain_colors = {
            "Computer Vision": "#FF6B6B",
            "Natural Language Processing": "#4ECDC4", 
            "Machine Learning": "#45B7D1",
            "Deep Learning": "#96CEB4",
            "Artificial Intelligence": "#A8E6CF",
            "Healthcare": "#FFEAA7",
            "Robotics": "#DDA0DD",
            "Data Science": "#98D8C8",
            "Bioinformatics": "#FFB6C1",
            "Computer Graphics": "#F0E68C",
            "General Research": "#F7DC6F",
            "Unknown": "#D3D3D3"
        }
        
        # Enhanced relationship weights with AI considerations
        self.relationship_weights = {
            "builds_upon": 0.95,
            "improves": 0.85,
            "extends": 0.75,
            "complements": 0.65,
            "applies": 0.55,
            "related": 0.45,
            "contradicts": 0.35,
            "unrelated": 0.15,
            "methodology_shared": 0.70,
            "domain_overlap": 0.60,
            "competing": 0.50
        }
        
        # AI agent quality weights
        self.agent_quality_weights = {
            "kimi_analysis": 1.2,
            "kimi_context": 1.15,
            "groq_detailed": 1.1,
            "groq_fast": 1.0,
            "fallback": 0.8
        }
        
        # Language priorities for visualization
        self.language_priorities = {
            'en': 1.0, 'zh': 0.9, 'de': 0.85, 'fr': 0.8, 'ja': 0.75, 
            'ko': 0.7, 'es': 0.75, 'ru': 0.7, 'it': 0.65, 'pt': 0.6
        }
        
        logger.info("🎨 Enhanced intelligent graph builder initialized")
    
    def build_graph(self, papers: List[Dict[str, Any]], relationships: List[Dict[str, Any]], 
                   multilingual_keywords: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Build enhanced graph with multilingual and AI agent support
        ✅ FIXED: Proper error handling and fallback mechanisms
        """
        try:
            # Create enhanced nodes
            nodes = []
            node_map = {}
            
            for paper in papers:
                node = self._create_enhanced_node(paper)
                nodes.append(node.__dict__)
                node_map[str(paper.get("id", ""))] = node
            
            # Create enhanced edges
            edges = []
            for rel in relationships:
                paper1_id = str(rel.get("paper1_id", ""))
                paper2_id = str(rel.get("paper2_id", ""))
                
                if (paper1_id in node_map and paper2_id in node_map and 
                    rel.get("relationship_strength", 0) > 0.3):
                    edge = self._create_enhanced_edge(rel)
                    edges.append(edge.__dict__)
            
            # ✅ FIXED: Calculate enhanced metrics with correct arguments
            metrics = self._calculate_enhanced_metrics(nodes, edges, papers, relationships)
            
            # Create multilingual clusters
            clusters = self._create_multilingual_clusters(papers)
            
            # Generate AI-enhanced insights
            insights = self._generate_ai_insights(papers, relationships, metrics, multilingual_keywords)
            
            # Create layout suggestions with AI considerations
            layout_suggestions = self._suggest_ai_layout(nodes, edges, papers)
            
            # Generate visualization config
            vis_config = self._generate_visualization_config(nodes, edges, clusters)
            
            graph_data = {
                "nodes": nodes,
                "edges": edges,
                "metrics": metrics,
                "clusters": clusters,
                "insights": insights,
                "layout_suggestions": layout_suggestions,
                "visualization_config": vis_config,
                "metadata": {
                    "total_papers": len(papers),
                    "total_relationships": len(relationships),
                    "languages_present": list(set(p.get('language', 'unknown') for p in papers)),
                    "ai_agents_used": list(set(p.get('ai_agent_used', 'unknown') for p in papers)),
                    "generation_timestamp": datetime.now().isoformat(),
                    "graph_type": "enhanced_multilingual_ai"
                }
            }
            
            logger.info(f"✅ Built enhanced graph: {len(nodes)} nodes, {len(edges)} edges, {len(clusters)} clusters")
            return graph_data
            
        except Exception as e:
            logger.error(f"❌ Enhanced graph building failed: {e}")
            # Fallback to basic graph
            return self._build_fallback_graph(papers, relationships)
    
    def _create_enhanced_node(self, paper: Dict[str, Any]) -> GraphNode:
        """Create enhanced graph node with multilingual and AI features"""
        try:
            # Calculate enhanced size based on multiple factors
            base_size = 40
            quality_score = paper.get("context_quality_score", 0.5)
            analysis_confidence = paper.get("analysis_confidence", 0.8)
            ai_agent_used = paper.get("ai_agent_used", "unknown")
            
            # Size calculation with AI agent quality weighting
            agent_weight = self.agent_quality_weights.get(ai_agent_used, 1.0)
            quality_bonus = int(quality_score * analysis_confidence * agent_weight * 40)
            size = max(30, min(100, base_size + quality_bonus))
            
            # Enhanced domain detection and color
            domain = paper.get("research_domain", "General Research")
            if domain not in self.domain_colors:
                domain = "General Research"
            
            # Color enhancement based on confidence and language
            base_color = self.domain_colors[domain]
            language = paper.get("language", "en")
            lang_priority = self.language_priorities.get(language, 0.5)
            
            # Adjust color brightness based on confidence and language priority
            enhanced_color = self._enhance_color_by_confidence(
                base_color, analysis_confidence, lang_priority
            )
            
            return GraphNode(
                id=str(paper.get("id", "")),
                label=self._truncate_title(paper.get("title", "Untitled"), 60),
                title=paper.get("title", "Untitled"),
                research_domain=domain,
                context_summary=paper.get("context_summary", "")[:300],
                methodology=paper.get("methodology", "")[:200],
                innovations=paper.get("innovations", [])[:3],
                contributions=paper.get("contributions", [])[:3],
                quality_score=quality_score,
                size=size,
                color=enhanced_color,
                language=language,
                search_language=paper.get("search_language", "unknown"),
                ai_agent_used=ai_agent_used,
                search_method=paper.get("search_method", "traditional"),
                analysis_confidence=analysis_confidence,
                embedding_similarity=paper.get("similarity_score", 0.0),
                citation_potential=min(1.0, quality_score * analysis_confidence)
            )
            
        except Exception as e:
            logger.error(f"❌ Node creation failed: {e}")
            return self._create_fallback_node(paper)
    
    def _create_enhanced_edge(self, relationship: Dict[str, Any]) -> GraphEdge:
        """Create enhanced edge with AI agent metadata"""
        try:
            rel_type = relationship.get("relationship_type", "related")
            strength = relationship.get("relationship_strength", 0.5)
            ai_agent_used = relationship.get("ai_agent_used", "unknown")
            confidence_score = relationship.get("confidence_score", 0.7)
            
            # Calculate enhanced weight
            base_weight = self.relationship_weights.get(rel_type, 0.5)
            agent_weight = self.agent_quality_weights.get(ai_agent_used, 1.0)
            final_weight = base_weight * strength * confidence_score * agent_weight
            
            return GraphEdge(
                source=str(relationship.get("paper1_id", "")),
                target=str(relationship.get("paper2_id", "")),
                relationship_type=rel_type,
                strength=strength,
                context=relationship.get("relationship_context", "")[:200],
                reasoning=relationship.get("connection_reasoning", "")[:300],
                weight=max(0.1, min(1.0, final_weight)),
                ai_agent_used=ai_agent_used,
                confidence_score=confidence_score,
                analysis_type=relationship.get("analysis_type", "standard"),
                semantic_similarity=relationship.get("semantic_similarity", 0.5)
            )
            
        except Exception as e:
            logger.error(f"❌ Edge creation failed: {e}")
            return self._create_fallback_edge(relationship)
    
    def _calculate_enhanced_metrics(self, nodes: List[Dict], edges: List[Dict], 
                                  papers: List[Dict] = None, relationships: List[Dict] = None, 
                                  *args, **kwargs) -> Dict[str, Any]:
        """
        ✅ FIXED: Calculate comprehensive graph metrics with correct method signature
        Now accepts: nodes, edges, papers, relationships + optional args
        """
        try:
            # ✅ FIXED: Build NetworkX graph from nodes and edges lists
            graph = nx.Graph()
            
            # Add nodes to graph
            for node in nodes:
                node_id = node.get('id') if isinstance(node, dict) else str(node)
                if node_id:
                    graph.add_node(node_id)
            
            # Add edges to graph
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get('source')
                    target = edge.get('target')
                    weight = edge.get('weight', 1.0)
                    if source and target:
                        graph.add_edge(source, target, weight=weight)
            
            # Initialize papers and relationships if not provided
            if papers is None:
                papers = []
            if relationships is None:
                relationships = []
            
            metrics = {
                'basic_metrics': {},
                'centrality_metrics': {},
                'clustering_metrics': {},
                'research_metrics': {},
                'ai_metrics': {},
                'language_metrics': {},
                'quality_metrics': {}
            }
            
            # ✅ FIXED: Check for empty graph
            if graph.number_of_nodes() == 0:
                logger.warning("📊 Empty graph - returning default metrics")
                return self._get_default_metrics()
            
            # Basic metrics with safe calculations
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()
            
            metrics['basic_metrics'] = {
                'node_count': node_count,
                'edge_count': edge_count,
                'density': edge_count / max(node_count * (node_count - 1) / 2, 1) if node_count > 1 else 0.0,
                'average_degree': (2 * edge_count) / max(node_count, 1),
                'connectivity': 'connected' if nx.is_connected(graph) else 'disconnected' if node_count > 0 else 'empty'
            }
            
            # Centrality metrics (only if graph has nodes)
            if node_count > 0:
                try:
                    centrality = nx.degree_centrality(graph)
                    betweenness = nx.betweenness_centrality(graph)
                    closeness = nx.closeness_centrality(graph)
                    
                    metrics['centrality_metrics'] = {
                        'max_degree_centrality': max(centrality.values()) if centrality else 0.0,
                        'avg_betweenness': sum(betweenness.values()) / len(betweenness) if betweenness else 0.0,
                        'avg_closeness': sum(closeness.values()) / len(closeness) if closeness else 0.0
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Centrality calculation failed: {e}")
                    metrics['centrality_metrics'] = {}
            
            # Clustering metrics
            try:
                metrics['clustering_metrics'] = {
                    'average_clustering': nx.average_clustering(graph) if node_count > 0 else 0.0,
                    'transitivity': nx.transitivity(graph) if node_count > 0 else 0.0
                }
            except Exception as e:
                logger.warning(f"⚠️ Clustering calculation failed: {e}")
                metrics['clustering_metrics'] = {'average_clustering': 0.0, 'transitivity': 0.0}
            
            # Research-specific metrics using papers data
            metrics['research_metrics'] = self._calculate_research_metrics(papers)
            
            # AI-specific metrics
            metrics['ai_metrics'] = self._calculate_ai_metrics(papers, relationships)
            
            # Language-specific metrics
            metrics['language_metrics'] = self._calculate_language_metrics(papers)
            
            # Quality metrics
            metrics['quality_metrics'] = self._calculate_quality_metrics(papers, relationships)
            
            # Overall graph health score
            metrics['graph_health_score'] = self._calculate_graph_health(
                metrics['basic_metrics'], 
                metrics['ai_metrics'], 
                metrics['quality_metrics']
            )
            
            logger.info(f"📊 Calculated comprehensive metrics for {node_count} nodes, {edge_count} edges")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Enhanced metrics calculation failed: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics structure when calculation fails"""
        return {
            'basic_metrics': {
                'node_count': 0,
                'edge_count': 0,
                'density': 0.0,
                'average_degree': 0.0,
                'connectivity': 'empty'
            },
            'centrality_metrics': {},
            'clustering_metrics': {
                'average_clustering': 0.0,
                'transitivity': 0.0
            },
            'research_metrics': {
                'domain_diversity': 0.0,
                'language_diversity': 0.0,
                'temporal_span_days': 0
            },
            'ai_metrics': {},
            'language_metrics': {},
            'quality_metrics': {},
            'graph_health_score': 0.0
        }
    
    def _calculate_research_metrics(self, papers: List[Dict]) -> Dict[str, Any]:
        """Calculate research-specific metrics"""
        try:
            domains = set()
            languages = set()
            dates = []
            
            for paper in papers:
                if paper.get('research_domain'):
                    domains.add(paper['research_domain'])
                if paper.get('language'):
                    languages.add(paper['language'])
                if paper.get('published_date'):
                    dates.append(paper['published_date'])
            
            temporal_span = 0
            if len(dates) > 1:
                try:
                    date_objects = [pd.to_datetime(d) for d in dates if d]
                    if date_objects:
                        temporal_span = (max(date_objects) - min(date_objects)).days
                except Exception:
                    temporal_span = 0
            
            return {
                'domain_diversity': len(domains) / max(len(papers), 1),
                'language_diversity': len(languages) / max(len(papers), 1),
                'temporal_span_days': temporal_span,
                'total_domains': len(domains),
                'total_languages': len(languages)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Research metrics calculation failed: {e}")
            return {'domain_diversity': 0.0, 'language_diversity': 0.0, 'temporal_span_days': 0}
    
    def _calculate_ai_metrics(self, papers: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Calculate AI-specific metrics"""
        try:
            # AI agent distribution
            agent_usage = {}
            for paper in papers:
                agent = paper.get("ai_agent_used", "unknown")
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
            
            # Analysis confidence distribution
            confidences = [p.get("analysis_confidence", 0.8) for p in papers]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Relationship analysis quality
            rel_confidences = [r.get("confidence_score", 0.7) for r in relationships]
            avg_rel_confidence = sum(rel_confidences) / len(rel_confidences) if rel_confidences else 0
            
            return {
                "agent_usage_distribution": agent_usage,
                "average_analysis_confidence": avg_confidence,
                "average_relationship_confidence": avg_rel_confidence,
                "high_confidence_papers": sum(1 for c in confidences if c > 0.8),
                "ai_enhancement_score": (avg_confidence + avg_rel_confidence) / 2
            }
            
        except Exception as e:
            logger.error(f"❌ AI metrics calculation failed: {e}")
            return {}
    
    def _calculate_language_metrics(self, papers: List[Dict]) -> Dict[str, Any]:
        """Calculate multilingual metrics"""
        try:
            # Language distribution
            languages = {}
            search_languages = {}
            
            for paper in papers:
                lang = paper.get("language", "unknown")
                search_lang = paper.get("search_language", "unknown")
                
                languages[lang] = languages.get(lang, 0) + 1
                search_languages[search_lang] = search_languages.get(search_lang, 0) + 1
            
            # Language diversity score
            unique_languages = len([l for l in languages.keys() if l != "unknown"])
            diversity_score = min(1.0, unique_languages / 5)  # Normalize to max 5 languages
            
            return {
                "paper_languages": languages,
                "search_languages": search_languages,
                "language_diversity_score": diversity_score,
                "multilingual_coverage": unique_languages,
                "dominant_language": max(languages.keys(), key=languages.get) if languages else "unknown"
            }
            
        except Exception as e:
            logger.error(f"❌ Language metrics calculation failed: {e}")
            return {}
    
    def _calculate_quality_metrics(self, papers: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Calculate quality and confidence metrics"""
        try:
            # Quality score distribution
            quality_scores = [p.get("context_quality_score", 0.5) for p in papers]
            
            # Relationship strength distribution
            rel_strengths = [r.get("relationship_strength", 0.5) for r in relationships]
            
            return {
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "quality_distribution": {
                    "high_quality": sum(1 for q in quality_scores if q > 0.8),
                    "medium_quality": sum(1 for q in quality_scores if 0.5 <= q <= 0.8),
                    "low_quality": sum(1 for q in quality_scores if q < 0.5)
                },
                "average_relationship_strength": sum(rel_strengths) / len(rel_strengths) if rel_strengths else 0,
                "strong_relationships": sum(1 for s in rel_strengths if s > 0.7),
                "weak_relationships": sum(1 for s in rel_strengths if s < 0.4)
            }
            
        except Exception as e:
            logger.error(f"❌ Quality metrics calculation failed: {e}")
            return {}
    
    def _calculate_graph_health(self, basic_metrics: Dict, ai_metrics: Dict, 
                              quality_metrics: Dict) -> float:
        """Calculate overall graph health score"""
        try:
            # Components of health score
            connectivity_score = min(1.0, basic_metrics.get("density", 0) * 2)
            quality_score = quality_metrics.get("average_quality_score", 0.5)
            ai_enhancement_score = ai_metrics.get("ai_enhancement_score", 0.7)
            
            # Weighted health score
            health_score = (
                connectivity_score * 0.3 +
                quality_score * 0.4 +
                ai_enhancement_score * 0.3
            )
            
            return max(0.0, min(1.0, health_score))
            
        except Exception:
            return 0.5  # Default health score
    
    def _create_multilingual_clusters(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create enhanced clusters considering multiple dimensions"""
        try:
            # Domain-based clusters
            domain_clusters = {}
            language_clusters = {}
            agent_clusters = {}
            method_clusters = {}
            
            for paper in papers:
                paper_id = str(paper.get("id", ""))
                
                # Domain clustering
                domain = paper.get("research_domain", "General Research")
                if domain not in domain_clusters:
                    domain_clusters[domain] = []
                domain_clusters[domain].append(paper_id)
                
                # Language clustering
                language = paper.get("language", "unknown")
                if language not in language_clusters:
                    language_clusters[language] = []
                language_clusters[language].append(paper_id)
                
                # AI agent clustering
                agent = paper.get("ai_agent_used", "unknown")
                if agent not in agent_clusters:
                    agent_clusters[agent] = []
                agent_clusters[agent].append(paper_id)
                
                # Search method clustering
                method = paper.get("search_method", "traditional")
                if method not in method_clusters:
                    method_clusters[method] = []
                method_clusters[method].append(paper_id)
            
            return {
                "domain_clusters": domain_clusters,
                "language_clusters": language_clusters,
                "ai_agent_clusters": agent_clusters,
                "search_method_clusters": method_clusters,
                "cluster_summary": {
                    "total_domains": len(domain_clusters),
                    "total_languages": len(language_clusters),
                    "total_ai_agents": len(agent_clusters),
                    "total_search_methods": len(method_clusters)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Multilingual clustering failed: {e}")
            return {}
    
    def _generate_ai_insights(self, papers: List[Dict], relationships: List[Dict], 
                            metrics: Dict, multilingual_keywords: Dict = None) -> Dict[str, Any]:
        """Generate AI-enhanced insights"""
        try:
            return {
                "key_findings": self._extract_key_findings(papers, relationships),
                "recommendations": self._generate_recommendations(papers, relationships, metrics),
                "research_gaps": self._identify_research_gaps(papers, relationships),
                "cross_domain_opportunities": self._find_cross_domain_opportunities(papers),
                "language_effectiveness": self._calculate_language_effectiveness(papers),
                "ai_agent_performance": self._analyze_agent_performance(papers, relationships)
            }
            
        except Exception as e:
            logger.error(f"❌ AI insights generation failed: {e}")
            return {"error": str(e)}
    
    def _extract_key_findings(self, papers: List[Dict], relationships: List[Dict]) -> List[str]:
        """Extract key research findings with safe calculations"""
        try:
            findings = []
            
            if not papers and not relationships:
                return ["No data available for analysis"]
            
            # Domain analysis
            if papers:
                domains = {}
                for paper in papers:
                    domain = paper.get('research_domain', 'Unknown')
                    domains[domain] = domains.get(domain, 0) + 1
                
                if domains:
                    most_common_domain = max(domains, key=domains.get)
                    findings.append(f"Primary research focus: {most_common_domain} ({domains[most_common_domain]} papers)")
            
            # Quality analysis
            if papers:
                quality_scores = [p.get('context_quality_score', 0.5) for p in papers if p.get('context_quality_score')]
                if quality_scores:
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    findings.append(f"Average research quality: {avg_quality:.2f}/1.0")
            
            # Relationship analysis
            if relationships:
                rel_types = {}
                for rel in relationships:
                    rel_type = rel.get('relationship_type', 'unknown')
                    rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                
                if rel_types:
                    most_common_rel = max(rel_types, key=rel_types.get)
                    findings.append(f"Primary connection type: {most_common_rel}")
            
            return findings[:5]  # Return top 5 findings
            
        except Exception as e:
            logger.error(f"❌ Key findings extraction failed: {e}")
            return ["Unable to extract findings due to analysis error"]
    
    def _generate_recommendations(self, papers: List[Dict], relationships: List[Dict], 
                                metrics: Dict) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Quality recommendations
            quality_metrics = metrics.get("quality_metrics", {})
            avg_quality = quality_metrics.get("average_quality_score", 0.5)
            if avg_quality < 0.7:
                recommendations.append("Consider refining search criteria to improve paper quality")
            
            # Connection recommendations  
            basic_metrics = metrics.get("basic_metrics", {})
            density = basic_metrics.get("density", 0)
            if density < 0.2:
                recommendations.append("Expand search scope to find more interconnected papers")
            
            # Language recommendations
            lang_metrics = metrics.get("language_metrics", {})
            lang_diversity = lang_metrics.get("language_diversity_score", 0)
            if lang_diversity < 0.4:
                recommendations.append("Consider searching in additional languages")
            
            return recommendations[:5]
            
        except Exception as e:
            logger.error(f"❌ Recommendations generation failed: {e}")
            return ["Unable to generate recommendations"]
    
    def _identify_research_gaps(self, papers: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Identify potential research gaps"""
        try:
            # Find isolated papers
            paper_connections = {}
            for rel in relationships:
                paper1_id = str(rel.get("paper1_id", ""))
                paper2_id = str(rel.get("paper2_id", ""))
                
                paper_connections[paper1_id] = paper_connections.get(paper1_id, 0) + 1
                paper_connections[paper2_id] = paper_connections.get(paper2_id, 0) + 1
            
            isolated_papers = [
                paper_id for paper_id, connections in paper_connections.items()
                if connections <= 1
            ]
            
            return {
                "isolated_papers": isolated_papers[:5],
                "research_gap_score": len(isolated_papers) / len(papers) if papers else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Research gaps identification failed: {e}")
            return {}
    
    def _find_cross_domain_opportunities(self, papers: List[Dict]) -> List[Dict]:
        """Find cross-domain research opportunities"""
        try:
            domain_methods = {}
            
            for paper in papers:
                domain = paper.get("research_domain", "Unknown")
                methodology = paper.get("methodology", "")
                
                if domain not in domain_methods:
                    domain_methods[domain] = set()
                
                if methodology:
                    domain_methods[domain].add(methodology)
            
            opportunities = []
            domains = list(domain_methods.keys())
            
            for i, domain1 in enumerate(domains):
                for domain2 in domains[i+1:]:
                    common_methods = domain_methods[domain1].intersection(domain_methods[domain2])
                    if common_methods:
                        opportunities.append({
                            "domain1": domain1,
                            "domain2": domain2,
                            "shared_methodologies": list(common_methods),
                            "opportunity_score": len(common_methods)
                        })
            
            return sorted(opportunities, key=lambda x: x["opportunity_score"], reverse=True)[:5]
            
        except Exception as e:
            logger.error(f"❌ Cross-domain opportunities analysis failed: {e}")
            return []
    
    def _calculate_language_effectiveness(self, papers: List[Dict]) -> Dict[str, float]:
        """Calculate effectiveness of different search languages"""
        try:
            lang_quality = {}
            for paper in papers:
                search_lang = paper.get("search_language", "unknown")
                quality = paper.get("context_quality_score", 0.5)
                
                if search_lang not in lang_quality:
                    lang_quality[search_lang] = {"total": 0, "count": 0}
                
                lang_quality[search_lang]["total"] += quality
                lang_quality[search_lang]["count"] += 1
            
            effectiveness = {}
            for lang, data in lang_quality.items():
                effectiveness[lang] = data["total"] / data["count"] if data["count"] > 0 else 0
            
            return effectiveness
            
        except Exception as e:
            logger.error(f"❌ Language effectiveness calculation failed: {e}")
            return {}
    
    def _analyze_agent_performance(self, papers: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Analyze AI agent performance"""
        try:
            agent_performance = {}
            
            for paper in papers:
                agent = paper.get("ai_agent_used", "unknown")
                quality = paper.get("context_quality_score", 0.5)
                confidence = paper.get("analysis_confidence", 0.8)
                
                if agent not in agent_performance:
                    agent_performance[agent] = {
                        "count": 0,
                        "total_quality": 0,
                        "total_confidence": 0
                    }
                
                agent_performance[agent]["count"] += 1
                agent_performance[agent]["total_quality"] += quality
                agent_performance[agent]["total_confidence"] += confidence
            
            # Calculate averages
            performance_summary = {}
            for agent, data in agent_performance.items():
                if data["count"] > 0:
                    performance_summary[agent] = {
                        "average_quality": data["total_quality"] / data["count"],
                        "average_confidence": data["total_confidence"] / data["count"],
                        "papers_processed": data["count"]
                    }
            
            return performance_summary
            
        except Exception as e:
            logger.error(f"❌ Agent performance analysis failed: {e}")
            return {}
    
    def _suggest_ai_layout(self, nodes: List[Dict], edges: List[Dict], 
                          papers: List[Dict]) -> Dict[str, Any]:
        """Suggest layout optimized for AI-enhanced graphs"""
        try:
            node_count = len(nodes)
            edge_count = len(edges)
            
            # Calculate graph density
            max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 1
            density = edge_count / max_edges
            
            # Suggest layout based on characteristics
            if node_count < 15:
                recommended_layout = "spring"
            elif node_count < 40 and density > 0.3:
                recommended_layout = "fruchterman_reingold" 
            elif node_count < 100:
                recommended_layout = "kamada_kawai"
            else:
                recommended_layout = "spring"
            
            return {
                "recommended_layout": recommended_layout,
                "show_edge_labels": edge_count < 30,
                "node_size_scaling": True,
                "edge_weight_scaling": True,
                "ai_enhancements": {
                    "highlight_high_confidence": True,
                    "show_ai_agent_badges": node_count < 50,
                    "size_by_quality": True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ AI layout suggestion failed: {e}")
            return {"recommended_layout": "spring", "error": str(e)}
    
    def _generate_visualization_config(self, nodes: List[Dict], edges: List[Dict], 
                                     clusters: Dict) -> Dict[str, Any]:
        """Generate comprehensive visualization configuration"""
        try:
            return {
                "node_config": {
                    "min_size": 20,
                    "max_size": 80,
                    "label_threshold": 50,
                    "quality_opacity_mapping": True
                },
                "edge_config": {
                    "min_width": 1,
                    "max_width": 8,
                    "opacity_by_confidence": True,
                    "color_by_type": True
                },
                "cluster_config": {
                    "show_domain_boundaries": len(clusters.get("domain_clusters", {})) > 1,
                    "show_language_indicators": len(clusters.get("language_clusters", {})) > 2,
                    "cluster_spacing": 100
                },
                "performance_config": {
                    "enable_webgl": len(nodes) > 100,
                    "use_octree": len(nodes) > 1000,
                    "level_of_detail": len(nodes) > 500
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Visualization config generation failed: {e}")
            return {}
    
    def _enhance_color_by_confidence(self, base_color: str, confidence: float, 
                                   lang_priority: float) -> str:
        """Enhance color based on confidence and language priority"""
        try:
            # Convert hex to RGB
            base_color = base_color.lstrip('#')
            rgb = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))
            
            # Convert to HSV for brightness adjustment
            hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            
            # Adjust brightness based on confidence and language priority
            brightness_factor = 0.7 + (confidence * lang_priority * 0.3)
            new_hsv = (hsv[0], hsv[1], min(1.0, hsv[2] * brightness_factor))
            
            # Convert back to RGB and hex
            new_rgb = colorsys.hsv_to_rgb(*new_hsv)
            new_hex = '#' + ''.join(f'{int(c*255):02x}' for c in new_rgb)
            
            return new_hex
            
        except Exception:
            return base_color  # Return original color if enhancement fails
    
    def _truncate_title(self, title: str, max_length: int = 50) -> str:
        """Truncate title for display"""
        if len(title) <= max_length:
            return title
        return title[:max_length-3] + "..."
    
    # Helper methods for fallback
    def _build_fallback_graph(self, papers: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Build basic fallback graph when enhanced building fails"""
        try:
            nodes = []
            for i, paper in enumerate(papers):
                nodes.append({
                    "id": str(paper.get("id", i)),
                    "label": paper.get("title", "Untitled")[:50],
                    "title": paper.get("title", "Untitled"),
                    "size": 40,
                    "color": "#F7DC6F"
                })
            
            edges = []
            for rel in relationships:
                if rel.get("relationship_strength", 0) > 0.3:
                    edges.append({
                        "source": str(rel.get("paper1_id", "")),
                        "target": str(rel.get("paper2_id", "")),
                        "weight": rel.get("relationship_strength", 0.5)
                    })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "metadata": {"type": "fallback_graph", "timestamp": datetime.now().isoformat()}
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback graph building failed: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    def _create_fallback_node(self, paper: Dict[str, Any]) -> GraphNode:
        """Create fallback node when enhanced creation fails"""
        return GraphNode(
            id=str(paper.get("id", "")),
            label=paper.get("title", "Untitled")[:50],
            title=paper.get("title", "Untitled"),
            research_domain="Unknown",
            context_summary="",
            methodology="",
            innovations=[],
            contributions=[],
            quality_score=0.5,
            size=40,
            color="#F7DC6F"
        )
    
    def _create_fallback_edge(self, relationship: Dict[str, Any]) -> GraphEdge:
        """Create fallback edge when enhanced creation fails"""
        return GraphEdge(
            source=str(relationship.get("paper1_id", "")),
            target=str(relationship.get("paper2_id", "")),
            relationship_type="related",
            strength=0.5,
            context="",
            reasoning="",
            weight=0.5
        )

# Backward compatibility
class IntelligentGraphBuilder(EnhancedIntelligentGraphBuilder):
    """Backward compatibility wrapper"""
    def __init__(self):
        super().__init__()
        logger.info("🔄 Using enhanced graph builder with backward compatibility")
