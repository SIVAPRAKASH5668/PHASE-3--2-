import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from core.graph_builder import EnhancedIntelligentGraphBuilder
from agents.relationship_agent import RelationshipAgent

logger = logging.getLogger(__name__)

router = APIRouter()

# Request Models
class GraphBuildRequest(BaseModel):
    papers: List[Dict[str, Any]] = Field(..., min_items=1)
    relationships: Optional[List[Dict[str, Any]]] = Field([])
    multilingual_keywords: Optional[Dict[str, str]] = Field({})
    include_metrics: Optional[bool] = Field(True)
    layout_type: Optional[str] = Field("auto")

class GraphAnalyticsRequest(BaseModel):
    graph_id: Optional[str] = Field(None)
    papers: List[Dict[str, Any]] = Field(..., min_items=1)
    relationships: List[Dict[str, Any]] = Field(...)

@router.post("/build")
async def build_research_graph(request: GraphBuildRequest):
    """
    Build research graph optimized for React visualization
    """
    try:
        start_time = time.time()
        logger.info(f"🎨 Building graph for {len(request.papers)} papers")
        
        # Initialize graph builder
        graph_builder = EnhancedIntelligentGraphBuilder()
        
        # If no relationships provided, generate them
        relationships = request.relationships
        if not relationships and len(request.papers) >= 2:
            logger.info("🔗 Generating relationships for graph")
            
            relationship_agent = RelationshipAgent()
            await relationship_agent.initialize()
            
            relationships = await relationship_agent.analyze_relationships_batch(
                papers=request.papers,
                analysis_depth="standard",
                max_pairs=min(50, len(request.papers) * 2)
            )
            
            # Convert to dict format
            relationships = [
                {
                    "paper1_id": rel.paper1_id,
                    "paper2_id": rel.paper2_id,
                    "relationship_type": rel.relationship_type,
                    "relationship_strength": rel.relationship_strength,
                    "relationship_context": rel.relationship_context,
                    "connection_reasoning": rel.connection_reasoning,
                    "confidence_score": rel.confidence_score,
                    "ai_agent_used": rel.agent_used
                }
                for rel in relationships
            ]
        
        # Build comprehensive graph
        graph_data = graph_builder.build_graph(
            papers=request.papers,
            relationships=relationships,
            multilingual_keywords=request.multilingual_keywords
        )
        
        # Optimize for React frontend
        optimized_graph = {
            "nodes": [
                {
                    "id": node["id"],
                    "label": node["label"],
                    "title": node["title"],
                    "domain": node["research_domain"],
                    "size": node["size"],
                    "color": node["color"],
                    "language": node.get("language", "en"),
                    "quality": node["quality_score"],
                    "confidence": node.get("analysis_confidence", 0.8),
                    "aiAgent": node.get("ai_agent_used", "unknown"),
                    "metadata": {
                        "summary": node.get("context_summary", "")[:200],
                        "methodology": node.get("methodology", "")[:150],
                        "innovations": node.get("innovations", [])[:3],
                        "contributions": node.get("contributions", [])[:3
                        ]
                    }
                }
                for node in graph_data.get("nodes", [])
            ],
            "edges": [
                {
                    "id": f"{edge['source']}-{edge['target']}",
                    "source": edge["source"],
                    "target": edge["target"],
                    "type": edge["relationship_type"],
                    "weight": edge["weight"],
                    "strength": edge["strength"],
                    "color": _get_edge_color(edge["relationship_type"]),
                    "width": max(1, min(8, edge["weight"] * 8)),
                    "metadata": {
                        "context": edge.get("context", "")[:150],
                        "reasoning": edge.get("reasoning", "")[:200],
                        "confidence": edge.get("confidence_score", 0.7),
                        "aiAgent": edge.get("ai_agent_used", "unknown")
                    }
                }
                for edge in graph_data.get("edges", [])
            ]
        }
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "graph": optimized_graph,
            "metrics": graph_data.get("metrics", {}) if request.include_metrics else {},
            "clusters": graph_data.get("clusters", {}),
            "insights": graph_data.get("insights", {}),
            "layout_config": {
                "recommended_layout": _get_recommended_layout(len(optimized_graph["nodes"])),
                "performance_mode": len(optimized_graph["nodes"]) > 500,
                "show_labels": len(optimized_graph["nodes"]) < 100,
                "enable_physics": len(optimized_graph["nodes"]) < 200
            },
            "visualization_config": graph_data.get("visualization_config", {}),
            "metadata": {
                "processing_time": processing_time,
                "node_count": len(optimized_graph["nodes"]),
                "edge_count": len(optimized_graph["edges"]),
                "relationship_types": list(set(e["type"] for e in optimized_graph["edges"])),
                "domains_present": list(set(n["domain"] for n in optimized_graph["nodes"])),
                "languages_present": list(set(n["language"] for n in optimized_graph["nodes"]))
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Graph building failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "graph": {"nodes": [], "edges": []},
            "metadata": {"processing_time": time.time() - start_time}
        }

@router.post("/analytics")
async def get_graph_analytics(request: GraphAnalyticsRequest):
    """
    Get comprehensive graph analytics
    """
    try:
        start_time = time.time()
        
        # Build graph for analysis
        graph_builder = EnhancedIntelligentGraphBuilder()
        graph_data = graph_builder.build_graph(
            papers=request.papers,
            relationships=request.relationships
        )
        
        metrics = graph_data.get("metrics", {})
        
        return {
            "success": True,
            "analytics": {
                "network_metrics": metrics.get("basic_metrics", {}),
                "centrality_metrics": metrics.get("centrality_metrics", {}),
                "clustering_metrics": metrics.get("clustering_metrics", {}),
                "research_metrics": metrics.get("research_metrics", {}),
                "quality_metrics": metrics.get("quality_metrics", {}),
                "language_metrics": metrics.get("language_metrics", {}),
                "ai_metrics": metrics.get("ai_metrics", {}),
                "graph_health_score": metrics.get("graph_health_score", 0.5)
            },
            "insights": graph_data.get("insights", {}),
            "recommendations": _generate_graph_recommendations(metrics),
            "metadata": {
                "processing_time": time.time() - start_time,
                "analysis_depth": "comprehensive"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Graph analytics failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "analytics": {}
        }

@router.get("/layout-options")
async def get_layout_options():
    """
    Get available graph layout options for React frontend
    """
    return {
        "success": True,
        "layouts": {
            "spring": {
                "name": "Spring Layout",
                "description": "Force-directed layout good for small to medium graphs",
                "best_for": "< 50 nodes",
                "physics_enabled": True
            },
            "hierarchical": {
                "name": "Hierarchical Layout", 
                "description": "Tree-like structure showing relationships hierarchy",
                "best_for": "Research lineage",
                "physics_enabled": False
            },
            "circular": {
                "name": "Circular Layout",
                "description": "Nodes arranged in a circle",
                "best_for": "Domain clustering",
                "physics_enabled": False
            },
            "grid": {
                "name": "Grid Layout",
                "description": "Regular grid arrangement",
                "best_for": "Large datasets",
                "physics_enabled": False
            },
            "force_atlas": {
                "name": "Force Atlas",
                "description": "Advanced force-directed with clustering",
                "best_for": "Community detection",
                "physics_enabled": True
            }
        }
    }

def _get_edge_color(relationship_type: str) -> str:
    """Get color for edge based on relationship type"""
    colors = {
        "builds_upon": "#2196F3",
        "improves": "#4CAF50", 
        "extends": "#FF9800",
        "complements": "#9C27B0",
        "applies": "#795548",
        "related": "#607D8B",
        "contradicts": "#F44336",
        "methodology_shared": "#3F51B5",
        "domain_overlap": "#009688",
        "competing": "#E91E63"
    }
    return colors.get(relationship_type, "#757575")

def _get_recommended_layout(node_count: int) -> str:
    """Get recommended layout based on graph size"""
    if node_count < 15:
        return "spring"
    elif node_count < 50:
        return "force_atlas"
    elif node_count < 100:
        return "hierarchical"
    else:
        return "grid"

def _generate_graph_recommendations(metrics: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on graph metrics"""
    recommendations = []
    
    basic_metrics = metrics.get("basic_metrics", {})
    quality_metrics = metrics.get("quality_metrics", {})
    
    # Density recommendations
    density = basic_metrics.get("density", 0)
    if density < 0.1:
        recommendations.append("Graph is sparse - consider expanding search scope")
    elif density > 0.7:
        recommendations.append("Graph is dense - consider filtering for stronger relationships")
    
    # Quality recommendations
    avg_quality = quality_metrics.get("average_quality_score", 0.5)
    if avg_quality < 0.6:
        recommendations.append("Consider refining search criteria for higher quality papers")
    
    # Connectivity recommendations
    connectivity = basic_metrics.get("connectivity", "unknown")
    if connectivity == "disconnected":
        recommendations.append("Graph has isolated components - broaden search terms")
    
    return recommendations[:5]  # Limit to top 5 recommendations

@router.get("/health")
async def graph_health_check():
    """
    Health check for graph service
    """
    try:
        # Test graph builder
        graph_builder = EnhancedIntelligentGraphBuilder()
        
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "components": {
                "graph_builder": "healthy",
                "networkx": "available",
                "visualization_engine": "ready"
            },
            "capabilities": [
                "graph_construction",
                "network_analysis",
                "visualization_optimization",
                "metrics_calculation"
            ]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
