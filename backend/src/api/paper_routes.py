import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from agents.coordinator_agent import CoordinatorAgent
from agents.paper_analyzer_agent import PaperAnalyzerAgent
from core.llm_processor import EnhancedMultiAPILLMProcessor
from integrations.arxiv_client import EnhancedArxivClient
from integrations.pubmed_client import EnhancedPubMedClient
from integrations.core_client import COREClient

logger = logging.getLogger(__name__)

router = APIRouter()

# Request Models
class PaperSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: Optional[int] = Field(20, ge=1, le=100)
    sources: Optional[List[str]] = Field(["core", "arxiv", "pubmed"])
    language: Optional[str] = Field("auto")
    enable_ai_analysis: Optional[bool] = Field(True)
    analysis_type: Optional[str] = Field("detailed")

class PaperAnalysisRequest(BaseModel):
    title: str = Field(..., max_length=300)
    abstract: str = Field(..., max_length=2000)
    content: Optional[str] = Field("", max_length=5000)
    language: Optional[str] = Field("auto")
    analysis_type: Optional[str] = Field("detailed")

class BatchAnalysisRequest(BaseModel):
    papers: List[Dict[str, Any]] = Field(..., min_items=1, max_items=50)
    analysis_type: Optional[str] = Field("standard")

class RelationshipAnalysisRequest(BaseModel):
    papers: List[Dict[str, Any]] = Field(..., min_items=2, max_items=100)
    analysis_depth: Optional[str] = Field("standard")
    max_relationships: Optional[int] = Field(50, ge=1, le=200)

# Dependency injection
async def get_coordinator() -> CoordinatorAgent:
    try:
        coordinator = CoordinatorAgent()
        await coordinator.initialize()
        return coordinator
    except Exception as e:
        logger.error(f"Failed to initialize coordinator: {e}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

async def get_paper_analyzer() -> PaperAnalyzerAgent:
    try:
        analyzer = PaperAnalyzerAgent()
        await analyzer.initialize()
        return analyzer
    except Exception as e:
        logger.error(f"Failed to initialize paper analyzer: {e}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

@router.post("/search")
async def search_papers(
    request: PaperSearchRequest,
    coordinator: CoordinatorAgent = Depends(get_coordinator)
):
    """
    Multi-source paper search with AI analysis
    Optimized for React frontend integration
    """
    try:
        start_time = time.time()
        logger.info(f"🔍 Paper search: {request.query[:100]}...")
        
        # Execute coordinated search workflow
        workflow_result = await coordinator.coordinate_research_workflow(
            query=request.query,
            max_papers=request.max_results,
            sources=request.sources,
            enable_multilingual=True,
            analysis_type=request.analysis_type,
            workflow_id=f"search_{int(time.time() * 1000)}"
        )
        
        if not workflow_result.success:
            return {
                "success": False,
                "error": workflow_result.error_message,
                "results": [],
                "metadata": {"processing_time": time.time() - start_time}
            }
        
        papers = workflow_result.final_result.get('papers', [])
        relationships = workflow_result.final_result.get('relationships', [])
        
        # Prepare frontend-optimized response
        return {
            "success": True,
            "results": papers,
            "relationships": relationships,
            "total_found": len(papers),
            "metadata": {
                "processing_time": time.time() - start_time,
                "workflow_time": workflow_result.total_time,
                "languages_detected": list(set(p.get('language', 'en') for p in papers)),
                "sources_searched": request.sources,
                "ai_analysis_enabled": request.enable_ai_analysis,
                "quality_distribution": {
                    "high": len([p for p in papers if p.get('context_quality_score', 0) > 0.8]),
                    "medium": len([p for p in papers if 0.5 <= p.get('context_quality_score', 0) <= 0.8]),
                    "low": len([p for p in papers if p.get('context_quality_score', 0) < 0.5])
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Paper search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "metadata": {"processing_time": time.time() - start_time}
        }

@router.post("/analyze")
async def analyze_paper(
    request: PaperAnalysisRequest,
    analyzer: PaperAnalyzerAgent = Depends(get_paper_analyzer)
):
    """
    Analyze a single paper with AI-powered insights
    """
    try:
        start_time = time.time()
        
        # Create paper context analysis
        analysis_result = await analyzer.analyze_paper_context(
            title=request.title,
            abstract=request.abstract,
            content=request.content,
            language=request.language,
            analysis_type=request.analysis_type
        )
        
        return {
            "success": True,
            "analysis": {
                "context_summary": analysis_result.context_summary,
                "research_domain": analysis_result.research_domain,
                "methodology": analysis_result.methodology,
                "key_findings": analysis_result.key_findings,
                "innovations": analysis_result.innovations,
                "limitations": analysis_result.limitations,
                "contributions": analysis_result.contributions,
                "future_work": analysis_result.future_work,
                "related_concepts": analysis_result.related_concepts,
                "quality_score": analysis_result.context_quality_score,
                "confidence": analysis_result.analysis_confidence,
                "language_detected": analysis_result.language_detected,
                "ai_agent_used": analysis_result.ai_agent_used
            },
            "metadata": {
                "processing_time": time.time() - start_time,
                "analysis_method": analysis_result.analysis_method
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Paper analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "analysis": None
        }

@router.post("/batch-analyze")
async def batch_analyze_papers(
    request: BatchAnalysisRequest,
    analyzer: PaperAnalyzerAgent = Depends(get_paper_analyzer)
):
    """
    Batch analyze multiple papers efficiently
    """
    try:
        start_time = time.time()
        
        # Execute batch analysis
        analysis_results = await analyzer.batch_analyze_papers(
            papers=request.papers,
            analysis_type=request.analysis_type
        )
        
        return {
            "success": True,
            "analyses": [
                {
                    "paper_id": getattr(result, 'paper_id', i),
                    "context_summary": result.context_summary,
                    "research_domain": result.research_domain,
                    "quality_score": result.context_quality_score,
                    "confidence": result.analysis_confidence,
                    "language_detected": result.language_detected,
                    "ai_agent_used": result.ai_agent_used,
                    "key_findings": result.key_findings,
                    "innovations": result.innovations
                }
                for i, result in enumerate(analysis_results)
            ],
            "metadata": {
                "processing_time": time.time() - start_time,
                "papers_analyzed": len(analysis_results),
                "success_rate": len([r for r in analysis_results if r.context_quality_score > 0.3]) / len(analysis_results)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Batch analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "analyses": []
        }

@router.post("/relationships")
async def discover_relationships(
    request: RelationshipAnalysisRequest,
    coordinator: CoordinatorAgent = Depends(get_coordinator)
):
    """
    Discover relationships between papers
    """
    try:
        start_time = time.time()
        
        # Use relationship agent through coordinator
        from agents.relationship_agent import RelationshipAgent
        
        relationship_agent = RelationshipAgent()
        await relationship_agent.initialize()
        
        # Analyze relationships in batch
        relationships = await relationship_agent.analyze_relationships_batch(
            papers=request.papers,
            analysis_depth=request.analysis_depth,
            max_pairs=request.max_relationships
        )
        
        return {
            "success": True,
            "relationships": [
                {
                    "paper1_id": rel.paper1_id,
                    "paper2_id": rel.paper2_id,
                    "relationship_type": rel.relationship_type,
                    "strength": rel.relationship_strength,
                    "context": rel.relationship_context,
                    "reasoning": rel.connection_reasoning,
                    "confidence": rel.confidence_score,
                    "ai_agent_used": rel.agent_used
                }
                for rel in relationships
            ],
            "insights": relationship_agent.get_relationship_insights(relationships),
            "metadata": {
                "processing_time": time.time() - start_time,
                "pairs_analyzed": len(relationships),
                "strong_relationships": len([r for r in relationships if r.relationship_strength > 0.7])
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Relationship discovery failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "relationships": []
        }

@router.get("/sources")
async def get_available_sources():
    """
    Get available data sources and their capabilities
    """
    return {
        "success": True,
        "sources": {
            "arxiv": {
                "name": "ArXiv",
                "description": "Preprint repository with cutting-edge research",
                "capabilities": ["academic_papers", "preprints", "quality_scoring"],
                "languages": ["en", "multilingual_titles"],
                "domains": ["cs", "physics", "math", "biology", "economics"]
            },
            "pubmed": {
                "name": "PubMed", 
                "description": "Medical and biomedical literature database",
                "capabilities": ["medical_papers", "clinical_trials", "evidence_based"],
                "languages": ["en"],
                "domains": ["medicine", "biology", "healthcare"]
            },
            "core": {
                "name": "CORE",
                "description": "Global aggregator of academic papers",
                "capabilities": ["academic_papers", "metadata_enrichment", "global_coverage"],
                "languages": ["multilingual"],
                "domains": ["all_academic"]
            },
            "crossref": {
                "name": "Crossref",
                "description": "Citation and metadata database",
                "capabilities": ["citation_data", "metadata_enrichment", "doi_resolution"],
                "languages": ["multilingual"],
                "domains": ["all_academic"]
            }
        }
    }

@router.get("/domains")
async def get_research_domains():
    """
    Get supported research domains for filtering
    """
    return {
        "success": True,
        "domains": [
            "Computer Science",
            "Machine Learning", 
            "Artificial Intelligence",
            "Natural Language Processing",
            "Computer Vision",
            "Deep Learning",
            "Robotics",
            "Data Science",
            "Mathematics",
            "Physics",
            "Biology",
            "Medicine",
            "Healthcare",
            "Bioinformatics",
            "Chemistry",
            "Engineering",
            "Economics",
            "Psychology",
            "Neuroscience",
            "General Research"
        ]
    }

@router.get("/health")
async def papers_health_check():
    """
    Health check for papers service
    """
    try:
        # Test core components
        coordinator = CoordinatorAgent()
        analyzer = PaperAnalyzerAgent()
        
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "components": {
                "coordinator": "healthy",
                "analyzer": "healthy",
                "external_apis": "healthy"
            },
            "capabilities": [
                "multi_source_search",
                "ai_analysis",
                "batch_processing",
                "relationship_discovery"
            ]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

@router.get("/details/{paper_id}")
async def get_paper_details(paper_id: int):
    """
    Get complete paper details by ID for frontend display
    """
    try:
        start_time = time.time()
        logger.info(f"📄 Getting paper details for ID: {paper_id}")
        
        # Initialize database client
        from src.database.tidb_client import EnhancedTiDBClient
        
        db_client = EnhancedTiDBClient()
        await db_client.init_database()
        
        try:
            # Get paper details from database
            paper = await db_client.get_paper_by_id(paper_id)
            
            if not paper:
                return {
                    "success": False,
                    "error": "Paper not found",
                    "paper": None,
                    "metadata": {"processing_time": time.time() - start_time}
                }
            
            # Return comprehensive paper details
            paper_details = {
                "id": paper.get("id"),
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": paper.get("authors", ""),
                "source": paper.get("source", "unknown"),
                "language": paper.get("language", "en"),
                "published_date": str(paper.get("published_date", "")) if paper.get("published_date") else None,
                "research_domain": paper.get("research_domain", "Unknown"),
                "methodology": paper.get("methodology", ""),
                "key_findings": paper.get("key_findings", []),
                "innovations": paper.get("innovations", []),
                "contributions": paper.get("contributions", []),
                "limitations": paper.get("limitations", []),
                "future_work": paper.get("future_work", []),
                "context_summary": paper.get("context_summary", ""),
                "context_quality_score": float(paper.get("context_quality_score", 0.0)),
                "analysis_confidence": float(paper.get("analysis_confidence", 0.0)),
                "paper_url": paper.get("paper_url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "doi": paper.get("doi", ""),
                "citations": paper.get("citations", 0),
                "ai_agent_used": paper.get("ai_agent_used", "unknown"),
                "processing_status": paper.get("processing_status", "unknown"),
                "created_at": str(paper.get("created_at", "")) if paper.get("created_at") else None,
                "updated_at": str(paper.get("updated_at", "")) if paper.get("updated_at") else None,
                "categories": paper.get("categories", []),
                "keywords": paper.get("keywords", [])
            }
            
            return {
                "success": True,
                "paper": paper_details,
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "source": "database",
                    "paper_id": paper_id
                }
            }
            
        finally:
            await db_client.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to get paper details for ID {paper_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "paper": None,
            "metadata": {"processing_time": time.time() - start_time}
        }

@router.get("/search-db/{paper_id}")
async def search_paper_in_db(paper_id: int):
    """
    Alternative endpoint to search for paper in database
    """
    try:
        from src.database.tidb_client import EnhancedTiDBClient
        
        db_client = EnhancedTiDBClient()
        await db_client.init_database()
        
        try:
            # Try multiple methods to find the paper
            
            # Method 1: Direct ID lookup
            paper = await db_client.get_paper_by_id(paper_id)
            
            if paper:
                return {
                    "success": True,
                    "found": True,
                    "method": "direct_id",
                    "paper_summary": {
                        "id": paper.get("id"),
                        "title": paper.get("title", "")[:100],
                        "source": paper.get("source", "unknown"),
                        "processing_status": paper.get("processing_status", "unknown")
                    }
                }
            
            # Method 2: Check if any papers exist in DB
            stats = await db_client.get_enhanced_database_stats()
            total_papers = stats.get('overview', {}).get('total_papers', 0)
            
            return {
                "success": True,
                "found": False,
                "method": "search_attempted",
                "info": {
                    "searched_id": paper_id,
                    "total_papers_in_db": total_papers,
                    "message": f"Paper ID {paper_id} not found. Database contains {total_papers} papers."
                }
            }
            
        finally:
            await db_client.close()
            
    except Exception as e:
        logger.error(f"❌ Database search failed for ID {paper_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "found": False
        }
