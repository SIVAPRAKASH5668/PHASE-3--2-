import asyncio
import logging
import time
import sys
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Union
from types import ModuleType

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ✅ FIXED: Import with proper type handling and correct logging
graph_router = None
paper_router = None
translation_router = None
SingletonManager = None
CoordinatorAgent = None
settings = None

try:
    from src.api.graph_routes import router as graph_router
    logger.info("✅ Graph routes imported successfully")
except ImportError as e:
    logger.warning(f"Could not import graph_routes: {e}")

try:
    from src.api.paper_routes import router as paper_router  
    logger.info("✅ Paper routes imported successfully")
except ImportError as e:
    logger.warning(f"Could not import paper_routes: {e}")

try:
    from src.api.translation_routes import router as translation_router
    logger.info("✅ Translation routes imported successfully")
except ImportError as e:
    logger.warning(f"Could not import translation_routes: {e}")

try:
    from src.core.singleton_manager import SingletonManager
    logger.info("✅ SingletonManager imported successfully")
except ImportError as e:
    logger.warning(f"Could not import SingletonManager: {e}")

try:
    from src.agents.coordinator_agent import CoordinatorAgent
    logger.info("✅ CoordinatorAgent imported successfully")
except ImportError as e:
    logger.warning(f"Could not import CoordinatorAgent: {e}")

try:
    from src.config.settings import settings
    logger.info("✅ Settings imported successfully")
except ImportError as e:
    logger.warning(f"Could not import settings: {e}")

# Request/Response Models for React Integration
class ResearchDiscoveryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Research query")
    max_results: Optional[int] = Field(20, ge=1, le=100, description="Maximum results")
    enable_multilingual: Optional[bool] = Field(True, description="Enable multilingual processing")
    sources: Optional[List[str]] = Field(["core", "arxiv", "pubmed"], description="Data sources")
    multilingual_keywords: Optional[Dict[str, str]] = Field({}, description="Keywords in different languages")
    analysis_type: Optional[str] = Field("detailed", description="Analysis depth")
    enable_graph: Optional[bool] = Field(True, description="Enable graph generation")

class SystemStatusResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]
    performance: Dict[str, Any]
    capabilities: List[str]

class APIResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Application lifespan management with safe initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown management"""
    logger.info("🚀 Starting Multilingual Research Discovery Platform")
    
    # Initialize app state
    app.state.singleton_manager = None
    app.state.coordinator = None
    app.state.initialization_status = "starting"
    
    try:
        # Safe SingletonManager initialization
        if SingletonManager:
            try:
                singleton_manager = SingletonManager()
                
                # Check if initialize method exists
                if hasattr(singleton_manager, 'initialize'):
                    await singleton_manager.initialize()
                    logger.info("✅ SingletonManager initialized with initialize() method")
                elif hasattr(singleton_manager, 'setup'):
                    await singleton_manager.setup()
                    logger.info("✅ SingletonManager initialized with setup() method")
                else:
                    logger.info("ℹ️ SingletonManager created without async initialization")
                
                app.state.singleton_manager = singleton_manager
            except Exception as e:
                logger.warning(f"⚠️ SingletonManager initialization failed: {e}")
                app.state.singleton_manager = None
        else:
            logger.warning("⚠️ SingletonManager not available")
        
        # Safe CoordinatorAgent initialization
        if CoordinatorAgent:
            try:
                coordinator = CoordinatorAgent()
                
                # Check if initialize method exists
                if hasattr(coordinator, 'initialize'):
                    await coordinator.initialize()
                    logger.info("✅ CoordinatorAgent initialized with initialize() method")
                elif hasattr(coordinator, 'setup'):
                    await coordinator.setup()
                    logger.info("✅ CoordinatorAgent initialized with setup() method")
                else:
                    logger.info("ℹ️ CoordinatorAgent created without async initialization")
                
                app.state.coordinator = coordinator
            except Exception as e:
                logger.warning(f"⚠️ CoordinatorAgent initialization failed: {e}")
                app.state.coordinator = None
        else:
            logger.warning("⚠️ CoordinatorAgent not available")
        
        app.state.initialization_status = "completed"
        logger.info("✅ Backend initialization completed (with fallbacks for missing components)")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        app.state.initialization_status = "failed"
        # Don't raise - continue with limited functionality
        yield
    finally:
        # Cleanup on shutdown
        logger.info("🔄 Shutting down backend services...")
        try:
            if hasattr(app.state, 'singleton_manager') and app.state.singleton_manager:
                if hasattr(app.state.singleton_manager, 'shutdown'):
                    await app.state.singleton_manager.shutdown()
                elif hasattr(app.state.singleton_manager, 'cleanup'):
                    await app.state.singleton_manager.cleanup()
            
            if hasattr(app.state, 'coordinator') and app.state.coordinator:
                if hasattr(app.state.coordinator, 'shutdown'):
                    await app.state.coordinator.shutdown()
                elif hasattr(app.state.coordinator, 'cleanup'):
                    await app.state.coordinator.cleanup()
            
            logger.info("✅ Backend shutdown completed")
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

# Create FastAPI application optimized for React
app = FastAPI(
    title="Multilingual Research Discovery Platform",
    description="Advanced AI-powered research discovery with TiDB vector search",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration for React Vite (development and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",  # Self reference for testing
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add compression for better performance
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Safe router inclusion
if paper_router:
    app.include_router(paper_router, prefix="/api/papers", tags=["Papers"])
    logger.info("✅ Paper routes loaded")
else:
    logger.warning("⚠️ Paper routes not available")

if graph_router:
    app.include_router(graph_router, prefix="/api/graph", tags=["Graph"])
    logger.info("✅ Graph routes loaded")
else:
    logger.warning("⚠️ Graph routes not available")

if translation_router:
    app.include_router(translation_router, prefix="/api/translate", tags=["Translation"])
    logger.info("✅ Translation routes loaded")
else:
    logger.warning("⚠️ Translation routes not available")

# ✅ FIXED: Safe dependency injection with proper typing
async def get_coordinator() -> Any:
    """Get coordinator agent from app state"""
    if hasattr(app.state, 'coordinator') and app.state.coordinator:
        return app.state.coordinator
    raise HTTPException(
        status_code=503, 
        detail="Coordinator not available - service may be starting up or have initialization issues"
    )

async def get_singleton_manager() -> Any:
    """Get singleton manager from app state"""
    if hasattr(app.state, 'singleton_manager') and app.state.singleton_manager:
        return app.state.singleton_manager
    raise HTTPException(
        status_code=503, 
        detail="Singleton manager not available - service may be starting up or have initialization issues"
    )

# Basic health endpoint that always works
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multilingual Research Discovery Platform API",
        "version": "2.0.0",
        "status": "active",
        "api_docs": "/api/docs",
        "frontend_integration": "React Vite optimized",
        "initialization_status": getattr(app.state, 'initialization_status', 'unknown'),
        "available_services": {
            "papers": paper_router is not None,
            "graph": graph_router is not None,
            "translation": translation_router is not None,
            "coordinator": getattr(app.state, 'coordinator', None) is not None,
            "singleton_manager": getattr(app.state, 'singleton_manager', None) is not None
        },
        "capabilities": [
            "Multi-source research discovery",
            "AI-powered analysis",
            "Graph visualization",
            "Multilingual support (20+ languages)",
            "TiDB vector search"
        ]
    }

@app.get("/health")
async def basic_health():
    """Basic health check that works without dependencies"""
    return {
        "status": "healthy" if getattr(app.state, 'initialization_status', 'unknown') == "completed" else "starting",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": "API server is running",
        "version": "2.0.0",
        "initialization_status": getattr(app.state, 'initialization_status', 'unknown'),
        "services": {
            "coordinator": getattr(app.state, 'coordinator', None) is not None,
            "singleton_manager": getattr(app.state, 'singleton_manager', None) is not None
        }
    }

@app.get("/api/health")
async def detailed_health():
    """Detailed health check with component status"""
    try:
        # Try to get detailed health if singleton manager is available
        if hasattr(app.state, 'singleton_manager') and app.state.singleton_manager:
            if hasattr(app.state.singleton_manager, 'get_system_health'):
                health_data = await app.state.singleton_manager.get_system_health()
                return health_data
        
        # Fallback health response
        return {
            "status": "healthy" if getattr(app.state, 'initialization_status') == "completed" else "degraded",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "components": {
                "api_server": "healthy",
                "coordinator": "healthy" if getattr(app.state, 'coordinator') else "unavailable",
                "singleton_manager": "healthy" if getattr(app.state, 'singleton_manager') else "unavailable"
            },
            "performance": {
                "uptime": "< 1 hour",
                "initialization_status": getattr(app.state, 'initialization_status', 'unknown')
            },
            "capabilities": [
                "basic_api",
                "health_monitoring",
                "error_handling"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "error": str(e),
            "components": {"error": "health_check_failed"}
        }

# Core endpoint for React frontend integration (with safe fallback)
@app.post("/api/research/discover")
async def discover_research(request: ResearchDiscoveryRequest):
    """
    ✅ FIXED: Main research discovery endpoint with proper graph data extraction
    """
    try:
        start_time = time.time()
        logger.info(f"🔍 Research discovery request: {request.query[:100]}...")
        
        if not hasattr(app.state, 'coordinator') or not app.state.coordinator:
            return {
                "success": False,
                "error": "Research discovery service is initializing. Please try again in a moment.",
                "data": None,
                "metadata": {"processing_time": time.time() - start_time, "service_status": "initializing"}
            }
        
        try:
            coordinator = app.state.coordinator
            workflow_result = await coordinator.coordinate_research_workflow(
                query=request.query,
                max_papers=request.max_results,
                workflow_id=f"discover_{int(time.time() * 1000)}"
            )
            
            if workflow_result and workflow_result.success:
                # ✅ FIXED: Extract data properly from the result structure
                final_result = workflow_result.final_result
                
                if isinstance(final_result, dict):
                    papers = final_result.get('papers', [])
                    relationships = final_result.get('relationships', [])
                    graph_data = final_result.get('graph_data', {})  # ✅ CRITICAL: Get graph data!
                    
                    # ✅ CRITICAL: Build the proper response with graph data
                    response_data = {
                        "papers": papers,
                        "relationships": relationships,
                        "total_found": len(papers),
                        "has_graph_data": len(papers) >= 2 and len(relationships) > 0
                    }
                    
                    # ✅ CRITICAL: Include the actual graph structure
                    if graph_data and isinstance(graph_data, dict):
                        response_data['graph'] = {
                            'nodes': graph_data.get('nodes', []),
                            'edges': graph_data.get('edges', []),
                            'clusters': graph_data.get('clusters', {}),
                            'metrics': graph_data.get('metrics', {}),
                            'insights': graph_data.get('insights', {}),
                            'metadata': graph_data.get('metadata', {})
                        }
                        response_data['has_graph_data'] = True
                        
                        logger.info(f"📊 Graph included: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges")
                    
                    return {
                        "success": True,
                        "data": response_data,
                        "metadata": {
                            "processing_time": time.time() - start_time,
                            "workflow_time": workflow_result.total_time,
                            "workflow_id": workflow_result.workflow_id,
                            "phases_completed": workflow_result.phases_completed
                        }
                    }
                
                # Fallback for unexpected result structure
                return {
                    "success": True,
                    "data": {
                        "papers": [],
                        "relationships": [],
                        "total_found": 0,
                        "has_graph_data": False,
                        "debug_info": {
                            "result_type": str(type(final_result)),
                            "result_keys": list(final_result.keys()) if isinstance(final_result, dict) else "not_dict",
                            "workflow_success": workflow_result.success
                        }
                    },
                    "metadata": {
                        "processing_time": time.time() - start_time,
                        "workflow_time": workflow_result.total_time,
                        "debug_mode": True
                    }
                }
            else:
                logger.error(f"❌ Workflow failed: {workflow_result}")
                return {
                    "success": False,
                    "error": f"Workflow failed: {getattr(workflow_result, 'error_summary', 'Unknown error')}",
                    "data": None,
                    "metadata": {"processing_time": time.time() - start_time}
                }
            
        except Exception as workflow_error:
            logger.error(f"❌ Workflow execution failed: {workflow_error}")
            return {
                "success": False,
                "error": f"Research workflow error: {str(workflow_error)}",
                "data": None,
                "metadata": {"processing_time": time.time() - start_time}
            }
        
    except Exception as e:
        logger.error(f"❌ Research discovery failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": None,
            "metadata": {"processing_time": time.time() - start_time}
        }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "message": "The requested API endpoint does not exist"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    print("🚀 Starting Multilingual Research Discovery Platform...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[:3]}...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )