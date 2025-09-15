import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
import logging
import json


logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Enhanced application settings with comprehensive configuration
    """
    
    # ✅ FIXED: Updated to use SettingsConfigDict and field_validator
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # This allows extra env vars without errors
    )
    
    # Application Settings
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    APP_NAME: str = "PolyResearch Agent"
    APP_VERSION: str = "2.0.0"
    
    # Database Configuration (TiDB Cloud)
    TIDB_HOST: str = "gateway01.ap-northeast-1.prod.aws.tidbcloud.com"
    TIDB_PORT: int = 4000
    TIDB_USER: str = "mBwAfVedsv1M387.root"
    TIDB_PASSWORD: str = "YeDQOXhzOPxHXsn7"
    TIDB_DATABASE: str = "test"
    
    # ✅ FIXED: Enhanced AI API Configuration with proper field definitions
    GROQ_API_KEY: str = Field(default="")  # Single key field
    GROQ_API_KEYS: List[str] = Field(default_factory=list)  # List of keys
    KIMI_API_KEY: str = "sk-UFGzFkgm7e7JEf7M50JeLWF1CtwB7auEfwBWlhI5hAmXwoN6"
    KIMI_MODEL: str = "moonshot-v1-8k"
    KIMI_BASE_URL: str = "https://api.moonshot.cn/v1"
    
    # External API Keys
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    CORE_API_KEY: str = "3wyaMDkxpeIzV7H4YhvGtg9mJnTBXsod"
    
    # Multilingual Configuration
    DEFAULT_LANGUAGE: str = "en"
    SUPPORTED_LANGUAGES: List[str] = Field(default=[
        "en", "zh", "de", "fr", "ja", "ko", "es", "ru", "it", "pt", "ar", "hi",
        "nl", "sv", "no", "da", "fi", "tr", "pl", "cs", "hu"
    ])
    RESEARCH_PRIORITY_LANGUAGES: List[str] = Field(default=[
        "en", "zh", "de", "fr", "ja", "ko", "es", "ru", "it", "pt", "ar", "hi"
    ])
    
    # Translation Service Configuration
    GOOGLE_TRANSLATE_API_KEY: str = ""
    DEEPL_API_KEY: str = ""
    AZURE_TRANSLATOR_KEY: str = ""
    TRANSLATION_CACHE_TTL: int = 3600
    MAX_TRANSLATION_CACHE_SIZE: int = 10000
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_CACHE_SIZE: int = 1000
    BATCH_EMBEDDING_SIZE: int = 32
    
    # LLM Processing Configuration
    LLM_REQUEST_TIMEOUT: int = 30
    LLM_MAX_RETRIES: int = 3
    LLM_RATE_LIMIT_DELAY: float = 1.0
    LLM_TOKENS_PER_MINUTE: int = 5800
    LLM_REQUESTS_PER_MINUTE: int = 28
    LLM_BATCH_SIZE: int = 4
    LLM_MAX_CONCURRENT_BATCHES: int = 2
    
    # Multi-AI Agent Configuration
    MAX_CONCURRENT_AGENTS: int = 12
    AGENT_TASK_TIMEOUT: int = 60
    PAPER_ANALYSIS_MAX_CONCURRENT: int = 6
    RELATIONSHIP_ANALYSIS_MAX_CONCURRENT: int = 4
    COORDINATOR_WORKFLOW_TIMEOUT: int = 300
    
    # Search Configuration
    MAX_PAPERS_PER_SEARCH: int = 50
    MAX_PAPERS_DEFAULT: int = 25
    ARXIV_MAX_RESULTS: int = 100
    PUBMED_MAX_RESULTS: int = 100
    SEARCH_TIMEOUT: int = 30
    
    # Graph Building Configuration
    MAX_GRAPH_NODES: int = 100
    MAX_GRAPH_EDGES: int = 200
    GRAPH_RELATIONSHIP_THRESHOLD: float = 0.3
    GRAPH_QUALITY_THRESHOLD: float = 0.5
    GRAPH_CACHE_TTL: int = 1800
    
    # Vector Operations Configuration
    VECTOR_SIMILARITY_THRESHOLD: float = 0.6
    VECTOR_SEARCH_LIMIT: int = 20
    VECTOR_BATCH_SIZE: int = 100
    TIDB_VECTOR_DIMENSION: int = 384
    
    # Performance Configuration
    ENABLE_CACHING: bool = True
    CACHE_TTL_DEFAULT: int = 3600
    MAX_CACHE_SIZE: int = 10000
    ENABLE_PERFORMANCE_MONITORING: bool = True
    ENABLE_DETAILED_LOGGING: bool = True
    
    # Workflow Configuration
    WORKFLOW_MAX_PHASES: int = 8
    WORKFLOW_PHASE_TIMEOUT: int = 120
    ENABLE_ERROR_RECOVERY: bool = True
    ENABLE_ADAPTIVE_TIMEOUTS: bool = True
    ENABLE_SMART_ERROR_RECOVERY: bool = True
    
    # Text Processing Configuration
    TEXT_CLEANING_PRESERVE_ACADEMIC: bool = True
    MAX_TEXT_LENGTH: int = 5000
    MIN_WORD_LENGTH: int = 3
    MAX_KEYWORDS_EXTRACT: int = 20
    TEXT_SIMILARITY_THRESHOLD: float = 0.4
    
    # API Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = True
    ACCESS_LOG: bool = True
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(default=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080"
    ])
    CORS_METHODS: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    CORS_HEADERS: List[str] = Field(default=["*"])
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Domain-Specific Configuration
    RESEARCH_DOMAINS: List[str] = Field(default=[
        "Computer Vision",
        "Natural Language Processing", 
        "Machine Learning",
        "Deep Learning",
        "Artificial Intelligence",
        "Healthcare",
        "Robotics",
        "Data Science",
        "Bioinformatics",
        "Computer Graphics",
        "General Research"
    ])
    
    DOMAIN_LANGUAGE_PREFERENCES: Dict[str, List[str]] = Field(default={
        "Computer Science": ["en", "zh", "de", "ja", "ko"],
        "Medicine": ["en", "de", "fr", "zh", "ja"],
        "Engineering": ["en", "de", "zh", "ja", "ko"],
        "Physics": ["en", "de", "fr", "ru", "zh"],
        "Biology": ["en", "de", "fr", "zh", "ja"],
        "Mathematics": ["en", "de", "fr", "ru", "zh"]
    })
    
    # Feature Flags
    ENABLE_MULTILINGUAL_SEARCH: bool = True
    ENABLE_VECTOR_SEARCH: bool = True
    ENABLE_AI_AGENTS: bool = True
    ENABLE_KIMI_INTEGRATION: bool = True
    ENABLE_RELATIONSHIP_ANALYSIS: bool = True
    ENABLE_GRAPH_VISUALIZATION: bool = True
    ENABLE_PERFORMANCE_OPTIMIZATION: bool = True
    
    # Monitoring and Analytics
    ENABLE_METRICS_COLLECTION: bool = True
    METRICS_ENDPOINT: str = "/metrics"
    HEALTH_CHECK_ENDPOINT: str = "/health"
    STATUS_ENDPOINT: str = "/status"
    
    # ✅ FIXED: Updated validators to use field_validator (Pydantic v2)
    @field_validator('GROQ_API_KEYS', mode='before')
    @classmethod
    def validate_groq_keys(cls, v):
        """Parse GROQ API keys from JSON string or return list"""
        if isinstance(v, str):
            if v.strip():
                try:
                    # Try to parse as JSON array
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return parsed
                    else:
                        return [v]  # Single key as string
                except json.JSONDecodeError:
                    return [v]  # Treat as single key
            else:
                return []
        elif isinstance(v, list):
            return v
        else:
            return []
    
    @field_validator('LLM_TOKENS_PER_MINUTE')
    @classmethod
    def validate_token_limits(cls, v):
        if v <= 0 or v > 10000:
            raise ValueError('LLM_TOKENS_PER_MINUTE must be between 1 and 10000')
        return v
    
    @field_validator('SUPPORTED_LANGUAGES')
    @classmethod
    def validate_languages(cls, v):
        if not v or len(v) < 1:
            raise ValueError('At least one language must be supported')
        return v
    
    # ✅ ADDED: Property to combine API keys intelligently
    @property
    def all_groq_api_keys(self) -> List[str]:
        """Get all GROQ API keys (single + list)"""
        keys = []
        
        # Add single key if available
        if self.GROQ_API_KEY and self.GROQ_API_KEY.strip():
            keys.append(self.GROQ_API_KEY.strip())
        
        # Add keys from list
        if self.GROQ_API_KEYS:
            for key in self.GROQ_API_KEYS:
                key_str = str(key).strip()
                if key_str and key_str not in keys:
                    keys.append(key_str)
        
        return keys
    
    def get_database_url(self) -> str:
        """Get complete database URL"""
        return f"mysql+pymysql://{self.TIDB_USER}:{self.TIDB_PASSWORD}@{self.TIDB_HOST}:{self.TIDB_PORT}/{self.TIDB_DATABASE}?ssl_verify_cert=true&charset=utf8mb4"
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "api_keys": self.all_groq_api_keys,  # Use the combined property
            "tokens_per_minute": self.LLM_TOKENS_PER_MINUTE,
            "requests_per_minute": self.LLM_REQUESTS_PER_MINUTE,
            "timeout": self.LLM_REQUEST_TIMEOUT,
            "max_retries": self.LLM_MAX_RETRIES,
            "batch_size": self.LLM_BATCH_SIZE
        }
    
    def get_kimi_config(self) -> Dict[str, Any]:
        """Get Kimi AI configuration"""
        return {
            "api_key": self.KIMI_API_KEY,
            "model": self.KIMI_MODEL,
            "base_url": self.KIMI_BASE_URL,
            "enabled": self.ENABLE_KIMI_INTEGRATION
        }
    
    def get_multilingual_config(self) -> Dict[str, Any]:
        """Get multilingual configuration"""
        return {
            "default_language": self.DEFAULT_LANGUAGE,
            "supported_languages": self.SUPPORTED_LANGUAGES,
            "priority_languages": self.RESEARCH_PRIORITY_LANGUAGES,
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_dimension": self.EMBEDDING_DIMENSION,
            "enabled": self.ENABLE_MULTILINGUAL_SEARCH
        }
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get multi-agent configuration"""
        return {
            "max_concurrent_agents": self.MAX_CONCURRENT_AGENTS,
            "task_timeout": self.AGENT_TASK_TIMEOUT,
            "paper_analysis_concurrent": self.PAPER_ANALYSIS_MAX_CONCURRENT,
            "relationship_analysis_concurrent": self.RELATIONSHIP_ANALYSIS_MAX_CONCURRENT,
            "coordinator_timeout": self.COORDINATOR_WORKFLOW_TIMEOUT,
            "enabled": self.ENABLE_AI_AGENTS
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            "enable_caching": self.ENABLE_CACHING,
            "cache_ttl": self.CACHE_TTL_DEFAULT,
            "max_cache_size": self.MAX_CACHE_SIZE,
            "enable_monitoring": self.ENABLE_PERFORMANCE_MONITORING,
            "detailed_logging": self.ENABLE_DETAILED_LOGGING,
            "optimization_enabled": self.ENABLE_PERFORMANCE_OPTIMIZATION
        }
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        feature_flags = {
            "multilingual_search": self.ENABLE_MULTILINGUAL_SEARCH,
            "vector_search": self.ENABLE_VECTOR_SEARCH,
            "ai_agents": self.ENABLE_AI_AGENTS,
            "kimi_integration": self.ENABLE_KIMI_INTEGRATION,
            "relationship_analysis": self.ENABLE_RELATIONSHIP_ANALYSIS,
            "graph_visualization": self.ENABLE_GRAPH_VISUALIZATION,
            "performance_optimization": self.ENABLE_PERFORMANCE_OPTIMIZATION
        }
        return feature_flags.get(feature.lower(), False)

    
# Global settings instance
settings = Settings()


# Logging configuration
def configure_logging():
    """Configure application logging"""
    log_level = getattr(logging, settings.LOG_LEVEL.upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific loggers
    if settings.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} - Logging configured")
    logger.info(f"⚙️ Configuration loaded - Debug: {settings.DEBUG}")
    logger.info(f"🔑 GROQ API Keys configured: {len(settings.all_groq_api_keys)}")


# Initialize logging
configure_logging()
