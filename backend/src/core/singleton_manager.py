import logging
import time
import asyncio
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SingletonManager:
    """Manage singleton instances to prevent reinitialization with enhanced functionality"""
    
    _instances = {}
    _initialization_time = time.time()
    _system_stats = {
        'instances_created': 0,
        'instances_cleared': 0,
        'total_operations': 0,
        'last_health_check': None
    }
    
    def __init__(self):
        """Initialize the singleton manager instance"""
        self.initialized = False
        self.health_status = "starting"
        logger.info("🔧 SingletonManager instance created")
    
    async def initialize(self):
        """Initialize the singleton manager asynchronously"""
        try:
            logger.info("🚀 Initializing SingletonManager...")
            
            # Simulate async initialization
            await asyncio.sleep(0.1)
            
            self.initialized = True
            self.health_status = "healthy"
            self._system_stats['last_health_check'] = time.time()
            
            logger.info("✅ SingletonManager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ SingletonManager initialization failed: {e}")
            self.health_status = "unhealthy"
            raise
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        try:
            current_time = time.time()
            uptime = current_time - self._initialization_time
            
            # Update last health check
            self._system_stats['last_health_check'] = current_time
            
            return {
                "overall_status": self.health_status,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time)),
                "components": {
                    "singleton_manager": self.health_status,
                    "instance_registry": "healthy" if self._instances else "empty",
                    "statistics_tracker": "healthy"
                },
                "performance": {
                    "uptime_seconds": uptime,
                    "uptime_hours": uptime / 3600,
                    "total_instances": len(self._instances),
                    "instances_created": self._system_stats['instances_created'],
                    "instances_cleared": self._system_stats['instances_cleared'],
                    "total_operations": self._system_stats['total_operations']
                },
                "health_check_time": current_time
            }
            
        except Exception as e:
            logger.error(f"❌ System health check failed: {e}")
            return {
                "overall_status": "error",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": str(e),
                "components": {"error": "health_check_failed"}
            }
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            current_time = time.time()
            uptime = current_time - self._initialization_time
            
            return {
                "system_info": {
                    "singleton_manager_version": "2.0.0",
                    "initialization_time": self._initialization_time,
                    "uptime_seconds": uptime,
                    "current_status": self.health_status,
                    "initialized": self.initialized
                },
                "instance_registry": {
                    "total_instances": len(self._instances),
                    "instance_names": list(self._instances.keys()),
                    "registry_size": len(self._instances)
                },
                "operations_stats": {
                    **self._system_stats,
                    "operations_per_hour": self._system_stats['total_operations'] / max(uptime / 3600, 0.01)
                },
                "performance_metrics": {
                    "memory_efficiency": "high" if len(self._instances) < 50 else "medium",
                    "registry_health": "optimal" if len(self._instances) < 100 else "monitoring",
                    "last_operation": self._system_stats.get('last_operation', 'none')
                },
                "capabilities": [
                    "singleton_management",
                    "instance_registry", 
                    "health_monitoring",
                    "statistics_tracking",
                    "async_operations"
                ],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
        except Exception as e:
            logger.error(f"❌ Comprehensive stats failed: {e}")
            return {
                "error": str(e),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": "error"
            }
    
    async def shutdown(self):
        """Shutdown the singleton manager and cleanup resources"""
        try:
            logger.info("🔒 Shutting down SingletonManager...")
            
            # Clear all instances
            instance_count = len(self._instances)
            self.clear_all()
            
            # Update status
            self.health_status = "shutdown"
            self.initialized = False
            
            # Update stats
            self._system_stats['total_operations'] += 1
            self._system_stats['last_operation'] = 'shutdown'
            
            logger.info(f"✅ SingletonManager shutdown completed - cleared {instance_count} instances")
            
        except Exception as e:
            logger.error(f"❌ SingletonManager shutdown failed: {e}")
    
    @classmethod
    def get_instance(cls, component_name: str, factory_func, *args, **kwargs):
        """Get singleton instance of component"""
        try:
            if component_name not in cls._instances:
                cls._instances[component_name] = factory_func(*args, **kwargs)
                cls._system_stats['instances_created'] += 1
                cls._system_stats['total_operations'] += 1
                cls._system_stats['last_operation'] = f'created_{component_name}'
                logger.info(f"✅ Created singleton: {component_name}")
            else:
                logger.info(f"♻️ Retrieved existing singleton: {component_name}")
                
            cls._system_stats['total_operations'] += 1
            return cls._instances[component_name]
            
        except Exception as e:
            logger.error(f"❌ Failed to create singleton {component_name}: {e}")
            cls._system_stats['total_operations'] += 1
            raise
    
    @classmethod
    def clear_instance(cls, component_name: str):
        """Clear singleton instance"""
        try:
            if component_name in cls._instances:
                del cls._instances[component_name]
                cls._system_stats['instances_cleared'] += 1
                cls._system_stats['total_operations'] += 1
                cls._system_stats['last_operation'] = f'cleared_{component_name}'
                logger.info(f"🗑️ Cleared singleton: {component_name}")
            else:
                logger.warning(f"⚠️ Attempted to clear non-existent singleton: {component_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to clear singleton {component_name}: {e}")
    
    @classmethod
    def clear_all(cls):
        """Clear all singleton instances"""
        try:
            instance_count = len(cls._instances)
            cls._instances.clear()
            cls._system_stats['instances_cleared'] += instance_count
            cls._system_stats['total_operations'] += 1
            cls._system_stats['last_operation'] = 'clear_all'
            logger.info(f"🗑️ Cleared all singletons ({instance_count} instances)")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear all singletons: {e}")
    
    @classmethod
    def get_registry_info(cls) -> Dict[str, Any]:
        """Get registry information"""
        return {
            "total_instances": len(cls._instances),
            "instance_names": list(cls._instances.keys()),
            "statistics": cls._system_stats.copy(),
            "registry_healthy": True
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of this singleton manager instance"""
        return {
            "initialized": self.initialized,
            "health_status": self.health_status,
            "total_instances": len(self._instances),
            "uptime": time.time() - self._initialization_time
        }
