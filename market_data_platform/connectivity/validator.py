"""
Connectivity Validator - Validates service connectivity and health
Provides health checks, connectivity tests, and status reporting
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import httpx
import redis
import psycopg2
from psycopg2 import sql

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNREACHABLE = "unreachable"


@dataclass
class ServiceHealth:
    """Service health status"""
    name: str
    status: ServiceStatus
    response_time: float
    details: Dict
    timestamp: float


class ConnectivityValidator:
    """Validates connectivity for all services"""
    
    SERVICES = {
        "database_postgres": {
            "type": "postgres",
            "host": "localhost",
            "port": 5432,
            "user": "mdp_user",
            "password": "mdp_password",
            "database": "market_data"
        },
        "cache_redis": {
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "storage_influxdb": {
            "type": "http",
            "url": "http://localhost:8086/health"
        },
        "monitoring_prometheus": {
            "type": "http",
            "url": "http://localhost:9090/-/healthy"
        },
        "monitoring_grafana": {
            "type": "http",
            "url": "http://localhost:3000/api/health"
        },
        "api_python": {
            "type": "http",
            "url": "http://localhost:8000/health"
        },
        "gateway_go": {
            "type": "http",
            "url": "http://localhost:8080/health"
        },
        "messaging_zmq_publisher": {
            "type": "zmq",
            "endpoint": "tcp://127.0.0.1:5555"
        },
        "messaging_zmq_subscriber": {
            "type": "zmq",
            "endpoint": "tcp://127.0.0.1:5556"
        }
    }
    
    def __init__(self, timeout: float = 5.0):
        """Initialize validator with timeout"""
        self.timeout = timeout
        self.health_history: Dict[str, List[ServiceHealth]] = {}
    
    async def validate_all(self) -> Dict[str, ServiceHealth]:
        """Validate all services"""
        results = {}
        tasks = []
        
        for service_name, config in self.SERVICES.items():
            tasks.append(self._validate_service(service_name, config))
        
        validations = await asyncio.gather(*tasks, return_exceptions=True)
        
        for service_name, health in zip(self.SERVICES.keys(), validations):
            if isinstance(health, Exception):
                results[service_name] = ServiceHealth(
                    name=service_name,
                    status=ServiceStatus.UNREACHABLE,
                    response_time=0,
                    details={"error": str(health)},
                    timestamp=time.time()
                )
            else:
                results[service_name] = health
                # Keep history
                if service_name not in self.health_history:
                    self.health_history[service_name] = []
                self.health_history[service_name].append(health)
        
        return results
    
    async def _validate_service(
        self,
        service_name: str,
        config: Dict
    ) -> ServiceHealth:
        """Validate a single service"""
        start_time = time.time()
        service_type = config.get("type")
        
        try:
            if service_type == "http":
                health = await self._validate_http(service_name, config)
            elif service_type == "postgres":
                health = await self._validate_postgres(service_name, config)
            elif service_type == "redis":
                health = await self._validate_redis(service_name, config)
            elif service_type == "zmq":
                health = await self._validate_zmq(service_name, config)
            else:
                health = ServiceHealth(
                    name=service_name,
                    status=ServiceStatus.UNHEALTHY,
                    response_time=time.time() - start_time,
                    details={"error": f"Unknown service type: {service_type}"},
                    timestamp=time.time()
                )
            
            return health
        
        except asyncio.TimeoutError:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.UNREACHABLE,
                response_time=time.time() - start_time,
                details={"error": "Connection timeout"},
                timestamp=time.time()
            )
        except Exception as e:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.UNREACHABLE,
                response_time=time.time() - start_time,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    async def _validate_http(
        self,
        service_name: str,
        config: Dict
    ) -> ServiceHealth:
        """Validate HTTP service"""
        url = config.get("url")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                status = ServiceStatus.HEALTHY
            else:
                status = ServiceStatus.DEGRADED
            
            return ServiceHealth(
                name=service_name,
                status=status,
                response_time=response.elapsed.total_seconds(),
                details={
                    "status_code": response.status_code,
                    "url": url
                },
                timestamp=time.time()
            )
    
    async def _validate_postgres(
        self,
        service_name: str,
        config: Dict
    ) -> ServiceHealth:
        """Validate PostgreSQL database"""
        try:
            conn = psycopg2.connect(
                host=config.get("host"),
                port=config.get("port"),
                user=config.get("user"),
                password=config.get("password"),
                database=config.get("database"),
                connect_timeout=int(self.timeout)
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.HEALTHY,
                response_time=0,
                details={"connection": "successful"},
                timestamp=time.time()
            )
        
        except Exception as e:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.UNREACHABLE,
                response_time=0,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    async def _validate_redis(
        self,
        service_name: str,
        config: Dict
    ) -> ServiceHealth:
        """Validate Redis cache"""
        try:
            r = redis.Redis(
                host=config.get("host"),
                port=config.get("port"),
                db=config.get("db"),
                socket_connect_timeout=int(self.timeout)
            )
            
            response = r.ping()
            
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.HEALTHY if response else ServiceStatus.DEGRADED,
                response_time=0,
                details={"ping": str(response)},
                timestamp=time.time()
            )
        
        except Exception as e:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.UNREACHABLE,
                response_time=0,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    async def _validate_zmq(
        self,
        service_name: str,
        config: Dict
    ) -> ServiceHealth:
        """Validate ZMQ messaging endpoint"""
        endpoint = config.get("endpoint")
        
        try:
            import zmq
            context = zmq.Context()
            socket = context.socket(zmq.DEALER)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(endpoint)
            socket.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
            
            socket.close()
            context.term()
            
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.HEALTHY,
                response_time=0,
                details={"endpoint": endpoint},
                timestamp=time.time()
            )
        
        except Exception as e:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.UNREACHABLE,
                response_time=0,
                details={"error": str(e), "endpoint": endpoint},
                timestamp=time.time()
            )
    
    def get_summary(self, results: Dict[str, ServiceHealth]) -> Dict:
        """Get connectivity summary"""
        healthy_count = sum(
            1 for h in results.values()
            if h.status == ServiceStatus.HEALTHY
        )
        degraded_count = sum(
            1 for h in results.values()
            if h.status == ServiceStatus.DEGRADED
        )
        unhealthy_count = sum(
            1 for h in results.values()
            if h.status == ServiceStatus.UNHEALTHY
        )
        unreachable_count = sum(
            1 for h in results.values()
            if h.status == ServiceStatus.UNREACHABLE
        )
        
        overall_status = ServiceStatus.HEALTHY
        if unreachable_count > 0 or unhealthy_count > 0:
            overall_status = ServiceStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = ServiceStatus.DEGRADED
        
        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "summary": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "unreachable": unreachable_count,
                "total": len(results)
            },
            "services": {
                name: {
                    "status": health.status.value,
                    "response_time": health.response_time,
                    "details": health.details,
                    "timestamp": health.timestamp
                }
                for name, health in results.items()
            }
        }
    
    async def wait_for_service(
        self,
        service_name: str,
        max_retries: int = 30,
        retry_interval: float = 1.0
    ) -> bool:
        """Wait for a specific service to become healthy"""
        config = self.SERVICES.get(service_name)
        if not config:
            raise ValueError(f"Unknown service: {service_name}")
        
        for attempt in range(max_retries):
            health = await self._validate_service(service_name, config)
            if health.status == ServiceStatus.HEALTHY:
                logger.info(f"Service {service_name} is healthy")
                return True
            
            logger.debug(
                f"Service {service_name} not ready (attempt {attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(retry_interval)
        
        logger.error(f"Service {service_name} did not become healthy")
        return False


# Synchronous wrapper for CLI usage
class ConnectivityValidatorSync:
    """Synchronous wrapper for connectivity validator"""
    
    def __init__(self, timeout: float = 5.0):
        self.validator = ConnectivityValidator(timeout)
    
    def validate_all(self) -> Dict[str, ServiceHealth]:
        """Validate all services (synchronous)"""
        return asyncio.run(self.validator.validate_all())
    
    def get_summary(self) -> Dict:
        """Get connectivity summary"""
        results = self.validate_all()
        return self.validator.get_summary(results)
    
    def wait_for_service(
        self,
        service_name: str,
        max_retries: int = 30
    ) -> bool:
        """Wait for service (synchronous)"""
        return asyncio.run(
            self.validator.wait_for_service(service_name, max_retries)
        )


if __name__ == "__main__":
    validator = ConnectivityValidatorSync()
    summary = validator.get_summary()
    print(json.dumps(summary, indent=2, default=str))
