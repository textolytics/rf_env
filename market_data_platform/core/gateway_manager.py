"""
Gateway Manager - Core component for managing all gateways
"""

import logging
from typing import Dict, Optional, List
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class GatewayType(Enum):
    """Supported gateway types"""
    PYTHON = "python"
    GO = "go"
    RUST = "rust"
    NATIVE = "native"


class BaseGateway(ABC):
    """Base class for all gateway implementations"""
    
    def __init__(self, name: str, gateway_type: GatewayType):
        self.name = name
        self.gateway_type = gateway_type
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the gateway"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the gateway"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict:
        """Get gateway status"""
        pass


class GatewayManager:
    """Manages all system gateways"""
    
    def __init__(self):
        self.gateways: Dict[str, BaseGateway] = {}
        self.active_gateway: Optional[str] = None
        logger.info("GatewayManager initialized")
    
    def register_gateway(self, name: str, gateway: BaseGateway) -> None:
        """Register a new gateway"""
        self.gateways[name] = gateway
        logger.info(f"Gateway registered: {name}")
    
    def unregister_gateway(self, name: str) -> bool:
        """Unregister a gateway"""
        if name in self.gateways:
            del self.gateways[name]
            logger.info(f"Gateway unregistered: {name}")
            return True
        return False
    
    def get_gateway(self, name: str) -> Optional[BaseGateway]:
        """Get a specific gateway"""
        return self.gateways.get(name)
    
    def get_all_gateways(self) -> Dict[str, BaseGateway]:
        """Get all registered gateways"""
        return dict(self.gateways)
    
    def connect_gateway(self, name: str) -> bool:
        """Connect to a specific gateway"""
        gateway = self.gateways.get(name)
        if not gateway:
            logger.warning(f"Gateway not found: {name}")
            return False
        
        if gateway.connect():
            self.active_gateway = name
            logger.info(f"Connected to gateway: {name}")
            return True
        return False
    
    def disconnect_gateway(self, name: str) -> bool:
        """Disconnect from a specific gateway"""
        gateway = self.gateways.get(name)
        if not gateway:
            return False
        
        if gateway.disconnect():
            if self.active_gateway == name:
                self.active_gateway = None
            logger.info(f"Disconnected from gateway: {name}")
            return True
        return False
    
    def get_active_gateway(self) -> Optional[BaseGateway]:
        """Get the currently active gateway"""
        if self.active_gateway:
            return self.gateways.get(self.active_gateway)
        return None
