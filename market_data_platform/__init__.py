"""
Market Data Platform - Core Module
Main entry point for the platform
"""

__version__ = "1.0.0"
__author__ = "Market Data Platform Team"
__license__ = "MIT"

from .core import GatewayManager
try:
    from .api import APIServer, WebSocketAPI
except ImportError:
    APIServer = None
    WebSocketAPI = None
try:
    from .cli import CLIManager
except ImportError:
    CLIManager = None
try:
    from .utils import setup_logging, load_config
except ImportError:
    setup_logging = None
    load_config = None

__all__ = [
    "GatewayManager",
    "setup_logging",
    "load_config",
]
