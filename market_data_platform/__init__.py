"""
Market Data Platform - Core Module
Main entry point for the platform
"""

__version__ = "1.0.0"
__author__ = "Market Data Platform Team"
__license__ = "MIT"

from .core import GatewayManager, SessionManager
from .api import APIServer, WebSocketAPI
from .cli import CLIManager
from .utils import setup_logging, load_config

__all__ = [
    "GatewayManager",
    "SessionManager",
    "APIServer",
    "WebSocketAPI",
    "CLIManager",
    "setup_logging",
    "load_config",
]
