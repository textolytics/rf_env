"""
Pytest Configuration and Shared Fixtures
Coordinate testing across C++, Python, Rust, and Go modules
"""

import pytest
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Generator
import json
import time

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MARKET_DATA_PATH = PROJECT_ROOT / "market_data_platform"
CONNECTIVITY_PATH = MARKET_DATA_PATH / "connectivity"
TESTING_PATH = MARKET_DATA_PATH / "testing"


class TestEnvironment:
    """Global test environment configuration"""
    
    # Module paths
    PYTHON_PATH = CONNECTIVITY_PATH / "python"
    GO_PATH = CONNECTIVITY_PATH / "go"
    RUST_PATH = CONNECTIVITY_PATH / "rust"
    CPP_PATH = CONNECTIVITY_PATH / "cpp"
    
    # Service endpoints
    SERVICE_ENDPOINTS = {
        "gateio_api": "https://api.gateio.ws/api/v4",
        "gateio_ws": "wss://api.gateio.ws/ws/v4/",
        "zmq_pub": "tcp://127.0.0.1:5555",
        "zmq_router": "tcp://127.0.0.1:5559",
        "influxdb": "http://127.0.0.1:8086",
        "grafana": "http://127.0.0.1:3000",
        "redis": "redis://127.0.0.1:6379"
    }
    
    # Test configuration
    TIMEOUT_SECONDS = 30
    CONNECTION_TIMEOUT = 5
    OPERATION_TIMEOUT = 30


# =============================================================================
# PYTEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Register pytest markers"""
    markers = [
        "python: Python module tests",
        "cpp: C++ module tests",
        "rust: Rust module tests",
        "go: Go module tests",
        "zmq: ZMQ bus integration tests",
        "dataflow: End-to-end data flow tests",
        "performance: Performance benchmark tests",
        "gateio: Gate.io connectivity tests",
        "integration: Integration tests",
        "slow: Slow running tests",
        "unit: Unit tests",
    ]
    
    for marker_name, marker_desc in markers:
        config.addinivalue_line("markers", f"{marker_name}: {marker_desc}")


# =============================================================================
# SESSION-SCOPED FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def test_environment():
    """Provide global test environment configuration"""
    return TestEnvironment()


@pytest.fixture(scope="session")
def project_structure():
    """Verify project structure before tests"""
    structure = {
        "python": TestEnvironment.PYTHON_PATH.exists(),
        "go": TestEnvironment.GO_PATH.exists(),
        "rust": TestEnvironment.RUST_PATH.exists(),
        "cpp": TestEnvironment.CPP_PATH.exists(),
        "testing": TESTING_PATH.exists(),
    }
    
    return structure


# =============================================================================
# FUNCTION-SCOPED FIXTURES
# =============================================================================

@pytest.fixture
def gateio_config():
    """Gate.io test configuration"""
    return {
        "symbols": ["ETH_USDT", "BTC_USDT", "BNB_USDT", "BNBBTC"],
        "intervals": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "api_url": "https://api.gateio.ws/api/v4",
        "ws_url": "wss://api.gateio.ws/ws/v4/",
        "timeout": 10,
        "retry_count": 3,
        "retry_delay": 1.0
    }


@pytest.fixture
def zmq_config():
    """ZMQ configuration for tests"""
    return {
        "python_host": "127.0.0.1",
        "python_port": 5555,
        "go_host": "127.0.0.1",
        "go_port": 5556,
        "rust_host": "127.0.0.1",
        "rust_port": 5557,
        "cpp_host": "127.0.0.1",
        "cpp_port": 5558,
        "router_host": "127.0.0.1",
        "router_port": 5559,
        "timeout": 5,
        "context_timeout": 10
    }


@pytest.fixture
def subprocess_runner():
    """Execute subprocess commands with timeout and error handling"""
    def run_command(
        cmd: str, 
        timeout: int = 30,
        cwd: Path = None,
        check: bool = False
    ) -> Tuple[int, str, str]:
        """
        Execute command and return (returncode, stdout, stderr)
        """
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                timeout=timeout,
                text=True,
                cwd=cwd
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timeout after {timeout}s"
        except Exception as e:
            return -2, "", f"Error executing command: {str(e)}"
    
    return run_command


@pytest.fixture
def performance_timer():
    """Fixture for measuring performance"""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.measurements = {}
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, *args):
            self.end_time = time.time()
        
        @property
        def elapsed(self) -> float:
            """Get elapsed time in seconds"""
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0.0
        
        def mark(self, label: str):
            """Mark checkpoint"""
            self.measurements[label] = time.time() - self.start_time
        
        def report(self) -> Dict[str, float]:
            """Get all measurements"""
            return self.measurements.copy()
    
    return PerformanceTimer


@pytest.fixture
def test_data_factory():
    """Factory for creating test data structures"""
    
    def create_ohlc_data(symbol: str, timeframe: str, count: int = 1) -> list:
        """Create OHLC data samples"""
        data = []
        for i in range(count):
            data.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": 1234567890 + (i * 60),
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 102.0 + i,
                "volume": 1000.0 + i * 100
            })
        return data if count > 1 else data[0]
    
    def create_ticker_data(symbol: str) -> Dict[str, Any]:
        """Create ticker data sample"""
        return {
            "symbol": symbol,
            "last": 1234.56,
            "bid": 1234.50,
            "ask": 1234.60,
            "bid_volume": 100.0,
            "ask_volume": 95.0,
            "volume_24h": 100000.0,
            "change_percent": 2.5,
            "high_24h": 1250.00,
            "low_24h": 1200.00,
        }
    
    def create_zmq_message(topic: str, source: str, data: Dict) -> str:
        """Create ZMQ message in JSON format"""
        message = {
            "topic": topic,
            "type": "data",
            "source": source,
            "ts": int(time.time()),
            "data": data
        }
        return json.dumps(message)
    
    return {
        "create_ohlc": create_ohlc_data,
        "create_ticker": create_ticker_data,
        "create_zmq_message": create_zmq_message
    }


# =============================================================================
# AUTOUSE FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_test_state():
    """Reset state before each test"""
    yield
    # Cleanup after test
    pass


@pytest.fixture(autouse=True)
def log_test_info(request):
    """Log test information"""
    test_name = request.node.name
    markers = [m.name for m in request.node.iter_markers()]
    
    # Log start
    print(f"\\n📝 Starting: {test_name}")
    if markers:
        print(f"   Tags: {', '.join(markers)}")
    
    yield
    
    # Log completion
    print(f"✓ Completed: {test_name}")


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify collected tests"""
    
    # Auto-mark tests based on file location
    for item in items:
        fspath = str(item.fspath)
        
        # Mark by module
        if "test_python" in fspath or "python_test" in fspath:
            item.add_marker(pytest.mark.python)
        elif "test_go" in fspath or "go_test" in fspath:
            item.add_marker(pytest.mark.go)
        elif "test_rust" in fspath or "rust_test" in fspath:
            item.add_marker(pytest.mark.rust)
        elif "test_cpp" in fspath or "cpp_test" in fspath:
            item.add_marker(pytest.mark.cpp)
        
        # Mark by test type
        if "integration" in fspath:
            item.add_marker(pytest.mark.integration)
        elif "performance" in fspath or "bench" in fspath:
            item.add_marker(pytest.mark.performance)


# =============================================================================
# COMMAND-LINE OPTIONS
# =============================================================================

def pytest_addoption(parser):
    """Add custom command-line options"""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests"
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


def pytest_configure_custom(config):
    """Custom pytest configuration"""
    if not config.option.integration:
        config.addinivalue_line("markers", "integration: skip if --integration not given")
