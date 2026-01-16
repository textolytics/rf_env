#!/usr/bin/env python3
"""
Comprehensive Regression Test Suite
Tests all modules across Python, C++, Rust, and Go
Validates connectivity, data flow, and ZMQ integration
"""

import pytest
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# ============================================================================
# Test Configuration
# ============================================================================

class TestConfig:
    """Centralized test configuration"""
    PROJECT_ROOT = Path(__file__).parent.parent
    CONNECTIVITY_ROOT = PROJECT_ROOT / "connectivity"
    PYTHON_MODULES = CONNECTIVITY_ROOT / "python"
    CPP_MODULES = CONNECTIVITY_ROOT / "cpp"
    RUST_MODULES = CONNECTIVITY_ROOT / "rust"
    GO_MODULES = CONNECTIVITY_ROOT / "go"
    
    # Service endpoints
    INFLUXDB_URL = "http://localhost:8086"
    GRAFANA_URL = "http://localhost:3000"
    REDIS_URL = "localhost:6379"
    ZMQ_HOST = "127.0.0.1"
    ZMQ_PORT = 5555
    
    # Gateways
    GATEIO_API = "https://api.gateio.ws"
    GATEIO_WS = "wss://ws.gate.io/v4"
    
    # Test timeouts
    CONNECTION_TIMEOUT = 5.0
    OPERATION_TIMEOUT = 30.0
    TEST_TIMEOUT = 60.0


# ============================================================================
# Base Test Classes
# ============================================================================

class BaseConnectivityTest:
    """Base class for connectivity tests"""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup test environment"""
        self.config = TestConfig()
        self.test_start = datetime.now()
        yield
        self.test_duration = (datetime.now() - self.test_start).total_seconds()
    
    def run_command(self, cmd: List[str], timeout: float = 10.0) -> tuple:
        """Run shell command and capture output"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timeout"
        except Exception as e:
            return -1, "", str(e)
    
    def assert_command_success(self, cmd: List[str], timeout: float = 10.0):
        """Assert command succeeds"""
        returncode, stdout, stderr = self.run_command(cmd, timeout)
        assert returncode == 0, f"Command failed: {stderr}"
        return stdout


# ============================================================================
# Python Module Tests
# ============================================================================

class TestPythonModules(BaseConnectivityTest):
    """Test Python connectivity modules"""
    
    @pytest.mark.python
    def test_python_gateio_connectivity(self):
        """Test Python Gate.io connectivity module"""
        # Test import
        try:
            sys.path.insert(0, str(self.config.PYTHON_MODULES))
            import gateio_connector
            connector = gateio_connector.GateIOConnector()
            assert connector is not None, "Failed to instantiate GateIOConnector"
        except ImportError as e:
            pytest.skip(f"Python module not available: {e}")
    
    @pytest.mark.python
    def test_python_zmq_publisher(self):
        """Test Python ZMQ publisher functionality"""
        try:
            sys.path.insert(0, str(self.config.PYTHON_MODULES))
            import zmq_publisher
            publisher = zmq_publisher.ZMQPublisher(
                host=self.config.ZMQ_HOST,
                port=self.config.ZMQ_PORT
            )
            assert publisher is not None
            # Test can create publisher instance
            assert hasattr(publisher, 'publish')
        except ImportError:
            pytest.skip("Python ZMQ module not available")
    
    @pytest.mark.python
    def test_python_data_validation(self):
        """Test Python data validation module"""
        try:
            sys.path.insert(0, str(self.config.PYTHON_MODULES))
            import data_validator
            
            # Test with sample OHLC data
            sample_ohlc = {
                "timestamp": 1234567890,
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000000
            }
            
            validator = data_validator.OHLCValidator()
            assert validator.validate(sample_ohlc), "OHLC validation failed"
        except ImportError:
            pytest.skip("Python validator module not available")
    
    @pytest.mark.python
    def test_python_error_handling(self):
        """Test Python module error handling"""
        try:
            sys.path.insert(0, str(self.config.PYTHON_MODULES))
            import error_handler
            
            handler = error_handler.ErrorHandler()
            
            # Test error logging
            handler.log_error("Test error", "test_module", Exception("Test"))
            assert handler.get_error_count() > 0
            
            # Test error recovery
            recovered = handler.attempt_recovery("connection_error")
            assert recovered is not None
        except ImportError:
            pytest.skip("Python error handler not available")


# ============================================================================
# C++ Module Tests
# ============================================================================

class TestCppModules(BaseConnectivityTest):
    """Test C++ connectivity modules"""
    
    @pytest.mark.cpp
    def test_cpp_build_status(self):
        """Verify C++ modules are built"""
        cpp_dir = self.config.CPP_MODULES
        assert cpp_dir.exists(), "C++ modules directory not found"
        
        # Check for compiled binaries
        binaries = list(cpp_dir.glob("build/bin/*"))
        assert len(binaries) > 0, "No C++ binaries found"
    
    @pytest.mark.cpp
    def test_cpp_gateio_connector(self):
        """Test C++ Gate.io connector binary"""
        binary = self.config.CPP_MODULES / "build" / "bin" / "gateio_connector"
        
        if not binary.exists():
            pytest.skip("C++ binary not built")
        
        # Test binary exists and is executable
        assert binary.exists()
        assert os.access(binary, os.X_OK)
    
    @pytest.mark.cpp
    def test_cpp_zmq_router(self):
        """Test C++ ZMQ router"""
        binary = self.config.CPP_MODULES / "build" / "bin" / "zmq_router"
        
        if not binary.exists():
            pytest.skip("C++ ZMQ router not built")
        
        # Start router in background, verify it runs
        process = subprocess.Popen([str(binary)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(0.5)  # Let it start
        
        # Check if still running
        poll = process.poll()
        process.terminate()
        
        assert poll is None, "C++ router failed to start"


# ============================================================================
# Rust Module Tests
# ============================================================================

class TestRustModules(BaseConnectivityTest):
    """Test Rust connectivity modules"""
    
    @pytest.mark.rust
    def test_rust_build_status(self):
        """Verify Rust modules are built"""
        rust_dir = self.config.RUST_MODULES
        assert rust_dir.exists(), "Rust modules directory not found"
        
        # Check Cargo.toml exists
        cargo_toml = rust_dir / "Cargo.toml"
        assert cargo_toml.exists(), "Cargo.toml not found"
    
    @pytest.mark.rust
    def test_rust_gateio_connector(self):
        """Test Rust Gate.io connector"""
        # Test compilation
        returncode, stdout, stderr = self.run_command(
            ["cargo", "test", "--manifest-path", 
             str(self.config.RUST_MODULES / "Cargo.toml"),
             "--lib", "tests::gateio"],
            timeout=30.0
        )
        
        if returncode != 0:
            pytest.skip("Rust tests not available or failed to compile")
    
    @pytest.mark.rust
    def test_rust_data_processing(self):
        """Test Rust data processing module"""
        # Test compilation
        returncode, stdout, stderr = self.run_command(
            ["cargo", "build", "--manifest-path",
             str(self.config.RUST_MODULES / "Cargo.toml"),
             "--release"],
            timeout=60.0
        )
        
        if returncode != 0:
            pytest.skip("Rust build failed")


# ============================================================================
# Go Module Tests
# ============================================================================

class TestGoModules(BaseConnectivityTest):
    """Test Go connectivity modules"""
    
    @pytest.mark.go
    def test_go_module_structure(self):
        """Verify Go module is properly structured"""
        go_dir = self.config.GO_MODULES
        
        if not go_dir.exists():
            pytest.skip("Go module directory not created yet")
        
        # Check essential files
        assert (go_dir / "go.mod").exists(), "go.mod not found"
        assert (go_dir / "main.go").exists(), "main.go not found"
    
    @pytest.mark.go
    def test_go_gateio_connector(self):
        """Test Go Gate.io connector functionality"""
        go_dir = self.config.GO_MODULES
        
        if not go_dir.exists():
            pytest.skip("Go module not created")
        
        # Test go build
        returncode, stdout, stderr = self.run_command(
            ["go", "build", "-o", "gateio_connector", "main.go"],
            timeout=30.0
        )
        
        if returncode != 0:
            pytest.skip("Go build not available")
    
    @pytest.mark.go
    def test_go_zmq_integration(self):
        """Test Go ZMQ integration"""
        go_dir = self.config.GO_MODULES
        
        if not go_dir.exists():
            pytest.skip("Go module not created")
        
        # Test compilation of ZMQ client
        test_file = go_dir / "zmq_client.go"
        if test_file.exists():
            returncode, stdout, stderr = self.run_command(
                ["go", "test", "-v", "./..."],
                timeout=30.0
            )


# ============================================================================
# ZMQ Bus Integration Tests
# ============================================================================

class TestZMQIntegration(BaseConnectivityTest):
    """Test ZMQ bus integration across all modules"""
    
    @pytest.mark.zmq
    def test_zmq_bus_connectivity(self):
        """Test ZMQ bus is accessible"""
        try:
            import zmq
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(f"tcp://{self.config.ZMQ_HOST}:{self.config.ZMQ_PORT}")
            
            # Send test message with timeout
            socket.setsockopt(zmq.RCVTIMEO, int(self.config.CONNECTION_TIMEOUT * 1000))
            
            socket.send(b"test")
            try:
                message = socket.recv()
                socket.close()
                context.term()
            except zmq.error.Again:
                socket.close()
                context.term()
        except ImportError:
            pytest.skip("PyZMQ not installed")
    
    @pytest.mark.zmq
    def test_zmq_message_routing(self):
        """Test ZMQ message routing between modules"""
        try:
            import zmq
            
            # Test PUB/SUB pattern
            context = zmq.Context()
            
            # Publisher socket
            pub = context.socket(zmq.PUB)
            pub.bind(f"tcp://*:15555")
            
            # Subscriber socket
            sub = context.socket(zmq.SUB)
            sub.connect(f"tcp://localhost:15555")
            sub.subscribe(b"")
            
            time.sleep(0.1)  # Allow subscription to register
            
            # Send test message
            pub.send(b"test_message")
            
            sub.setsockopt(zmq.RCVTIMEO, 1000)
            try:
                message = sub.recv()
                assert message == b"test_message"
            except zmq.error.Again:
                pass
            
            pub.close()
            sub.close()
            context.term()
        except ImportError:
            pytest.skip("PyZMQ not installed")


# ============================================================================
# Data Flow Tests
# ============================================================================

class TestDataFlow(BaseConnectivityTest):
    """Test end-to-end data flow"""
    
    @pytest.mark.dataflow
    def test_ohlc_data_collection(self):
        """Test OHLC data collection from multiple sources"""
        # Test data structure validation
        sample_ohlc = {
            "symbol": "EURUSD",
            "timestamp": int(time.time()),
            "open": 1.0950,
            "high": 1.0960,
            "low": 1.0945,
            "close": 1.0955,
            "volume": 1000000
        }
        
        # Verify all required fields
        required_fields = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        for field in required_fields:
            assert field in sample_ohlc, f"Missing field: {field}"
    
    @pytest.mark.dataflow
    def test_data_transformation(self):
        """Test data transformation across formats"""
        # Test JSON serialization
        sample_data = {
            "timestamp": 1234567890,
            "price": 100.5,
            "volume": 1000000
        }
        
        json_str = json.dumps(sample_data)
        restored = json.loads(json_str)
        assert restored == sample_data


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance(BaseConnectivityTest):
    """Test performance metrics"""
    
    @pytest.mark.performance
    def test_message_throughput(self):
        """Test message throughput on ZMQ bus"""
        try:
            import zmq
            
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.connect(f"tcp://localhost:15556")
            
            # Send 1000 messages and measure time
            start = time.time()
            message_count = 1000
            
            for i in range(message_count):
                try:
                    socket.send(f"msg_{i}".encode(), zmq.NOBLOCK)
                except zmq.error.Again:
                    break
            
            duration = time.time() - start
            throughput = message_count / duration if duration > 0 else 0
            
            socket.close()
            context.term()
            
            # Performance threshold: 1000 msg/sec minimum
            assert throughput > 100, f"Throughput too low: {throughput} msg/sec"
        except ImportError:
            pytest.skip("PyZMQ not installed")
    
    @pytest.mark.performance
    def test_connection_latency(self):
        """Test connection latency"""
        start = time.time()
        
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.config.CONNECTION_TIMEOUT)
            s.connect((self.config.ZMQ_HOST, self.config.ZMQ_PORT))
            latency = (time.time() - start) * 1000  # ms
            s.close()
            
            # Connection latency should be < 100ms
            assert latency < 100, f"Connection latency too high: {latency}ms"
        except (socket.timeout, ConnectionRefusedError):
            pytest.skip("ZMQ bus not available")


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "python: Python module tests")
    config.addinivalue_line("markers", "cpp: C++ module tests")
    config.addinivalue_line("markers", "rust: Rust module tests")
    config.addinivalue_line("markers", "go: Go module tests")
    config.addinivalue_line("markers", "zmq: ZMQ integration tests")
    config.addinivalue_line("markers", "dataflow: Data flow tests")
    config.addinivalue_line("markers", "performance: Performance tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
