# Market Data Platform - Comprehensive Testing Infrastructure

Complete multi-language regression testing, ZMQ bus integration, and CLI enhancement framework.

## 📋 Overview

This testing infrastructure provides:

- **Multi-Language Testing**: Python, C++, Rust, Go regression tests
- **ZMQ Message Bus**: Unified communication protocol across all modules
- **CLI Enhancements**: Tab completion and interactive keyboard navigation
- **Robot Framework**: End-to-end system testing and task automation
- **Pytest Coordination**: Shared fixtures and configuration for all tests

## 📁 Structure

```
testing/
├── market_data_regression_tests.ipynb   # Main Jupyter notebook with all sections
├── regression_tests.py                  # Python/C++/Rust/Go regression test suite
├── conftest.py                          # Pytest configuration and fixtures
├── deployment.robot                     # Robot Framework deployment tests
├── gateways.robot                       # Robot Framework gateway tests
├── data_operations.robot                # Robot Framework data operation tests
├── multilang_integration.robot          # Robot Framework integration tests
└── keywords/                            # Robot Framework keyword libraries
```

## 🚀 Quick Start

### 1. Setup Build Environment

```bash
# Install dependencies for all languages
cd testing/
python market_data_regression_tests.ipynb  # Cell 1: Setup Build Environment
```

### 2. Run Python Regression Tests

```bash
# Run all Python tests
pytest regression_tests.py -m python -v

# Run specific test class
pytest regression_tests.py::TestPythonGateIOConnectivity -v

# Run with coverage
pytest regression_tests.py -m python --cov=connectivity
```

### 3. Run Go Tests

```bash
# Build and test Go module
cd connectivity/go/
go test -v -timeout 30s

# Run with verbose output
go test -v -run TestGateIOClient
```

### 4. Run Rust Tests

```bash
# Build and test Rust module
cd connectivity/rust/
cargo test --verbose

# Run specific test
cargo test test_gateio_client_initialization
```

### 5. Run C++ Tests

```bash
# Build C++ tests with CMake
cd connectivity/cpp/
mkdir -p build && cd build
cmake ..
make -j4
./market_data_tests

# Run specific test
./market_data_tests --gtest_filter=GateIOConnectorTest.*
```

### 6. Test ZMQ Bus Integration

```bash
# Python ZMQ validation
pytest regression_tests.py::TestPythonZMQIntegration -v

# Full ZMQ routing validation
python -c "
import sys
sys.path.insert(0, '.')
from market_data_regression_tests import ZMQRouter
router = ZMQRouter()
# Test routing
"
```

### 7. Run Robot Framework Tests

```bash
# Deployment tests
robot --include deployment deployment.robot

# Gateway tests
robot --include gateway gateways.robot

# Data operations tests
robot --include data data_operations.robot

# Multi-language integration tests
robot --include multilang multilang_integration.robot

# All tests with output
robot --outputdir results tests/
```

### 8. CLI Tab Completion

```bash
# Bash completion
source market_data_completion.sh

# Then use tab completion
market_data install [TAB]
market_data connect [TAB]
market_data price [TAB]

# Or use Python directly
python -c "
from enhanced_cli import CliCompletionRegistry
completions = CliCompletionRegistry.get_completions('install')
print(completions)
"
```

### 9. CLI Keyboard Navigation

```bash
# Interactive navigation
python market_data_cli.py

# Navigate with arrow keys:
# ← → : Move between command groups
# ↑ ↓ : Select commands
# Ctrl+1-5 : Jump to specific group
# Enter : Execute
# ? : Help
# q : Quit
```

## 📊 Test Categories

### Pytest Markers

```bash
# Run tests by category
pytest -m python          # Python module tests
pytest -m go              # Go module tests
pytest -m rust            # Rust module tests
pytest -m cpp             # C++ module tests
pytest -m zmq             # ZMQ integration tests
pytest -m integration     # Integration tests
pytest -m performance     # Performance tests

# Combine markers
pytest -m "python or go"               # Python OR Go tests
pytest -m "integration and zmq"        # Integration AND ZMQ tests
pytest -m "not slow"                   # All except slow tests
```

### Robot Framework Tags

```bash
# Run by tags
robot --include deployment tests/
robot --include gateway tests/
robot --include data tests/
robot --include multilang tests/

# Exclude tags
robot --exclude slow tests/

# Multiple tags
robot --include "deployment or startup" tests/
```

## 🔧 Configuration

### Pytest Configuration (pytest.ini)

Located in testing directory with markers, output options, and coverage settings.

### Pytest Fixtures (conftest.py)

- `test_environment`: Global configuration
- `gateio_config`: Gate.io test settings
- `zmq_config`: ZMQ endpoints
- `subprocess_runner`: Execute commands
- `performance_timer`: Measure execution time
- `test_data_factory`: Generate test data

### Robot Framework Configuration

Keywords defined in `*.robot` files:
- Deployment keywords
- Gateway keywords
- Data operation keywords
- Monitoring keywords
- ZMQ bus keywords

## 📈 Performance Benchmarks

### Expected Performance

- **Connection Latency**: < 100ms
- **Message Throughput**: > 100 msg/sec
- **Data Fetch Time**: < 1 second per symbol
- **Test Suite Execution**: < 60 seconds total

### Running Performance Tests

```bash
# Run performance tests only
pytest -m performance -v

# With timing report
pytest -m performance -v --durations=10
```

## 🌐 Multi-Language Coordination

### Data Flow

```
Gate.io API (REST/WebSocket)
    ↓
Go Module (REST client + WebSocket)
    ↓
ZMQ Publisher (PUB socket)
    ↓
ZMQ Router (ROUTER socket)
    ↓
Python/Rust/C++ Subscribers (SUB sockets)
```

### Message Format

```json
{
  "topic": "gateio.ohlc",
  "type": "ohlc",
  "source": "go",
  "ts": 1234567890,
  "data": {
    "symbol": "ETH_USDT",
    "open": 1234.5,
    "high": 1245.0,
    "low": 1223.0,
    "close": 1240.0,
    "volume": 1000.0
  }
}
```

### Routing Rules

| Topic | Routes To |
|-------|-----------|
| gateio.ohlc | python, go, router |
| gateio.ticker | go, python, router |
| kraken.ohlc | python, router |
| system.status | python, go, rust, cpp, router |

## 🛠️ Troubleshooting

### Python Tests Fail

```bash
# Check dependencies
python -m pip list | grep -E "pytest|pyzmq|requests"

# Install missing packages
python -m pip install pytest pytest-cov pyzmq requests

# Run with verbose output
pytest -v -s regression_tests.py
```

### Go Tests Fail

```bash
# Check Go version
go version  # Should be 1.21+

# Download dependencies
cd connectivity/go/
go mod tidy

# Run with verbose output
go test -v -run TestGateIOClient
```

### Rust Tests Fail

```bash
# Check Rust version
rustc --version  # Should be 1.70+

# Update dependencies
cd connectivity/rust/
cargo update

# Run with verbose output
cargo test -- --nocapture
```

### C++ Tests Fail

```bash
# Check CMake
cmake --version  # Should be 3.10+

# Rebuild from scratch
cd connectivity/cpp/
rm -rf build
mkdir build && cd build
cmake .. && make -j4
```

### ZMQ Bus Tests Fail

```bash
# Check ZMQ connectivity
python -c "import zmq; print(zmq.zmq_version())"

# Test endpoint reachability
python -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 5555))
"
```

### Robot Framework Tests Fail

```bash
# Check Robot Framework
robot --version  # Should be 7.4.1+

# Run with debug output
robot --loglevel DEBUG deployment.robot

# Generate detailed output
robot --outputdir results --log log.html deployment.robot
```

## 📝 Adding New Tests

### Python Test

```python
import pytest

@pytest.mark.python
@pytest.mark.gateio
class TestNewFeature:
    def test_feature(self, gateio_config):
        """Test description"""
        assert True
```

### Robot Framework Test

```robot
*** Test Cases ***
New Test Case
    [Documentation]    Test description
    [Tags]    tag1    tag2
    Log    Test execution
    Should Be Equal    1    1
```

### Go Test

```go
func TestNewFeature(t *testing.T) {
    // Test implementation
    if expected != actual {
        t.Errorf("Expected %v, got %v", expected, actual)
    }
}
```

## 📚 Documentation

- [BUILD_ENVIRONMENT.md](../README.md) - Build environment setup
- [PYTEST_CONFIGURATION.md](../README.md) - Pytest advanced configuration
- [ROBOT_FRAMEWORK.md](../README.md) - Robot Framework keyword reference
- [ZMQ_ROUTING.md](../README.md) - ZMQ message routing documentation
- [CLI_NAVIGATION.md](../README.md) - CLI enhancement guide

## 🎯 Test Execution Examples

### Full Suite

```bash
# Run everything
pytest -v
robot -d results tests/
```

### Quick Smoke Test

```bash
# Fast validation
pytest -m "python and not slow" -q
robot --include smoke tests/
```

### Continuous Integration

```bash
# CI pipeline execution
pytest --cov=connectivity --cov-report=xml -v
robot --exitonfailure --loglevel INFO tests/
```

## ✅ Validation Checklist

- [ ] Build environment verified (all languages installed)
- [ ] Python tests passing (pytest -m python -v)
- [ ] Go tests passing (go test -v)
- [ ] Rust tests passing (cargo test -v)
- [ ] C++ tests passing (make test)
- [ ] ZMQ bus connectivity validated
- [ ] Message routing verified
- [ ] Robot Framework tests passing
- [ ] CLI tab completion working
- [ ] CLI keyboard navigation working

## 🔗 Related Files

- [regression_tests.py](./regression_tests.py) - Python test suite
- [enhanced_cli.py](../cli/enhanced_cli.py) - CLI enhancements
- [go/main.go](../connectivity/go/main.go) - Go module
- [conftest.py](./conftest.py) - Pytest configuration

## 📞 Support

For issues or questions about the testing infrastructure:

1. Check [Troubleshooting](#-troubleshooting) section
2. Review relevant test file comments
3. Run with verbose/debug output
4. Check endpoint connectivity

---

**Last Updated**: 2024
**Version**: 1.0
**Status**: Production Ready
