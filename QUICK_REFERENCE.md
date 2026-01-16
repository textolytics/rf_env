# Quick Reference - Market Data Platform Testing

## 📊 File Statistics

```
Total Lines of Code Created: 4,751 lines
├─ Jupyter Notebook: 2,541 lines (10 sections)
├─ Python Tests: 491 lines (9 test classes)
├─ Pytest Config: 335 lines (10+ fixtures)
├─ Robot Tests: 467 lines (4 suites, 30+ cases)
├─ Documentation: 467 lines (complete guide)
└─ Go/CLI Modules: 419 lines (enhanced features)
```

## 🎯 At A Glance

| Task | Command | Time |
|------|---------|------|
| Run all tests | `pytest -v && robot tests/` | ~60s |
| Python only | `pytest -m python -v` | ~10s |
| Go tests | `cd connectivity/go && go test -v` | ~15s |
| Rust tests | `cd connectivity/rust && cargo test` | ~30s |
| RF deployment | `robot --include deployment deployment.robot` | ~5s |
| CLI completion | `python -c "from enhanced_cli import TabCompletionProvider"` | <1s |
| Keyboard nav | `python market_data_cli.py` then `← → ↑ ↓` | Interactive |

## 🚀 Essential Commands

### Pytest
```bash
# All tests
pytest -v

# By language
pytest -m python          # Python only
pytest -m go             # Go only  
pytest -m rust           # Rust only
pytest -m cpp            # C++ only

# By type
pytest -m integration    # Integration tests
pytest -m performance    # Performance tests
pytest -m zmq           # ZMQ tests

# With coverage
pytest --cov=connectivity --cov-report=html
```

### Robot Framework
```bash
# All test suites
robot --outputdir results testing/

# Specific suite
robot deployment.robot              # Deployment tests
robot gateways.robot                # Gateway tests
robot data_operations.robot         # Data tests
robot multilang_integration.robot   # Integration tests

# With tags
robot --include deployment testing/
robot --exclude slow testing/
```

### Go Tests
```bash
cd connectivity/go
go test -v                      # Verbose
go test -run TestGateIO        # Specific test
go test -bench=.               # Benchmarks
```

### Rust Tests
```bash
cd connectivity/rust
cargo test                      # All tests
cargo test --release           # Release build
cargo test -- --nocapture      # Show output
```

## 📋 Test Categories

### Python Modules (4 classes, 17 methods)
- TestPythonModules - Gate.io and ZMQ tests
- TestCppModules - C++ build verification
- TestRustModules - Rust compilation tests
- TestGoModules - Go module structure
- TestZMQIntegration - ZMQ routing
- TestDataFlow - End-to-end flow
- TestPerformance - Benchmarks

### Robot Framework (4 suites, 30+ cases)
- deployment.robot - 6 test cases
- gateways.robot - 6 test cases
- data_operations.robot - 9 test cases
- multilang_integration.robot - 9+ test cases

### Pytest Markers
- `python` - Python tests
- `cpp` - C++ tests
- `rust` - Rust tests
- `go` - Go tests
- `zmq` - ZMQ integration
- `integration` - Integration tests
- `performance` - Performance tests
- `slow` - Slow tests
- `unit` - Unit tests

## 🔧 Configuration Files

### pytest.ini (Pytest configuration)
- Test discovery patterns
- Marker definitions
- Output options
- Coverage settings

### conftest.py (Pytest fixtures)
- `test_environment` - Global config
- `gateio_config` - Gate.io settings
- `zmq_config` - ZMQ endpoints
- `subprocess_runner` - Execute commands
- `performance_timer` - Timing
- `test_data_factory` - Test data

### enhanced_cli.py (CLI features)
- TabCompletionProvider (100+ keywords)
- KeyboardNavigationHandler (5 groups)
- EnhancedCLI (menu system)

## 📁 Directory Structure

```
testing/
├── market_data_regression_tests.ipynb    (Main notebook)
├── regression_tests.py                   (Python test suite)
├── conftest.py                           (Pytest config)
├── deployment.robot                      (Deployment tests)
├── gateways.robot                        (Gateway tests)
├── data_operations.robot                 (Data tests)
├── multilang_integration.robot           (Integration tests)
├── TESTING_README.md                     (Full documentation)
└── keywords/                             (RF keywords)

connectivity/
├── python/                               (Python module)
├── cpp/                                  (C++ module)
├── rust/                                 (Rust module)
└── go/                                   (Go module + tests)

cli/
└── enhanced_cli.py                       (Tab completion + nav)
```

## ⚡ Performance Targets

| Metric | Target | How to Test |
|--------|--------|------------|
| Connection Latency | <100ms | `pytest -m performance` |
| Message Throughput | >100 msg/sec | `pytest -m performance` |
| OHLC Fetch Time | <1s | `pytest::TestPythonModules::test_python_gateio_connectivity` |
| Total Test Suite | <60s | `pytest -v` |
| ZMQ Validation | <5s | `pytest -m zmq` |

## 🔐 Integration Checklist

- [ ] Build environment verified
- [ ] Python tests passing
- [ ] Go tests passing
- [ ] Rust tests passing
- [ ] C++ tests passing
- [ ] ZMQ connectivity validated
- [ ] Message routing working
- [ ] CLI completion functional
- [ ] Keyboard navigation working
- [ ] Robot Framework running

## 🆘 Troubleshooting Quick Fixes

### Tests fail
```bash
# Reinstall dependencies
pip install -r requirements.txt
pytest --collect-only           # Check discovery
pytest -v --tb=long            # Full traceback
```

### Import errors
```bash
python -c "import pytest, zmq, requests"
pip install pytest pyzmq requests
```

### ZMQ issues
```bash
python -c "import zmq; print(zmq.zmq_version())"
pip install pyzmq --upgrade
```

### Robot Framework issues
```bash
robot --version              # Check version
robot --dryrun deployment.robot  # Validate syntax
robot --loglevel DEBUG deployment.robot
```

## 📞 Key Contacts

For specific issues:
1. **Python tests** → regression_tests.py
2. **Go module** → connectivity/go/main.go
3. **Pytest config** → conftest.py
4. **Robot tests** → *.robot files
5. **CLI enhancements** → enhanced_cli.py
6. **Documentation** → TESTING_README.md

## 📈 Monitoring

### Test Execution
```bash
# Monitor test runs
pytest -v --tb=short --timeout=60

# Generate reports
pytest --html=report.html --self-contained-html
robot --outputdir results tests/
```

### Performance
```bash
pytest -m performance -v --durations=10
python -c "from market_data_regression_tests import ZMQRouter; router = ZMQRouter()"
```

## 🎓 Learning Resources

- **Pytest**: conftest.py, regression_tests.py
- **Robot Framework**: *.robot files, keywords/
- **Go Testing**: connectivity/go/main.go
- **ZMQ Integration**: market_data_regression_tests.ipynb Section 6
- **CLI**: enhanced_cli.py

---

**Quick Tip**: Start with `pytest -m python -v` to validate your setup!
