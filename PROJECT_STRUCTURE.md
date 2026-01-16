# Market Data Platform - Complete Project Structure

## 🏗️ Project Organization by Language

```
market_data_platform/
├── 🐍 PYTHON/                          # Core platform and CLI
├── 🐹 GO/                              # Gate.io connectivity
├── 🦀 RUST/                            # Data processing
├── 🤖 ROBOT_FRAMEWORK/                 # Test automation
├── 📋 CONFIG/                          # Shared configuration
├── 🧪 TESTS/                           # Integration tests
├── 📚 DOCUMENTATION/                   # Project docs
└── 🔧 BUILD/                           # Build configuration

```

## 📦 Python Module (Core Platform)

```
python/
├── 📁 __pycache__/
├── 📁 market_data_platform/
│   ├── __init__.py
│   ├── __main__.py
│   ├── 📁 cli/                         # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py                     # CLI entry point
│   │   ├── terminal.py                 # Terminal interface
│   │   ├── advanced_menu_terminal.py   # Advanced menu system
│   │   ├── commander_terminal.py       # MC-style terminal
│   │   ├── advanced_dashboard.py       # System dashboard
│   │   ├── terminal_integration.py     # Integration utilities
│   │   ├── unified_terminal_launcher.py # Mode selector
│   │   ├── enhanced_cli.py             # Enhanced CLI features
│   │   └── test_components.py          # CLI tests
│   │
│   ├── 📁 core/                        # Core modules
│   │   ├── __init__.py
│   │   ├── gateway_manager.py          # Gateway management
│   │   ├── data_processor.py           # Data processing
│   │   ├── session_manager.py          # Session management
│   │   ├── event_bus.py                # Event handling
│   │   └── logger.py                   # Logging
│   │
│   ├── 📁 gateway/                     # Gateway modules
│   │   ├── __init__.py
│   │   ├── python_gateway.py           # Python gateway
│   │   ├── base_gateway.py             # Base gateway class
│   │   └── gateway_registry.py         # Gateway registry
│   │
│   ├── 📁 api/                         # API handlers
│   │   ├── __init__.py
│   │   ├── rest_api.py                 # REST API
│   │   ├── websocket_api.py            # WebSocket API
│   │   └── request_handler.py          # Request handling
│   │
│   ├── 📁 config/                      # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py                 # Main settings
│   │   ├── logger_config.py            # Logging config
│   │   ├── database_config.py          # Database config
│   │   └── service_config.py           # Service config
│   │
│   ├── 📁 utils/                       # Utilities
│   │   ├── __init__.py
│   │   ├── validators.py               # Input validation
│   │   ├── formatters.py               # Data formatting
│   │   ├── converters.py               # Type conversion
│   │   └── helpers.py                  # Helper functions
│   │
│   ├── 📁 models/                      # Data models
│   │   ├── __init__.py
│   │   ├── market_data.py              # Market data model
│   │   ├── trade.py                    # Trade model
│   │   ├── order.py                    # Order model
│   │   └── user.py                     # User model
│   │
│   └── 📁 storage/                     # Storage handlers
│       ├── __init__.py
│       ├── database.py                 # Database handler
│       ├── cache.py                    # Cache handler
│       └── file_storage.py             # File storage
│
├── 📁 tests/                           # Python tests
│   ├── __init__.py
│   ├── conftest.py                     # pytest configuration
│   ├── 📁 unit/
│   │   ├── __init__.py
│   │   ├── test_gateway_manager.py
│   │   ├── test_data_processor.py
│   │   ├── test_validators.py
│   │   └── test_models.py
│   │
│   ├── 📁 integration/
│   │   ├── __init__.py
│   │   ├── test_python_go.py
│   │   ├── test_python_rust.py
│   │   ├── test_api_integration.py
│   │   └── test_full_pipeline.py
│   │
│   └── 📁 fixtures/
│       ├── __init__.py
│       ├── mock_data.py
│       ├── test_data.py
│       └── conftest_fixtures.py
│
├── requirements.txt                    # Python dependencies
├── requirements-dev.txt                # Development dependencies
├── setup.py                            # Package setup
├── setup.cfg                           # Setup configuration
├── pyproject.toml                      # Modern Python project config
└── MANIFEST.in                         # Manifest for packaging
```

## 🐹 Go Module (Gate.io Gateway)

```
go/
├── go.mod                              # Go module definition
├── go.sum                              # Go module checksums
├── main.go                             # Entry point
├── 📁 cmd/
│   ├── gateway/
│   │   └── main.go                     # Gateway command
│   └── client/
│       └── main.go                     # CLI client
│
├── 📁 pkg/
│   ├── config/
│   │   ├── config.go                   # Configuration
│   │   ├── loader.go                   # Config loader
│   │   └── validator.go                # Config validator
│   │
│   ├── gateway/
│   │   ├── gateway.go                  # Gateway interface
│   │   ├── gateio/
│   │   │   ├── client.go               # Gate.io API client
│   │   │   ├── auth.go                 # Authentication
│   │   │   ├── rest.go                 # REST API
│   │   │   ├── websocket.go            # WebSocket API
│   │   │   ├── models.go               # Data models
│   │   │   └── errors.go               # Error handling
│   │   └── registry.go                 # Gateway registry
│   │
│   ├── server/
│   │   ├── server.go                   # Server setup
│   │   ├── routes.go                   # Route definitions
│   │   ├── handlers.go                 # HTTP handlers
│   │   └── middleware.go               # Middleware
│   │
│   ├── zmq/
│   │   ├── zmq.go                      # ZMQ integration
│   │   ├── publisher.go                # ZMQ publisher
│   │   └── subscriber.go               # ZMQ subscriber
│   │
│   ├── logger/
│   │   ├── logger.go                   # Logging setup
│   │   └── zap_logger.go               # Zap logger
│   │
│   ├── cache/
│   │   ├── cache.go                    # Cache interface
│   │   └── redis_cache.go              # Redis cache
│   │
│   └── utils/
│       ├── helpers.go                  # Helper functions
│       ├── validators.go               # Validators
│       └── converters.go               # Type converters
│
├── 📁 internal/
│   ├── auth/
│   │   ├── jwt.go                      # JWT authentication
│   │   └── api_key.go                  # API key auth
│   │
│   ├── models/
│   │   ├── market.go                   # Market data
│   │   ├── order.go                    # Order data
│   │   ├── trade.go                    # Trade data
│   │   └── user.go                     # User data
│   │
│   └── storage/
│       ├── postgres.go                 # PostgreSQL handler
│       └── redis.go                    # Redis handler
│
├── 📁 test/
│   ├── unit/
│   │   ├── gateway_test.go
│   │   ├── client_test.go
│   │   ├── auth_test.go
│   │   └── models_test.go
│   │
│   ├── integration/
│   │   ├── gateway_integration_test.go
│   │   ├── api_integration_test.go
│   │   └── zmq_integration_test.go
│   │
│   └── fixtures/
│       ├── mock_responses.go
│       ├── test_data.go
│       └── fixtures.go
│
├── Makefile                            # Build commands
├── docker/
│   ├── Dockerfile                      # Go container
│   └── docker-compose.yml              # Compose config
│
└── README.md                           # Go module documentation
```

## 🦀 Rust Module (Data Processor)

```
rust/
├── Cargo.toml                          # Rust manifest
├── Cargo.lock                          # Dependency lock
├── src/
│   ├── main.rs                         # Entry point
│   ├── lib.rs                          # Library root
│   │
│   ├── 📁 bin/
│   │   ├── processor.rs                # Data processor binary
│   │   ├── gateway.rs                  # Gateway binary
│   │   └── cli.rs                      # CLI binary
│   │
│   ├── 📁 config/
│   │   ├── mod.rs                      # Module definition
│   │   ├── loader.rs                   # Configuration loader
│   │   ├── validator.rs                # Configuration validator
│   │   └── settings.rs                 # Settings structure
│   │
│   ├── 📁 gateway/
│   │   ├── mod.rs
│   │   ├── base.rs                     # Base gateway trait
│   │   ├── connector.rs                # Gateway connector
│   │   ├── registry.rs                 # Gateway registry
│   │   └── errors.rs                   # Error types
│   │
│   ├── 📁 processor/
│   │   ├── mod.rs
│   │   ├── engine.rs                   # Processing engine
│   │   ├── pipeline.rs                 # Processing pipeline
│   │   ├── handlers.rs                 # Data handlers
│   │   └── rules.rs                    # Processing rules
│   │
│   ├── 📁 models/
│   │   ├── mod.rs
│   │   ├── market_data.rs              # Market data
│   │   ├── order.rs                    # Order data
│   │   ├── trade.rs                    # Trade data
│   │   └── candle.rs                   # Candle data
│   │
│   ├── 📁 zmq/
│   │   ├── mod.rs
│   │   ├── context.rs                  # ZMQ context
│   │   ├── socket.rs                   # ZMQ socket
│   │   ├── message.rs                  # Message handling
│   │   └── broker.rs                   # Message broker
│   │
│   ├── 📁 storage/
│   │   ├── mod.rs
│   │   ├── postgres.rs                 # PostgreSQL
│   │   ├── cache.rs                    # Cache layer
│   │   ├── query.rs                    # Query builder
│   │   └── pool.rs                     # Connection pool
│   │
│   ├── 📁 api/
│   │   ├── mod.rs
│   │   ├── server.rs                   # Server setup
│   │   ├── routes.rs                   # Route handlers
│   │   ├── middleware.rs               # Middleware
│   │   └── error_handler.rs            # Error handling
│   │
│   ├── 📁 utils/
│   │   ├── mod.rs
│   │   ├── logger.rs                   # Logging
│   │   ├── validators.rs               # Validators
│   │   ├── converters.rs               # Type converters
│   │   └── helpers.rs                  # Helpers
│   │
│   └── 📁 error/
│       ├── mod.rs
│       ├── kind.rs                     # Error kinds
│       └── handler.rs                  # Error handlers
│
├── 📁 tests/
│   ├── common/
│   │   ├── mod.rs
│   │   ├── fixtures.rs
│   │   └── helpers.rs
│   │
│   ├── unit/
│   │   ├── processor_test.rs
│   │   ├── gateway_test.rs
│   │   ├── models_test.rs
│   │   └── utils_test.rs
│   │
│   └── integration/
│       ├── processor_integration_test.rs
│       ├── zmq_integration_test.rs
│       └── api_integration_test.rs
│
├── 📁 benches/
│   ├── processor_bench.rs
│   ├── zmq_bench.rs
│   └── storage_bench.rs
│
├── Makefile                            # Build commands
├── .cargo/
│   └── config.toml                     # Cargo config
│
├── docker/
│   ├── Dockerfile                      # Rust container
│   └── docker-compose.yml
│
└── README.md                           # Rust module documentation
```

## 🤖 Robot Framework Tests

```
robot_framework/
├── 📁 keywords/
│   ├── common.robot                    # Common keywords
│   ├── gateway_keywords.robot          # Gateway keywords
│   ├── component_keywords.robot        # Component keywords
│   ├── data_keywords.robot             # Data keywords
│   ├── config_keywords.robot           # Config keywords
│   └── test_keywords.robot             # Test keywords
│
├── 📁 test_suites/
│   ├── 📁 gateway_tests/
│   │   ├── connect_tests.robot
│   │   ├── disconnect_tests.robot
│   │   ├── status_tests.robot
│   │   ├── data_stream_tests.robot
│   │   └── error_handling_tests.robot
│   │
│   ├── 📁 component_tests/
│   │   ├── start_component_tests.robot
│   │   ├── stop_component_tests.robot
│   │   ├── status_tests.robot
│   │   ├── restart_tests.robot
│   │   └── lifecycle_tests.robot
│   │
│   ├── 📁 data_tests/
│   │   ├── fetch_ohlc_tests.robot
│   │   ├── process_data_tests.robot
│   │   ├── store_data_tests.robot
│   │   ├── query_tests.robot
│   │   └── aggregate_tests.robot
│   │
│   ├── 📁 config_tests/
│   │   ├── set_config_tests.robot
│   │   ├── get_config_tests.robot
│   │   ├── load_config_tests.robot
│   │   ├── save_config_tests.robot
│   │   └── reset_config_tests.robot
│   │
│   ├── 📁 integration_tests/
│   │   ├── full_pipeline_test.robot
│   │   ├── cross_gateway_test.robot
│   │   ├── performance_test.robot
│   │   ├── stress_test.robot
│   │   └── regression_test.robot
│   │
│   └── 📁 system_tests/
│       ├── health_check_test.robot
│       ├── deployment_test.robot
│       ├── recovery_test.robot
│       └── cleanup_test.robot
│
├── 📁 resources/
│   ├── common.robot
│   ├── variables.robot
│   ├── settings.robot
│   └── fixtures.robot
│
├── 📁 notebooks/
│   ├── gateway_tests.ipynb
│   ├── component_tests.ipynb
│   ├── data_pipeline_tests.ipynb
│   ├── integration_tests.ipynb
│   └── system_health_check.ipynb
│
├── robot.cfg                           # Robot Framework config
├── robot.yml                           # YAML config
├── conftest.py                         # pytest-robotframework config
├── pytest.ini                          # pytest configuration
│
└── README.md                           # RF documentation
```

## 📋 Configuration Files

```
config/
├── application/
│   ├── config.yaml                     # Main config
│   ├── config.dev.yaml                 # Dev config
│   ├── config.prod.yaml                # Production config
│   └── config.test.yaml                # Test config
│
├── logging/
│   ├── logging.yaml                    # Logging config
│   ├── logging-python.yml
│   ├── logging-go.yml
│   └── logging-rust.yml
│
├── database/
│   ├── postgres.yml                    # PostgreSQL config
│   ├── migrations/
│   │   ├── 001_create_tables.sql
│   │   ├── 002_create_indexes.sql
│   │   └── 003_add_columns.sql
│   └── schema.sql                      # Database schema
│
├── zmq/
│   ├── zmq.yml                         # ZMQ config
│   ├── endpoints.yml                   # Endpoint config
│   └── patterns.yml                    # Pattern config
│
├── gateway/
│   ├── gateway.yml                     # Gateway config
│   ├── gateio.yml                      # Gate.io config
│   └── endpoints.yml                   # API endpoints
│
├── cache/
│   ├── redis.yml                       # Redis config
│   └── cache.yml                       # Cache config
│
├── api/
│   ├── openapi.yaml                    # OpenAPI spec
│   ├── rest-api.yml
│   └── websocket.yml
│
├── cli/
│   ├── cli.yml                         # CLI config
│   ├── commands.yml                    # Command config
│   └── options.yml                     # Options config
│
├── security/
│   ├── api_keys.yml                    # API keys (encrypted)
│   ├── jwt.yml                         # JWT config
│   └── ssl.yml                         # SSL config
│
├── monitoring/
│   ├── metrics.yml                     # Metrics config
│   ├── alerts.yml                      # Alert config
│   └── health.yml                      # Health check config
│
└── env/
    ├── .env.example                    # Example env file
    ├── .env.local                      # Local env (gitignored)
    └── .env.production                 # Production env
```

## 🧪 Integration Tests

```
tests/
├── integration/
│   ├── conftest.py                     # Test configuration
│   ├── pytest.ini                      # pytest config
│   │
│   ├── 📁 system/
│   │   ├── test_system_health.py
│   │   ├── test_components_startup.py
│   │   ├── test_full_pipeline.py
│   │   ├── test_error_recovery.py
│   │   └── test_graceful_shutdown.py
│   │
│   ├── 📁 gateways/
│   │   ├── test_python_gateway.py
│   │   ├── test_go_gateway.py
│   │   ├── test_rust_gateway.py
│   │   ├── test_gateway_switching.py
│   │   └── test_gateway_failover.py
│   │
│   ├── 📁 data_pipeline/
│   │   ├── test_data_flow.py
│   │   ├── test_data_processing.py
│   │   ├── test_data_storage.py
│   │   ├── test_data_retrieval.py
│   │   └── test_data_validation.py
│   │
│   ├── 📁 api/
│   │   ├── test_rest_api.py
│   │   ├── test_websocket_api.py
│   │   ├── test_authentication.py
│   │   ├── test_authorization.py
│   │   └── test_rate_limiting.py
│   │
│   ├── 📁 messaging/
│   │   ├── test_zmq_messaging.py
│   │   ├── test_message_routing.py
│   │   ├── test_pub_sub.py
│   │   └── test_req_rep.py
│   │
│   ├── 📁 performance/
│   │   ├── test_throughput.py
│   │   ├── test_latency.py
│   │   ├── test_memory_usage.py
│   │   └── test_concurrent_load.py
│   │
│   ├── 📁 cli/
│   │   ├── test_cli_commands.py
│   │   ├── test_menu_navigation.py
│   │   ├── test_terminal_ui.py
│   │   └── test_cli_integration.py
│   │
│   └── 📁 fixtures/
│       ├── conftest_fixtures.py
│       ├── mock_data.py
│       ├── test_database.py
│       ├── test_cache.py
│       └── test_services.py
│
├── e2e/
│   ├── conftest.py
│   ├── test_complete_workflow.py
│   ├── test_user_scenarios.py
│   ├── test_edge_cases.py
│   ├── test_recovery_scenarios.py
│   └── test_performance_scenarios.py
│
├── smoke/
│   ├── conftest.py
│   ├── test_basic_connectivity.py
│   ├── test_api_availability.py
│   ├── test_database_connection.py
│   └── test_service_health.py
│
├── regression/
│   ├── conftest.py
│   ├── 📁 python/
│   │   ├── test_python_modules.py
│   │   ├── test_python_gateways.py
│   │   └── test_python_regressions.py
│   │
│   ├── 📁 go/
│   │   ├── test_go_connectivity.py
│   │   ├── test_go_gateways.py
│   │   └── test_go_regressions.py
│   │
│   ├── 📁 rust/
│   │   ├── test_rust_processor.py
│   │   ├── test_rust_performance.py
│   │   └── test_rust_regressions.py
│   │
│   └── 📁 integration/
│       ├── test_cross_language.py
│       ├── test_data_flow.py
│       └── test_system_integration.py
│
└── fixtures/
    ├── mock_services/
    │   ├── mock_gateway.py
    │   ├── mock_database.py
    │   ├── mock_zmq.py
    │   └── mock_api.py
    │
    ├── test_data/
    │   ├── market_data.json
    │   ├── trades.json
    │   ├── orders.json
    │   └── user_data.json
    │
    └── docker/
        ├── docker-compose.test.yml
        ├── Dockerfile.test
        └── setup_test_env.sh
```

## 🔧 Build & Configuration

```
build/
├── 📁 docker/
│   ├── Dockerfile.python                # Python container
│   ├── Dockerfile.go                    # Go container
│   ├── Dockerfile.rust                  # Rust container
│   ├── Dockerfile.rf                    # RF container
│   ├── docker-compose.yml               # Full stack
│   ├── docker-compose.dev.yml           # Dev stack
│   ├── docker-compose.test.yml          # Test stack
│   ├── .dockerignore
│   └── entrypoints/
│       ├── python-entrypoint.sh
│       ├── go-entrypoint.sh
│       ├── rust-entrypoint.sh
│       └── rf-entrypoint.sh
│
├── 📁 kubernetes/
│   ├── namespace.yaml
│   ├── 📁 python/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   └── secrets.yaml
│   │
│   ├── 📁 go/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   │
│   ├── 📁 rust/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   │
│   ├── 📁 database/
│   │   ├── postgres-deployment.yaml
│   │   ├── postgres-service.yaml
│   │   └── postgres-pvc.yaml
│   │
│   ├── 📁 cache/
│   │   ├── redis-deployment.yaml
│   │   ├── redis-service.yaml
│   │   └── redis-pvc.yaml
│   │
│   └── 📁 monitoring/
│       ├── prometheus-deployment.yaml
│       ├── grafana-deployment.yaml
│       └── alerts.yaml
│
├── 📁 scripts/
│   ├── setup.sh                        # Setup script
│   ├── build-all.sh                    # Build all modules
│   ├── build-python.sh                 # Build Python
│   ├── build-go.sh                     # Build Go
│   ├── build-rust.sh                   # Build Rust
│   ├── run-tests.sh                    # Run tests
│   ├── deploy.sh                       # Deploy script
│   ├── cleanup.sh                      # Cleanup script
│   └── health-check.sh                 # Health check
│
├── 📁 ci-cd/
│   ├── .github/workflows/
│   │   ├── python-ci.yml
│   │   ├── go-ci.yml
│   │   ├── rust-ci.yml
│   │   ├── tests-ci.yml
│   │   ├── integration-ci.yml
│   │   ├── deploy-ci.yml
│   │   └── security-ci.yml
│   │
│   ├── .gitlab-ci.yml                  # GitLab CI
│   ├── .travis.yml                     # Travis CI
│   ├── Jenkinsfile                     # Jenkins
│   └── azure-pipelines.yml             # Azure Pipelines
│
├── Makefile                            # Make targets
├── CMakeLists.txt                      # CMake config
├── buildspec.yml                       # AWS CodeBuild
└── tox.ini                             # tox configuration
```

## 📚 Documentation

```
docs/
├── 📁 architecture/
│   ├── README.md                       # Architecture overview
│   ├── system_design.md                # System design
│   ├── module_dependencies.md          # Module dependencies
│   ├── data_flow.md                    # Data flow diagrams
│   └── integration_points.md           # Integration points
│
├── 📁 api/
│   ├── openapi.yaml                    # OpenAPI spec
│   ├── rest_api.md                     # REST API docs
│   ├── websocket_api.md                # WebSocket docs
│   ├── grpc_api.md                     # gRPC docs
│   └── examples.md                     # API examples
│
├── 📁 modules/
│   ├── python_module.md                # Python docs
│   ├── go_module.md                    # Go docs
│   ├── rust_module.md                  # Rust docs
│   └── robot_framework.md              # RF docs
│
├── 📁 guides/
│   ├── installation.md                 # Installation guide
│   ├── configuration.md                # Configuration guide
│   ├── deployment.md                   # Deployment guide
│   ├── troubleshooting.md              # Troubleshooting
│   ├── performance_tuning.md           # Performance tuning
│   └── security.md                     # Security guide
│
├── 📁 development/
│   ├── contributing.md                 # Contributing guide
│   ├── development_setup.md            # Dev setup
│   ├── coding_standards.md             # Coding standards
│   ├── testing_guide.md                # Testing guide
│   ├── release_process.md              # Release process
│   └── debugging.md                    # Debugging guide
│
├── 📁 examples/
│   ├── basic_usage.md
│   ├── gateway_integration.md
│   ├── data_processing.md
│   ├── cli_usage.md
│   ├── api_integration.md
│   └── advanced_scenarios.md
│
├── 📁 troubleshooting/
│   ├── common_issues.md
│   ├── faq.md
│   ├── error_codes.md
│   └── logs_analysis.md
│
└── CHANGELOG.md                        # Changelog
```

## 🏢 Root Level Files

```
/root/rf_env/
├── 📁 market_data_platform/            # Main project directory
├── 📁 tests/                           # Integration tests
├── 📁 build/                           # Build configuration
├── 📁 config/                          # Configuration files
├── 📁 docs/                            # Documentation
│
├── README.md                           # Project readme
├── CONTRIBUTING.md                     # Contributing guide
├── LICENSE                             # License file
├── .gitignore                          # Git ignore
├── .gitattributes                      # Git attributes
├── .editorconfig                       # Editor config
│
├── Makefile                            # Root makefile
├── setup.py                            # Python setup
├── docker-compose.yml                  # Docker compose
├── docker-compose.prod.yml             # Prod compose
│
├── requirements.txt                    # Root requirements
├── requirements-dev.txt                # Dev requirements
├── pyproject.toml                      # Python project config
├── setup.cfg                           # Setup config
├── tox.ini                             # Tox config
│
├── pytest.ini                          # pytest config
├── robot.yml                           # Robot Framework config
├── .pre-commit-config.yaml             # Pre-commit hooks
├── .pylintrc                           # Pylint config
│
├── VERSION                             # Version file
├── .env.example                        # Example env
├── .env.local                          # Local env (gitignored)
└── TERMINAL_SYSTEM_GUIDE.md            # Terminal guide
```

## 📊 Summary Statistics

| Language | Components | Files | Lines | Purpose |
|----------|-----------|-------|-------|---------|
| **Python** | Core Platform | 40+ | 15,000+ | CLI, APIs, Gateway Manager |
| **Go** | Gate.io Gateway | 25+ | 8,000+ | Gate.io Connectivity, Server |
| **Rust** | Data Processor | 20+ | 10,000+ | High-performance Processing |
| **Robot Framework** | Test Automation | 30+ | 5,000+ | System & Integration Tests |
| **YAML/Config** | Configuration | 20+ | 2,000+ | App Configuration |
| **SQL** | Database | 5+ | 1,000+ | Schema & Migrations |
| **Shell** | Build Scripts | 10+ | 1,000+ | Build & Deployment |
| **Docker** | Containerization | 8+ | 500+ | Container Setup |
| **Documentation** | Guides & Docs | 20+ | 10,000+ | Project Documentation |

**Total Project Size**: 100,000+ lines of code & documentation

