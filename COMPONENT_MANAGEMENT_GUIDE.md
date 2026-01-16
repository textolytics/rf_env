# Market Data Platform - Refactored Component Management System

## Overview

This document describes the refactored component management system that enables **graceful start/stop of service components** organized by logical groups in Docker containers with comprehensive connectivity validation and Robot Framework integration.

## Architecture

### Component Groups

Services are organized into 8 logical component groups with dependency management:

```
database
├── PostgreSQL (data storage)
└── Redis (caching)

storage
└── InfluxDB (time-series data)

monitoring
├── Prometheus (metrics)
└── Grafana (dashboards)

messaging
├── ZMQ Publisher (tcp://127.0.0.1:5555)
└── ZMQ Subscriber (tcp://127.0.0.1:5556)

api
└── Python FastAPI service

gateway
└── Go data gateway service

processor
└── Rust data processor

proxy
└── Nginx reverse proxy
```

### Dependency Graph

```
database (no dependencies)
    ↓
monitoring ← database
    ↓
storage ← database
    ↓
messaging (no dependencies)
    ↓
api ← database
gateway ← database
processor ← database, messaging
proxy ← api, gateway
```

## Core Components

### 1. Component Manager (`lib/component_manager.sh`)

Centralized bash script for managing service components.

**Features:**
- Start/stop individual or multiple components
- Automatic dependency resolution
- Health validation for each component
- Graceful shutdown with SIGTERM signals
- Process tracking via PID files
- Component status reporting

**Key Functions:**
```bash
start_component <name>        # Start component with dependencies
stop_component <name>         # Stop component gracefully
is_component_running <name>   # Check if running
validate_component_health <name>  # Health check
start_components [@names]     # Start multiple
stop_components [@names]      # Stop multiple
status_all_components         # Show all status
```

### 2. Connectivity Validator (`market_data_platform/connectivity/validator.py`)

Python module for validating service connectivity and health.

**Features:**
- Async health checks for all services
- Support for HTTP, PostgreSQL, Redis, ZMQ endpoints
- Detailed response time metrics
- Health history tracking
- JSON summary reports

**Services Validated:**
- `database_postgres` - PostgreSQL health
- `cache_redis` - Redis connectivity
- `storage_influxdb` - InfluxDB HTTP endpoint
- `monitoring_prometheus` - Prometheus health
- `monitoring_grafana` - Grafana API
- `api_python` - Python API service
- `gateway_go` - Go Gateway service
- `messaging_zmq_publisher` - ZMQ Publisher
- `messaging_zmq_subscriber` - ZMQ Subscriber

### 3. Enhanced Start/Stop Scripts

#### `bin/start.sh` - Graceful Startup

Implements layered startup with component groups:
1. **Step 1:** Database Layer (PostgreSQL + Redis)
2. **Step 2:** Storage Layer (InfluxDB)
3. **Step 3:** Monitoring Layer (Prometheus + Grafana)
4. **Step 4:** Messaging Layer (ZMQ)
5. **Step 5:** Application Services (API, Gateway, Processor)
6. **Step 6:** Proxy Layer (Nginx)

Each step waits for health validation before proceeding.

#### `bin/stop.sh` - Graceful Shutdown

Implements reverse order shutdown:
1. Stop Proxy (nginx)
2. Stop Application Services (processor, gateway, api)
3. Stop Messaging (ZMQ)
4. Stop Monitoring (prometheus, grafana)
5. Stop Storage (influxdb)
6. Stop Database (postgres, redis)

All services receive SIGTERM for graceful shutdown before SIGKILL.

### 4. Robot Framework Keywords (`robot_framework/resources/component_management.robot`)

Comprehensive RF keywords for component and connectivity testing.

**Component Management Keywords:**
- `Start Component` - Start single component
- `Start All Components` - Start all in order
- `Start Component Group` - Start multiple components
- `Stop Component` - Stop single component
- `Stop All Components` - Stop all in reverse
- `Restart Component` - Stop + start
- `Get Component Status` - Display status
- `Assert Component Is Running` - Assertion
- `Assert Component Is Stopped` - Assertion

**Connectivity Keywords:**
- `Validate Service Connectivity` - Single service validation
- `Validate All Services Connectivity` - All services
- `Get Service Health Status` - Detailed health info
- `Get Overall Connectivity Status` - System status
- `Assert Service Is Healthy` - Health assertion
- `Wait For Service To Be Ready` - Wait with timeout

**ZMQ Messaging Keywords:**
- `Validate ZMQ Publisher` - Publisher validation
- `Validate ZMQ Subscriber` - Subscriber validation
- `Validate Messaging Infrastructure` - Full messaging stack

**Data Warehousing Keywords:**
- `Validate Database Connection` - PostgreSQL
- `Validate Cache Connection` - Redis
- `Validate InfluxDB Connection` - InfluxDB
- `Validate Data Storage` - All warehousing

**System Keywords:**
- `Initialize System` - Full system startup
- `Shutdown System` - Full system shutdown
- `Reinitialize System` - Restart entire system

### 5. CLI Tool (`bin/mdp-cli`)

Comprehensive command-line interface for all operations.

**Usage:**
```bash
mdp-cli component start [COMPONENT]      # Start component(s)
mdp-cli component stop [COMPONENT]       # Stop component(s)
mdp-cli component status                 # Show status
mdp-cli component restart [COMPONENT]    # Restart component(s)
mdp-cli component logs [COMPONENT]       # Show logs

mdp-cli validate connectivity            # Validate all
mdp-cli validate service [SERVICE]       # Validate specific
mdp-cli validate database                # Validate warehousing
mdp-cli validate messaging               # Validate ZMQ

mdp-cli health check                     # Quick check
mdp-cli health report                    # Detailed report

mdp-cli test component                   # Test component management
mdp-cli test connectivity                # Test connectivity
mdp-cli test zmq                         # Test ZMQ
mdp-cli test warehousing                 # Test warehousing
mdp-cli test all                         # Run all tests

mdp-cli system init                      # Initialize system
mdp-cli system verify                    # Verify system
mdp-cli system shutdown                  # Shutdown system
mdp-cli system restart                   # Restart system

mdp-cli db shell                         # PostgreSQL shell
mdp-cli db redis                         # Redis CLI
mdp-cli db influx                        # InfluxDB CLI
```

## Test Suites

### 1. Component Management Tests
**File:** `robot_framework/test_suites/system_tests/component_management.robot`

Tests for graceful component start/stop and management:
- TC_001-010: Component lifecycle (start, stop, restart)
- TC_011-015: Dependencies and health
- TC_016-020: Benchmarking and reporting

### 2. Connectivity Validation Tests
**File:** `robot_framework/test_suites/system_tests/connectivity_validation.robot`

Tests for service connectivity validation:
- TC_C001-009: Individual service connectivity
- TC_C010-015: Service health assertions
- TC_C016-025: Overall system connectivity

### 3. ZMQ Messaging Tests
**File:** `robot_framework/test_suites/system_tests/zmq_messaging_tests.robot`

Tests for ZMQ messaging infrastructure:
- TC_Z001-010: Publisher/Subscriber lifecycle
- TC_Z011-015: ZMQ connectivity and health
- TC_Z016-020: Recovery and stress tests

### 4. Data Warehousing Tests
**File:** `robot_framework/test_suites/system_tests/data_warehousing_tests.robot`

Tests for database, cache, and storage:
- TC_DW001-010: Component startup and connectivity
- TC_DW011-015: Performance metrics
- TC_DW016-025: Recovery and integrity

## Usage Examples

### Starting Components

```bash
# Start specific component
./bin/mdp-cli component start database

# Start multiple components
./bin/mdp-cli component start messaging api gateway

# Start all components in order
./bin/mdp-cli component start

# Or directly via script
bash lib/component_manager.sh start database
```

### Validating Connectivity

```bash
# Validate all services
./bin/mdp-cli validate connectivity

# Validate specific service
./bin/mdp-cli validate service api_python

# Validate data warehousing
./bin/mdp-cli validate database

# Validate ZMQ messaging
./bin/mdp-cli validate messaging

# Get system health
./bin/mdp-cli health check
./bin/mdp-cli health report
```

### Running Tests

```bash
# Run component management tests
./bin/mdp-cli test component

# Run connectivity validation tests
./bin/mdp-cli test connectivity

# Run ZMQ messaging tests
./bin/mdp-cli test zmq

# Run data warehousing tests
./bin/mdp-cli test warehousing

# Run all tests
./bin/mdp-cli test all

# Or directly via Robot Framework
robot robot_framework/test_suites/system_tests/component_management.robot
robot robot_framework/test_suites/system_tests/connectivity_validation.robot
robot robot_framework/test_suites/system_tests/zmq_messaging_tests.robot
robot robot_framework/test_suites/system_tests/data_warehousing_tests.robot
```

### System Operations

```bash
# Full system startup
./bin/start.sh

# Full system shutdown
./bin/stop.sh

# Or via CLI
./bin/mdp-cli system init
./bin/mdp-cli system shutdown
./bin/mdp-cli system restart

# Verify system
./bin/mdp-cli system verify
```

## Configuration

### Component Definitions

Defined in `lib/component_manager.sh`:

```bash
COMPONENTS[database]="docker:postgres redis"
COMPONENTS[messaging]="binary:zmq-publisher zmq-subscriber"
COMPONENTS[api]="docker:python-api"
# etc...
```

### Dependencies

```bash
DEPENDS_ON[monitoring]="database"
DEPENDS_ON[api]="database"
DEPENDS_ON[processor]="database messaging"
DEPENDS_ON[proxy]="api gateway"
```

### Health Checks

Defined per component with service-specific checks:
- HTTP endpoints: GET with 200 status
- PostgreSQL: `pg_isready`
- Redis: `redis-cli ping`
- ZMQ: Socket connection test

## Logging

### Log Locations

- ZMQ Publisher: `logs/publisher.log`
- ZMQ Subscriber: `logs/subscriber.log`
- Docker Services: `docker-compose logs -f <service>`
- Robot Tests: `results/<test_suite>/output.xml`

### Viewing Logs

```bash
# Follow publisher logs
tail -f logs/publisher.log

# View docker service logs
docker-compose logs -f postgres
docker-compose logs -f redis

# View test results
robot --outputdir results robot_framework/test_suites/
```

## Graceful Shutdown

### SIGTERM Handling

1. Services receive SIGTERM signal
2. 5-second grace period for cleanup
3. Optional SIGKILL if not stopped

### Clean Shutdown Process

```bash
# Via script
bash bin/stop.sh

# Via CLI
./bin/mdp-cli system shutdown

# Via component manager
bash lib/component_manager.sh stop [component]
```

## Monitoring & Metrics

### Real-time Status

```bash
./bin/mdp-cli component status
./bin/mdp-cli health check
```

### Performance Benchmarking

```bash
./bin/mdp-cli benchmark services  # All services
./bin/mdp-cli benchmark api_python  # Specific service
```

### Connectivity Report

```bash
./bin/mdp-cli health report  # Detailed report
./bin/mdp-cli validate connectivity  # Validation output
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
tail -f logs/*.log

# Validate dependencies
./bin/mdp-cli component status

# Validate connectivity
./bin/mdp-cli validate connectivity

# Check ports
netstat -tuln | grep -E ':(8000|8080|5555|5556|6379|8086|3000|9090|5432)'
```

### Graceful Shutdown Not Working

```bash
# Check running processes
ps aux | grep -E 'publisher|subscriber|docker'

# Force stop via component manager
bash lib/component_manager.sh stop [component]

# Docker cleanup
docker-compose down --remove-orphans
```

### Connectivity Issues

```bash
# Detailed connectivity report
./bin/mdp-cli health report

# Check specific service
./bin/mdp-cli validate service database_postgres

# Wait for service to be ready
./bin/mdp-cli wait service database_postgres 60s
```

## Performance Characteristics

- **Component Startup:** ~5-10 seconds per group
- **Health Validation:** ~1-2 seconds per component
- **Service Response Time:** < 500ms (typical)
- **Graceful Shutdown:** ~10-15 seconds (all components)

## Best Practices

1. **Always use component groups** - Respects dependencies automatically
2. **Validate connectivity** - After startup, run connectivity validation
3. **Use graceful shutdown** - Give services time to cleanup
4. **Monitor logs** - Check logs for issues during startup
5. **Run tests regularly** - Component and connectivity tests catch issues
6. **Use health reports** - For diagnostics and performance tracking

## Future Enhancements

- Kubernetes integration for orchestration
- Component scaling (multiple instances)
- Advanced health check policies
- Persistent connectivity metrics
- Component performance profiles
- Automated recovery policies

---

**Last Updated:** January 2026  
**Version:** 2.0 - Refactored Component Management System
