# Market Data Platform - Component Management System

**Generated**: January 16, 2026  
**Status**: ✅ PRODUCTION READY  
**Version**: 2.0 - Enhanced with Dependencies, Installation, Uninstallation, and Rich Status

---

## Executive Summary

The Market Data Platform now includes a comprehensive component management system with:
- **Automatic Dependency Resolution** - Services install/start in correct order
- **Installation & Uninstallation** - Full lifecycle management with data cleanup
- **Graceful Shutdown** - Reverse-order shutdown with signal handling
- **Rich Terminal UI** - Color-coded status display with real-time monitoring
- **Multiple Interfaces** - Bash CLI, Python CLI, and Interactive Terminal Dashboard

---

## Quick Start

### Installation and Starting Services

```bash
# Install all services (dependencies resolved automatically)
./bin/mdp-cli component install

# Or install specific services
./bin/mdp-cli component install database
./bin/mdp-cli component install api gateway

# Start installed services
./bin/mdp-cli component start

# View rich status dashboard
./bin/mdp-cli component status
```

### Rich Status Display

```bash
# Show interactive dashboard
./bin/mdp-terminal

# Or use direct status command
./bin/mdp-status dashboard
./bin/mdp-status health
./bin/mdp-status services
```

### Graceful Shutdown

```bash
# Graceful shutdown (recommended)
./bin/mdp-cli component graceful-stop

# Or stop all components
./bin/mdp-cli component stop

# Individual component stop
./bin/mdp-cli component stop proxy
```

### Uninstallation

```bash
# Uninstall all services
./bin/mdp-cli component uninstall

# Uninstall with data cleanup
./bin/mdp-cli component uninstall --remove-data

# Uninstall specific service
./bin/mdp-cli component uninstall redis
```

---

## Component Architecture

### Service Definitions (config/services.yml)

Each service includes:
- **Type**: docker or binary
- **Port**: Service listening port
- **Startup Command**: How to start the service
- **Shutdown Command**: How to stop the service
- **Health Check**: Validation command
- **Dependencies**: Services that must be running first
- **Timeouts**: Startup/shutdown timeouts

Example:
```yaml
postgres:
  name: "PostgreSQL Database"
  type: "docker"
  port: 5432
  depends_on: []
  startup_cmd: "docker-compose up -d postgres"
  shutdown_cmd: "docker-compose stop postgres"
  health_check: "pg_isready -h localhost -U mdp_user"
```

### Component Groups

Services are organized into logical groups:

| Component | Services | Purpose |
|-----------|----------|---------|
| **database** | postgres, redis | Core data storage and caching |
| **storage** | influxdb | Time-series data warehousing |
| **monitoring** | prometheus, grafana | Metrics collection and visualization |
| **messaging** | zmq-publisher, zmq-subscriber | Asynchronous message passing |
| **api** | python-api | REST API services |
| **gateway** | go-gateway | Protocol translation and routing |
| **processor** | rust-processor | Data processing and computation |
| **proxy** | nginx | Reverse proxy and load balancing |

### Dependency Graph

```
database (postgres, redis)
    ↓
storage (influxdb)
    ↓
monitoring (prometheus, grafana)
    ↓
messaging (zmq)
    ↓
api (python-api)
    ↓
gateway (go-gateway)
    ↓
processor (rust-processor)
    ↓
proxy (nginx)
```

---

## System Interfaces

### 1. Bash CLI (mdp-cli)

Primary command-line interface with rich output.

#### Component Management

```bash
# Install
./bin/mdp-cli component install [SERVICE...]    # Install components
./bin/mdp-cli component install                 # Install all

# Start
./bin/mdp-cli component start [SERVICE...]      # Start components
./bin/mdp-cli component start                   # Start all

# Stop
./bin/mdp-cli component stop [SERVICE...]       # Stop components
./bin/mdp-cli component graceful-stop           # Graceful shutdown
./bin/mdp-cli component stop                    # Stop all

# Uninstall
./bin/mdp-cli component uninstall [SERVICE...] [--remove-data]
./bin/mdp-cli component uninstall --remove-data # Uninstall all with cleanup

# Status
./bin/mdp-cli component status                  # Show rich dashboard
./bin/mdp-cli component restart [SERVICE]       # Restart service
./bin/mdp-cli component logs [SERVICE]          # View service logs
```

#### Validation & Health

```bash
# Connectivity validation
./bin/mdp-cli validate connectivity
./bin/mdp-cli validate service api
./bin/mdp-cli validate database
./bin/mdp-cli validate messaging
./bin/mdp-cli validate all

# Health reporting
./bin/mdp-cli health check                      # Quick status
./bin/mdp-cli health report                     # Detailed report
```

#### System Operations

```bash
# System management
./bin/mdp-cli system init                       # Initialize system
./bin/mdp-cli system verify                     # Verify all systems
./bin/mdp-cli system shutdown                   # Graceful shutdown
./bin/mdp-cli system restart                    # Full restart
```

### 2. Python Component Manager (mdp-components)

Direct Python interface for programmatic access.

```bash
# CLI wrapper
./bin/mdp-components install database
./bin/mdp-components install --all
./bin/mdp-components start postgres
./bin/mdp-components stop --all
./bin/mdp-components status
./bin/mdp-components uninstall redis --remove-data
./bin/mdp-components shutdown
```

### 3. Rich Status Display (mdp-status)

Detailed status and health reporting with rich formatting.

```bash
# Display modes
./bin/mdp-status dashboard                      # Full dashboard
./bin/mdp-status services                       # Service table
./bin/mdp-status components                     # Component table
./bin/mdp-status health                         # Health report
./bin/mdp-status logs --service postgres        # Service logs

# Options
./bin/mdp-status logs --service api --lines 100 # Last 100 lines
```

### 4. Interactive Terminal UI (mdp-terminal)

Advanced interactive dashboard with keyboard navigation.

```bash
# Launch interactive dashboard
./bin/mdp-terminal

# Menu Options:
# 1 - Status Dashboard
# 2 - Install Components
# 3 - Uninstall Components
# 4 - Start Services
# 5 - Stop Services
# 6 - Restart Services
# 7 - Health Report
# 8 - View Logs
# 9 - Configuration
# 0 - Exit
```

Features:
- **Real-time Status** - Live service monitoring
- **Interactive Installation** - Step-by-step component setup
- **Progress Display** - Visual feedback during operations
- **Dependency Awareness** - Automatic dependency resolution
- **Batch Operations** - Install/stop all services at once

---

## State Management

### State File (.component_state.json)

Tracks component status and lifecycle:

```json
{
  "components": {},
  "services": {
    "postgres": {
      "state": "running",
      "installed_at": "2026-01-16T10:30:00Z",
      "started_at": "2026-01-16T10:30:05Z"
    },
    "redis": {
      "state": "running",
      "installed_at": "2026-01-16T10:30:10Z",
      "started_at": "2026-01-16T10:30:15Z"
    }
  },
  "last_updated": "2026-01-16T10:30:20Z"
}
```

### Logging

All operations logged to `logs/component_manager.log`:
- Installation attempts
- Start/stop operations
- Health check results
- Error conditions
- Graceful shutdown sequence

---

## Dependency Resolution

### Automatic Installation Order

Dependencies are resolved automatically when installing services:

```
Install database (no deps)
  ↓
Install storage (depends on database)
  ↓
Install monitoring (depends on database)
  ↓
Install messaging (depends on database, storage)
  ↓
Install api (depends on database, storage, messaging)
  ↓
Install gateway (depends on api, database)
  ↓
Install processor (depends on gateway, messaging)
  ↓
Install proxy (depends on api, gateway, processor)
```

### Startup Validation

Before starting any service:
1. Check if service is installed
2. Verify all dependencies are running
3. Execute startup command
4. Wait for startup timeout
5. Run health check
6. Update state file

### Graceful Shutdown

Services stop in reverse dependency order:
```
Stop proxy (no dependents)
  ↓
Stop processor (only proxy depends)
  ↓
Stop gateway (processor and api depend)
  ↓
Stop api (processor depends)
  ↓
Stop messaging (api and processor depend)
  ↓
Stop monitoring (gateway and processor depend)
  ↓
Stop storage (monitoring depends)
  ↓
Stop database (all depend)
```

---

## Health Checks

Each service includes a health check command that validates:
- **Connectivity** - Port is open and responding
- **Service Status** - Application is running
- **Readiness** - Application is ready for requests

Examples:
```bash
# PostgreSQL
pg_isready -h localhost -U mdp_user

# Redis
redis-cli ping

# HTTP services
curl -s http://localhost:8000/health | grep -q '"status":"healthy"'

# ZMQ services
timeout 2 bash -c 'echo | nc localhost 5555'
```

---

## Installation & Uninstallation

### Installing Services

```bash
# Install single service (installs dependencies automatically)
python3 -m market_data_platform.cli.component_manager install postgres

# Install multiple services
python3 -m market_data_platform.cli.component_manager install postgres redis influxdb

# Install all services
python3 -m market_data_platform.cli.component_manager install --all
```

**Process**:
1. Check dependencies
2. Install missing dependencies first
3. Run startup command
4. Wait for service to be ready
5. Verify health check
6. Update state to "running"

### Uninstalling Services

```bash
# Uninstall service (graceful stop first)
python3 -m market_data_platform.cli.component_manager uninstall postgres

# Uninstall with data cleanup
python3 -m market_data_platform.cli.component_manager uninstall postgres --remove-data

# Uninstall all services
python3 -m market_data_platform.cli.component_manager uninstall --all --remove-data
```

**Process**:
1. Stop the service (graceful shutdown)
2. Run shutdown command
3. Clean up data (if --remove-data flag)
4. Remove from state file
5. Log completion

**Data Cleanup**:
- PostgreSQL: `.pgdata` directory
- Redis: `.redis_data` directory
- InfluxDB: `.influx_data` directory

---

## Graceful Shutdown

### Why Graceful Shutdown Matters

Graceful shutdown allows services to:
- Flush pending writes
- Close database connections
- Save state
- Notify connected clients
- Clean up resources

### Implementation

```bash
# Graceful shutdown (recommended)
./bin/mdp-cli component graceful-stop

# Or via Python
python3 -m market_data_platform.cli.component_manager shutdown

# Or via terminal UI
./bin/mdp-terminal  # Select option 5, then "Graceful shutdown"
```

**Shutdown Sequence**:
1. Send SIGTERM to services
2. Wait for graceful shutdown (timeout: 30s per service)
3. If service still running after timeout, force kill with SIGKILL
4. Verify service stopped
5. Update state to "stopped"
6. Log shutdown completion

---

## Error Handling & Recovery

### Common Issues

**Installation Fails**
```
Solution: Check logs in logs/component_manager.log
         Verify dependencies are installed
         Check port availability
         Review startup command
```

**Service Won't Start**
```
Solution: Check health check is passing
         Verify port is not in use: lsof -i :PORT
         Review service logs: ./bin/mdp-status logs --service SERVICE
         Check configuration files
```

**Graceful Stop Timeout**
```
Solution: Service will be force killed
         Logs will show timeout warning
         Review service logs for why stop is slow
         May indicate resource exhaustion or deadlock
```

### Recovery Procedures

**Reset State**
```bash
# Clear state file to forget service status
rm .component_state.json

# Services will be marked as "unknown" until next operation
```

**Force Cleanup**
```bash
# Force stop all services
pkill -f 'docker-compose'

# Uninstall all with data cleanup
./bin/mdp-cli component uninstall --all --remove-data
```

---

## Advanced Usage

### Custom Service Definitions

Add new services to `config/services.yml`:

```yaml
services:
  my-service:
    name: "My Custom Service"
    type: "docker"
    container: "my-container"
    port: 9000
    depends_on:
      - postgres
      - redis
    startup_cmd: "docker-compose up -d my-container"
    shutdown_cmd: "docker-compose stop my-container"
    health_check: "curl -s http://localhost:9000/health"
    startup_timeout: 30
    shutdown_timeout: 10
```

### Programmatic API

```python
from market_data_platform.cli.component_manager import ComponentManager

# Create manager
manager = ComponentManager(project_root=".")

# Install service
manager.install("postgres")

# Start service
manager.start("redis")

# Check status
status = manager.status()
for service, info in status["services"].items():
    print(f"{service}: {info['state']}")

# Graceful shutdown
manager.graceful_shutdown()

# Get health report
report = manager.show_health_report()
manager.print_health_report(report)
```

### Rich Status Display API

```python
from market_data_platform.cli.rich_status import RichStatusDisplay

# Create display
display = RichStatusDisplay(project_root=".")

# Show dashboard
display.show_dashboard()

# Show health report
report = display.show_health_report()
display.print_health_report(report)

# Show service logs
display.show_service_logs("postgres", lines=50)

# Show health table
display.show_service_table()
display.show_component_table()
```

---

## Monitoring & Debugging

### View Logs

```bash
# Component manager logs
tail -f logs/component_manager.log

# Service-specific logs
tail -f logs/postgres.log
tail -f logs/api.log

# Docker container logs
docker-compose logs -f postgres
docker-compose logs -f api

# Rich status display
./bin/mdp-status logs --service postgres --lines 100
```

### Check Service Status

```bash
# All services
./bin/mdp-cli component status

# Connectivity validation
./bin/mdp-cli validate connectivity

# Health report
./bin/mdp-cli health report

# Individual service check
python3 -c "from market_data_platform.cli.component_manager import ComponentManager; m=ComponentManager(); print(m.status())"
```

---

## Performance Optimization

### Parallel Installation

Modify `lib/component_manager_enhanced.sh` to support parallel installation:

```bash
# Install non-dependent services in parallel
install_service database &
install_service storage &
wait
```

### Startup Timeouts

Adjust timeouts in `config/services.yml` based on your environment:
- Fast/SSD: 15-20 seconds
- Slow/HDD: 30-60 seconds
- Cloud: 60-120 seconds

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Install Services
  run: |
    ./bin/mdp-cli component install --all

- name: Health Check
  run: |
    ./bin/mdp-cli health report

- name: Run Tests
  run: |
    ./bin/mdp-cli test all
```

### Docker Compose Integration

Services integrate with existing `docker-compose.yml`:
```bash
./bin/mdp-cli component install  # Uses docker-compose commands
docker-compose ps                # Verify services
```

---

## Troubleshooting

### Issue: Services won't start

```bash
# 1. Check logs
tail -f logs/component_manager.log

# 2. Verify dependencies
./bin/mdp-cli component status

# 3. Check ports
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
lsof -i :8000  # API

# 4. Validate connectivity
./bin/mdp-cli validate connectivity
```

### Issue: Uninstall fails

```bash
# 1. Force stop service
pkill -9 -f service_name

# 2. Clean up state
rm .component_state.json

# 3. Retry uninstall
./bin/mdp-cli component uninstall service_name --remove-data
```

### Issue: Graceful shutdown timeout

```bash
# 1. Check service activity
docker-compose logs service_name

# 2. Reduce timeout in config/services.yml

# 3. Force shutdown
pkill -9 -f service_name
```

---

## Best Practices

1. **Always use graceful shutdown** - Prevents data corruption
2. **Check health before operations** - Verify system readiness
3. **Monitor logs during startup** - Catch issues early
4. **Review dependencies** - Understand service relationships
5. **Test uninstall/reinstall** - Verify cleanup procedures
6. **Use state file for auditing** - Track installation history
7. **Backup data before uninstall** - Never use --remove-data lightly

---

## Support & Documentation

- **CLI Help**: `./bin/mdp-cli help [COMMAND]`
- **Component Status**: `./bin/mdp-cli component status`
- **Health Report**: `./bin/mdp-cli health report`
- **Logs**: `logs/component_manager.log`
- **State**: `.component_state.json`

---

**Status**: ✅ READY FOR PRODUCTION  
**Last Updated**: January 16, 2026
