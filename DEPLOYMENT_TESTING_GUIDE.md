# Component Management System - Deployment & Testing Guide

**Date**: January 16, 2026  
**Status**: ✅ READY FOR DEPLOYMENT  
**Version**: 2.0

---

## Pre-Deployment Checklist

### System Requirements

- [ ] Python 3.9+
- [ ] Docker & Docker Compose
- [ ] Bash 4.0+
- [ ] Required Python packages installed

### Verification

```bash
# Check Python version
python3 --version  # Should be 3.9+

# Check Docker
docker --version
docker-compose --version

# Check Bash
bash --version  # Should be 4.0+

# Check Python packages
python3 -c "import yaml; import rich; print('✓ Dependencies OK')"
```

---

## Installation Steps

### 1. Install Python Dependencies

```bash
# Install required packages
pip install --no-cache-dir -r requirements.txt

# Or specific packages for component management
pip install pyyaml rich typer
```

### 2. Configure Services

Verify `config/services.yml` exists with service definitions:

```bash
ls -lh config/services.yml
# Should show: -rw-r--r-- ... config/services.yml
```

### 3. Create Required Directories

```bash
# Create necessary directories
mkdir -p logs .pids .component_state
chmod 755 logs .pids

# Verify
ls -la | grep -E '^d.*logs|^d.*\.pids'
```

### 4. Initialize System

```bash
# Initialize component state
./bin/mdp-cli system init

# Or initialize via Python
python3 -m market_data_platform.cli.component_manager status
```

---

## Deployment Scenarios

### Scenario 1: Fresh Installation (All Services)

```bash
# Step 1: Install all components
./bin/mdp-cli component install

# Expected output:
# → Installing database...
# ✓ postgres installed and started
# → Installing redis...
# ✓ redis installed and started
# ... (continues for all services)

# Step 2: Verify status
./bin/mdp-cli component status

# Step 3: Run health check
./bin/mdp-cli health report
```

**Estimated Time**: 5-15 minutes (depends on docker image sizes)

### Scenario 2: Selective Installation

```bash
# Install only core services
./bin/mdp-cli component install database storage

# Install API services
./bin/mdp-cli component install api gateway

# Install monitoring
./bin/mdp-cli component install monitoring

# Verify
./bin/mdp-cli component status
```

### Scenario 3: Production Deployment

```bash
#!/bin/bash
# deployment.sh - Production deployment script

set -euo pipefail

echo "=== Production Deployment ==="
echo ""

# 1. Install core infrastructure
echo "Step 1: Installing database layer..."
./bin/mdp-cli component install database storage
./bin/mdp-cli health check

# 2. Install monitoring
echo "Step 2: Installing monitoring stack..."
./bin/mdp-cli component install monitoring

# 3. Install API services
echo "Step 3: Installing API services..."
./bin/mdp-cli component install api gateway

# 4. Install processing
echo "Step 4: Installing data processor..."
./bin/mdp-cli component install processor

# 5. Install proxy
echo "Step 5: Installing proxy layer..."
./bin/mdp-cli component install proxy

# 6. Final validation
echo "Step 6: Final validation..."
./bin/mdp-cli validate all
./bin/mdp-cli health report

echo ""
echo "✓ Production deployment complete"
```

---

## Testing Procedures

### Unit Tests

```bash
# Component Manager Tests
python3 -m pytest tests/test_component_manager.py -v

# Rich Status Display Tests
python3 -m pytest tests/test_rich_status.py -v

# Terminal UI Tests
python3 -m pytest tests/test_terminal_ui.py -v
```

### Integration Tests

```bash
# Test installation and uninstallation
./test_install_uninstall.sh

# Test graceful shutdown
./test_graceful_shutdown.sh

# Test dependency resolution
./test_dependencies.sh
```

### System Tests

```bash
# Run all component management tests
./bin/mdp-cli test component

# Run connectivity tests
./bin/mdp-cli test connectivity

# Run ZMQ messaging tests
./bin/mdp-cli test zmq

# Run data warehousing tests
./bin/mdp-cli test warehousing

# Run all tests
./bin/mdp-cli test all
```

### Manual Testing Checklist

#### Installation Tests
- [ ] Install database component
- [ ] Verify PostgreSQL is running
- [ ] Verify Redis is running
- [ ] Check state file updated

#### Start/Stop Tests
- [ ] Start all components
- [ ] Verify all services running
- [ ] Stop specific component
- [ ] Verify component stopped
- [ ] Graceful stop all
- [ ] Verify all stopped

#### Uninstall Tests
- [ ] Uninstall service
- [ ] Verify service stopped
- [ ] Uninstall with data cleanup
- [ ] Verify data deleted
- [ ] Reinstall and verify works

#### Status Display Tests
- [ ] Run `./bin/mdp-cli component status`
- [ ] Run `./bin/mdp-status dashboard`
- [ ] Run `./bin/mdp-terminal`
- [ ] Verify all output formatted correctly

#### Error Handling Tests
- [ ] Try to start service with missing dependency
- [ ] Try to uninstall non-existent service
- [ ] Kill service during shutdown
- [ ] Verify error messages are clear

---

## Testing Scripts

### test_install_uninstall.sh

```bash
#!/bin/bash
set -euo pipefail

echo "=== Testing Installation & Uninstallation ==="

# Test 1: Install single service
echo "Test 1: Install single service (postgres)"
./bin/mdp-cli component install postgres
sleep 2

# Verify installed
if ./bin/mdp-cli component status | grep -q "postgres"; then
    echo "✓ Service installed successfully"
else
    echo "✗ Service installation failed"
    exit 1
fi

# Test 2: Uninstall service
echo "Test 2: Uninstall service"
./bin/mdp-cli component uninstall postgres
sleep 1

# Verify uninstalled
if ! ./bin/mdp-cli component status | grep -q "running.*postgres"; then
    echo "✓ Service uninstalled successfully"
else
    echo "✗ Service uninstallation failed"
    exit 1
fi

echo ""
echo "=== All tests passed ==="
```

### test_graceful_shutdown.sh

```bash
#!/bin/bash
set -euo pipefail

echo "=== Testing Graceful Shutdown ==="

# Install services
echo "Installing test services..."
./bin/mdp-cli component install database

# Wait for startup
sleep 5

# Test graceful shutdown
echo "Testing graceful shutdown..."
timeout 120 ./bin/mdp-cli component graceful-stop

# Verify all stopped
sleep 2
if ! docker-compose ps --services --filter "status=running" | grep -q .; then
    echo "✓ Graceful shutdown successful"
else
    echo "✗ Some services still running"
    exit 1
fi

echo ""
echo "=== Graceful shutdown test passed ==="
```

### test_dependencies.sh

```bash
#!/bin/bash
set -euo pipefail

echo "=== Testing Dependency Resolution ==="

# Install service with dependencies
echo "Installing api (has dependencies)..."
./bin/mdp-cli component install api

# Verify all dependencies installed
echo "Verifying dependencies..."
for dep in postgres redis influxdb; do
    if ./bin/mdp-cli component status | grep -q "$dep"; then
        echo "✓ Dependency installed: $dep"
    else
        echo "✗ Dependency missing: $dep"
        exit 1
    fi
done

echo ""
echo "=== Dependency resolution test passed ==="
```

---

## Performance Testing

### Startup Performance

```bash
#!/bin/bash
# Measure startup time for each service

echo "=== Startup Performance Test ==="
echo ""

services=(postgres redis influxdb prometheus api gateway processor proxy)

for service in "${services[@]}"; do
    echo "Testing $service..."
    
    START=$(date +%s%N)
    ./bin/mdp-cli component install "$service" > /dev/null 2>&1
    END=$(date +%s%N)
    
    DURATION=$((($END - $START) / 1000000))  # Convert to ms
    echo "$service: ${DURATION}ms"
    
    ./bin/mdp-cli component uninstall "$service" > /dev/null 2>&1
done
```

### Shutdown Performance

```bash
#!/bin/bash
# Measure graceful shutdown time

echo "=== Graceful Shutdown Performance Test ==="

./bin/mdp-cli component install

START=$(date +%s)
./bin/mdp-cli component graceful-stop
END=$(date +%s)

DURATION=$((END - START))
echo "Total shutdown time: ${DURATION}s"

if [ $DURATION -lt 120 ]; then
    echo "✓ Shutdown performance acceptable (< 120s)"
else
    echo "⚠ Shutdown took longer than expected (> 120s)"
fi
```

---

## Rollback Procedures

### Emergency Rollback

```bash
# Force stop all services
docker-compose kill
pkill -9 -f market_data_platform

# Restore previous state
rm .component_state.json
git checkout config/

# Restart from clean state
./bin/mdp-cli component install
```

### Selective Rollback

```bash
# Uninstall problematic service
./bin/mdp-cli component uninstall api

# Verify system stability
./bin/mdp-cli health report

# Reinstall service
./bin/mdp-cli component install api
```

---

## Monitoring During Deployment

### Real-time Monitoring

```bash
# Terminal 1: Watch status
watch -n 1 './bin/mdp-cli component status'

# Terminal 2: Monitor logs
tail -f logs/component_manager.log

# Terminal 3: Interactive dashboard
./bin/mdp-terminal
```

### Log Monitoring

```bash
# Filter installation logs
grep "Installing" logs/component_manager.log

# Filter errors
grep "ERROR\|Failed" logs/component_manager.log

# Real-time errors
tail -f logs/component_manager.log | grep -i error
```

---

## Validation After Deployment

### Post-Deployment Checklist

- [ ] All services installed
- [ ] All services running
- [ ] Health checks passing
- [ ] Connectivity validation successful
- [ ] API endpoints responding
- [ ] Database connections working
- [ ] Cache operational
- [ ] Monitoring active
- [ ] Logs being generated
- [ ] State file updated

### Health Verification

```bash
# Comprehensive health check
./bin/mdp-cli health report

# Service connectivity
./bin/mdp-cli validate connectivity

# Database operations
./bin/mdp-cli db shell
# Run: SELECT 1;

# Cache operations
./bin/mdp-cli db redis
# Run: PING
```

---

## Troubleshooting During Deployment

### Issue: Services fail to start

```bash
# 1. Check Docker
docker ps -a
docker-compose logs

# 2. Check logs
tail -f logs/component_manager.log

# 3. Verify configuration
cat config/services.yml | grep -A 5 "failing_service"

# 4. Manual startup test
docker-compose up -d failing_service
docker-compose logs failing_service
```

### Issue: Dependency not satisfied

```bash
# 1. Check state file
cat .component_state.json | python3 -m json.tool

# 2. Verify dependency is installed
./bin/mdp-cli component status

# 3. Start dependency manually
./bin/mdp-cli component install postgres
./bin/mdp-cli component start api
```

### Issue: Installation timeout

```bash
# 1. Check if service is starting
docker-compose ps

# 2. Increase timeout in config/services.yml
vim config/services.yml  # Increase startup_timeout

# 3. Check resource usage
docker stats

# 4. Retry installation
./bin/mdp-cli component install service_name
```

---

## Documentation

### Generated Files

- `COMPONENT_MANAGEMENT_SYSTEM.md` - System overview and usage
- `CLI_QUICK_REFERENCE.md` - Quick CLI reference
- `.component_state.json` - Service state tracking
- `logs/component_manager.log` - Detailed operation log

### Running Help

```bash
# CLI help
./bin/mdp-cli help

# Component help
./bin/mdp-cli help component

# Validation help
./bin/mdp-cli help validate

# Python module help
python3 -m market_data_platform.cli.component_manager --help
python3 -m market_data_platform.cli.rich_status --help
```

---

## Support Resources

- **Component Manager**: `market_data_platform/cli/component_manager.py`
- **Rich Status**: `market_data_platform/cli/rich_status.py`
- **Terminal UI**: `market_data_platform/cli/terminal_ui.py`
- **Service Definitions**: `config/services.yml`
- **State File**: `.component_state.json`
- **Logs**: `logs/component_manager.log`

---

## Next Steps

1. **Run Pre-Deployment Checklist** - Verify all requirements
2. **Execute Deployment Script** - Install services
3. **Run Validation Tests** - Confirm all working
4. **Monitor Logs** - Watch for errors
5. **Verify Health** - Check system status
6. **Document Deviations** - Note any issues

---

**Status**: ✅ READY FOR DEPLOYMENT  
**Generated**: January 16, 2026
