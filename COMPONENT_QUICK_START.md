# Component Management System - Quick Start

## 🚀 Quick Start (5 minutes)

### 1. Start the System
```bash
cd /root/rf_env

# Option A: Full system startup with component management
./bin/mdp-cli system init

# Option B: Traditional method (still works)
bash bin/start.sh
```

### 2. Validate Connectivity
```bash
# Check all services
./bin/mdp-cli validate connectivity

# Get health report
./bin/mdp-cli health check

# Detailed report
./bin/mdp-cli health report
```

### 3. Check Component Status
```bash
./bin/mdp-cli component status
```

### 4. Run Tests
```bash
# Quick test - component management
./bin/mdp-cli test component

# Full test suite
./bin/mdp-cli test all
```

## 📋 Command Reference

### Component Operations
```bash
# Start
./bin/mdp-cli component start database           # Start database
./bin/mdp-cli component start messaging api      # Start multiple
./bin/mdp-cli component start                     # Start all

# Stop
./bin/mdp-cli component stop database
./bin/mdp-cli component stop

# Status
./bin/mdp-cli component status

# Restart
./bin/mdp-cli component restart database

# View Logs
./bin/mdp-cli component logs publisher
```

### Connectivity Validation
```bash
# All services
./bin/mdp-cli validate connectivity

# Specific service
./bin/mdp-cli validate service database_postgres

# By category
./bin/mdp-cli validate database        # PostgreSQL + Redis + InfluxDB
./bin/mdp-cli validate messaging       # ZMQ Publisher + Subscriber
```

### Health & Monitoring
```bash
# Quick check
./bin/mdp-cli health check

# Detailed report
./bin/mdp-cli health report

# Benchmarking
./bin/mdp-cli benchmark services       # All services
./bin/mdp-cli benchmark api_python     # Specific service
```

### Testing
```bash
# Individual test suites
./bin/mdp-cli test component           # Component management
./bin/mdp-cli test connectivity        # Connectivity validation
./bin/mdp-cli test zmq                 # ZMQ messaging
./bin/mdp-cli test warehousing         # Data warehousing

# All tests
./bin/mdp-cli test all
```

### System Operations
```bash
# Full lifecycle
./bin/mdp-cli system init              # Startup
./bin/mdp-cli system verify            # Verify
./bin/mdp-cli system shutdown          # Shutdown
./bin/mdp-cli system restart           # Restart

# Traditional methods (still work)
bash bin/start.sh
bash bin/stop.sh
bash bin/verify_services.sh
```

### Database Access
```bash
# PostgreSQL
./bin/mdp-cli db shell

# Redis
./bin/mdp-cli db redis

# InfluxDB
./bin/mdp-cli db influx
```

## 🏗️ Component Groups

### Database Layer
```bash
./bin/mdp-cli component start database
# Starts: PostgreSQL (5432) + Redis (6379)
```

### Storage Layer
```bash
./bin/mdp-cli component start storage
# Starts: InfluxDB (8086)
```

### Monitoring Layer
```bash
./bin/mdp-cli component start monitoring
# Starts: Prometheus (9090) + Grafana (3000)
```

### Messaging Layer
```bash
./bin/mdp-cli component start messaging
# Starts: ZMQ Publisher (5555) + Subscriber (5556)
```

### Application Services
```bash
./bin/mdp-cli component start api           # Python API (8000)
./bin/mdp-cli component start gateway       # Go Gateway (8080)
./bin/mdp-cli component start processor     # Rust Processor
```

### Proxy Layer
```bash
./bin/mdp-cli component start proxy
# Starts: Nginx (80/443)
```

## 🔗 Service Endpoints

| Service | URL | User/Pass |
|---------|-----|-----------|
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | N/A |
| Python API | http://localhost:8000 | N/A |
| API Docs | http://localhost:8000/docs | N/A |
| Go Gateway | http://localhost:8080 | N/A |

## 📊 Common Workflows

### Scenario 1: Development with ZMQ
```bash
# Start just what you need
./bin/mdp-cli component start database
./bin/mdp-cli component start messaging

# Validate
./bin/mdp-cli validate messaging

# Run ZMQ tests
./bin/mdp-cli test zmq

# Stop when done
./bin/mdp-cli component stop messaging database
```

### Scenario 2: Full System Test
```bash
# Complete startup
./bin/mdp-cli system init

# Wait for services
sleep 5

# Validate
./bin/mdp-cli validate connectivity

# Run all tests
./bin/mdp-cli test all

# Check results
./bin/mdp-cli health report
```

### Scenario 3: Data Warehousing Work
```bash
# Start warehousing components
./bin/mdp-cli component start database
./bin/mdp-cli component start storage

# Validate
./bin/mdp-cli validate database

# Run warehousing tests
./bin/mdp-cli test warehousing

# Access databases
./bin/mdp-cli db shell        # PostgreSQL
./bin/mdp-cli db redis        # Redis cache
./bin/mdp-cli db influx       # InfluxDB
```

### Scenario 4: Troubleshooting
```bash
# Check status
./bin/mdp-cli component status

# Detailed health report
./bin/mdp-cli health report

# Validate specific service
./bin/mdp-cli validate service database_postgres

# View logs
./bin/mdp-cli component logs publisher

# Wait for service
./bin/mdp-cli wait service database_postgres 60s
```

## 🧪 Running Robot Framework Tests Directly

```bash
# Component management tests
robot robot_framework/test_suites/system_tests/component_management.robot

# Connectivity tests
robot robot_framework/test_suites/system_tests/connectivity_validation.robot

# ZMQ tests
robot robot_framework/test_suites/system_tests/zmq_messaging_tests.robot

# Data warehousing tests
robot robot_framework/test_suites/system_tests/data_warehousing_tests.robot

# With tag filtering
robot --include "component" robot_framework/test_suites/system_tests/

# View results
firefox results/report.html
```

## 📝 Key Features

✅ **Graceful Start/Stop**
- Components start in dependency order
- Components stop in reverse order
- Health validation between each step
- SIGTERM-based graceful shutdown

✅ **Connectivity Validation**
- Validates all service endpoints
- Measures response times
- Provides health reports
- Detailed error reporting

✅ **ZMQ Messaging**
- Publisher validation
- Subscriber validation
- Automatic compilation
- Response time benchmarking

✅ **Data Warehousing**
- PostgreSQL validation
- Redis cache validation
- InfluxDB storage validation
- Multi-component coordination

✅ **Robot Framework**
- 30+ reusable keywords
- 80+ test cases
- Tag-based filtering
- Detailed reporting

## 🆘 Troubleshooting

### "Service won't start"
```bash
# Check connectivity
./bin/mdp-cli validate connectivity

# Check specific component
./bin/mdp-cli component status

# View detailed report
./bin/mdp-cli health report
```

### "Tests failing"
```bash
# Ensure all services are running
./bin/mdp-cli component status

# Validate connectivity
./bin/mdp-cli validate connectivity

# Run simpler test first
./bin/mdp-cli test component
```

### "Graceful shutdown not working"
```bash
# Try component stop
./bin/mdp-cli component stop [component]

# Force stop if needed
docker-compose down

# Clean up
rm -f .pids/*.pid
```

### "Need to see logs"
```bash
# Component logs
./bin/mdp-cli component logs publisher
./bin/mdp-cli component logs subscriber

# Docker logs
docker-compose logs -f postgres
docker-compose logs -f redis

# Test output
cat results/*/log.html
```

## 📚 Documentation

- **Full Guide:** `COMPONENT_MANAGEMENT_GUIDE.md`
- **Summary:** `COMPONENT_REFACTORING_SUMMARY.md`
- **CLI Help:** `./bin/mdp-cli help`
- **Component Help:** `./bin/mdp-cli help component`
- **Validation Help:** `./bin/mdp-cli help validate`

## ⚡ Pro Tips

1. **Use component groups** - Automatically handles dependencies
2. **Always validate** - After startup, run connectivity check
3. **Check health reports** - For diagnostics and performance
4. **Run tests regularly** - Catches issues early
5. **Use tags** - Filter Robot tests by category

## 📞 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Port already in use | Stop competing process or change port |
| Service timeout | Check connectivity: `./bin/mdp-cli health report` |
| Component won't stop | Force stop: `docker-compose kill` |
| ZMQ compilation fails | Install: `apt-get install libzmq3-dev` |
| Tests hang | Check logs and run connectivity validation |

---

**Quick Start Version:** 2.0  
**Last Updated:** January 2026  
**Status:** ✅ Ready to Use
