# Enhanced CLI - Container Deployment & Window Management Guide

## Overview

The enhanced Market Data Platform CLI now includes:
- **Multi-container runtime support** (Docker, Podman, LXC)
- **Service-specific management** (Install, Start, Stop, Restart, Logs, Health Check)
- **Tmux window groups** for organized terminal layout
- **Best practices deployment** with automatic runtime detection
- **Grouped command organization** for better navigation

---

## Quick Start

### 1. Launch Enhanced CLI
```bash
cd /root/rf_env/market_data_platform
python cli/terminal.py
```

### 2. Check Available Container Runtime
```
MDP> status
```
Detects and displays: Docker, Podman, or LXC

### 3. Deploy All Services
```
MDP> install all
MDP> start all
```

---

## Container Deployment Commands

### Install Services

**Install all services with auto-detected runtime:**
```
MDP> install all
```

**Install specific service:**
```
MDP> install influxdb
MDP> install grafana
MDP> install redis
MDP> install parquet
```

**Install with specific runtime:**
```
MDP> install all --runtime docker
MDP> install all --runtime podman
MDP> install all --runtime lxc
```

### Start Services

**Start all services:**
```
MDP> start all
```

**Start specific service:**
```
MDP> start influxdb
MDP> start grafana
```

**Switch runtime and deploy:**
```
MDP> deploy-docker influxdb grafana
MDP> deploy-podman redis
MDP> deploy-lxc all
```

### Stop Services

**Stop specific service:**
```
MDP> stop grafana
```

**Stop all running services:**
```
MDP> stop all
```

### Restart Services

**Restart with restart command:**
```
MDP> restart influxdb
MDP> restart all
```

---

## Service Management

### Check Status
```
MDP> status
```
Shows:
- Current container runtime
- Running services
- Available services

### View Service Logs
```
MDP> logs influxdb
MDP> logs grafana --lines 100
MDP> logs parquet --lines 50
```

### Health Check
```
MDP> health-check
MDP> health-check influxdb
MDP> health-check grafana
```

### Configure Service
```
MDP> configure-service influxdb
MDP> configure-service grafana
```

Displays configuration template for the selected runtime.

---

## Service Configurations

### InfluxDB

**Docker/Podman:**
- Image: `influxdb:2.7-alpine`
- Port: `8086:8086`
- Volume: `influxdb_data:/var/lib/influxdb`
- Startup: `docker run -d -p 8086:8086 -e INFLUXDB_DB=market_data influxdb:2.7-alpine`

**LXC:**
- Package: `influxdb2`
- Port: `8086`
- Config: `/etc/influxdb2/config.yml`

### Grafana

**Docker/Podman:**
- Image: `grafana/grafana:latest`
- Port: `3000:3000`
- Volume: `grafana_data:/var/lib/grafana`
- Startup: `docker run -d -p 3000:3000 -e GF_SECURITY_ADMIN_PASSWORD=admin grafana/grafana:latest`

**LXC:**
- Package: `grafana`
- Port: `3000`
- Config: `/etc/grafana/grafana.ini`

### Redis

**Docker/Podman:**
- Image: `redis:7-alpine`
- Port: `6379:6379`
- Volume: `redis_data:/data`

**LXC:**
- Package: `redis-server`
- Port: `6379`
- Config: `/etc/redis/redis.conf`

### Parquet

**Docker/Podman:**
- Image: `ubuntu:22.04`
- Port: `9090:9090`
- Setup: `apt-get install -y python3-pyarrow python3-pandas`

**LXC:**
- Packages: `python3-pyarrow python3-pandas`
- Port: `9090`

---

## Command Groups & Organization

The CLI organizes commands into 5 main groups for better navigation:

### 1. 🚀 DEPLOYMENT & INSTALLATION
Commands for container and service management:
```
install, start, stop, status, logs, restart
deploy-docker, deploy-podman, deploy-lxc
configure-service, health-check
```

### 2. 🔗 GATEWAY & CONNECTION MANAGEMENT
Commands for data gateway operations:
```
connect, disconnect, list-gateways, gateway-status
stream, stop-stream, test-gateway
```

### 3. 📊 DATA & MARKET OPERATIONS
Commands for market data handling:
```
price, ohlc, history, orderbook, depth
export, import, query, aggregate
```

### 4. 📈 ANALYTICS & ANALYSIS
Commands for analysis and research:
```
sentiment, correlation, indicators, backtest
portfolio, risk-analysis, alert
```

### 5. ⚙️ ADMINISTRATION & CONFIG
Commands for system administration:
```
config, settings, backup, restore, upgrade
security, performance, help, exit
```

---

## Tmux Window Layout

### Setup Multi-Window Environment

**Create tmux session with multiple windows:**
```bash
tmux new-session -s mdp -d

# Create windows for different groups
tmux new-window -t mdp -n deployment
tmux new-window -t mdp -n gateways
tmux new-window -t mdp -n data
tmux new-window -t mdp -n analytics
tmux new-window -t mdp -n admin
```

**From CLI:**
```
MDP> windows deployment
MDP> windows all
```

### Window Navigation

Within tmux:
- **Switch window**: `Ctrl+B <number>` or `Ctrl+B <window-name>`
- **Window 1** (Deployment): `Ctrl+B 1`
- **Window 2** (Gateways): `Ctrl+B 2`
- **Window 3** (Data): `Ctrl+B 3`
- **Window 4** (Analytics): `Ctrl+B 4`
- **Window 5** (Admin): `Ctrl+B 5`

### Suggested Window Usage

**Window 1 - Deployment:**
```bash
MDP> status
MDP> logs influxdb
MDP> health-check
```

**Window 2 - Gateways:**
```bash
MDP> connect oanda
MDP> stream oanda.eurusd
MDP> gateway-status
```

**Window 3 - Data:**
```bash
MDP> price EURUSD
MDP> ohlc EURUSD --timeframe 1h
MDP> export json /tmp/data.json
```

**Window 4 - Analytics:**
```bash
MDP> sentiment crypto
MDP> correlation EURUSD GBPUSD
MDP> backtest eurusd_strategy
```

**Window 5 - Admin:**
```bash
MDP> config show
MDP> backup
MDP> upgrade
```

---

## Best Practices

### 1. Container Runtime Selection

**Auto-detection (Recommended):**
```
MDP> status
# Shows detected runtime automatically
MDP> start all  # Uses detected runtime
```

**Explicit Selection:**
```
# Use Docker for maximum compatibility
MDP> deploy-docker all

# Use Podman for rootless operation
MDP> deploy-podman all

# Use LXC for system-level isolation
MDP> deploy-lxc all
```

### 2. Service Installation Order

**Recommended order:**
```
MDP> install redis        # Cache layer (fast)
MDP> install influxdb     # Time-series DB (medium)
MDP> install grafana      # Visualization (depends on InfluxDB)
MDP> install parquet      # Analytics (optional)
```

### 3. Health Monitoring

**Regular health checks:**
```bash
# Every minute
MDP> health-check

# Check specific service
MDP> health-check influxdb
MDP> logs influxdb --lines 20
```

### 4. Resource Management

**For Docker/Podman:**
```bash
# View resource usage
docker stats

# Limit container resources
docker run --memory 512m --cpus 1.0 ...
```

**For LXC:**
```bash
# View resource usage
lxc list

# Set resource limits
lxc config set <container> limits.memory 512MiB
```

### 5. Backup Strategy

**Before major changes:**
```
MDP> backup
MDP> logs all > /tmp/backup_logs.txt
```

---

## Troubleshooting

### Service Won't Start

1. **Check logs:**
   ```
   MDP> logs <service> --lines 100
   ```

2. **Verify container runtime:**
   ```
   MDP> status
   ```

3. **Check port availability:**
   ```bash
   netstat -tlnp | grep :<port>
   ```

### Health Check Fails

1. **Check service status:**
   ```
   MDP> status
   ```

2. **Review service logs:**
   ```
   MDP> logs <service>
   ```

3. **Verify connectivity:**
   ```bash
   curl http://localhost:8086/api/v2/health  # InfluxDB
   curl http://localhost:3000/api/health     # Grafana
   redis-cli ping                             # Redis
   ```

### Runtime Issues

1. **Docker not found:**
   ```bash
   apt-get install docker.io
   ```

2. **Podman not found:**
   ```bash
   apt-get install podman
   ```

3. **LXC not found:**
   ```bash
   apt-get install lxc lxd
   ```

---

## Advanced Usage

### Custom Service Configuration

**Edit service configuration:**
```
MDP> configure-service influxdb
# Review displayed config, then manually edit:
vi /root/rf_env/market_data_platform/config/influxdb.yaml
```

**Restart with new config:**
```
MDP> restart influxdb
```

### Multi-Runtime Deployment

**Mix runtimes for different services:**
```
MDP> deploy-docker influxdb grafana    # Critical services in Docker
MDP> deploy-podman redis               # Lightweight services in Podman
MDP> deploy-lxc parquet               # Isolated services in LXC
```

### Monitoring Setup

**Terminal 1 - Monitor status:**
```
watch -n 5 'MDP status'
```

**Terminal 2 - Watch logs:**
```
MDP logs influxdb --lines 50
```

**Terminal 3 - Run operations:**
```
MDP connect oanda
MDP stream oanda.eurusd
```

---

## Complete Installation Example

```bash
# 1. Launch CLI
cd /root/rf_env/market_data_platform
python cli/terminal.py

# 2. Check system
MDP> status

# 3. Install all services
MDP> install all

# 4. Start services
MDP> start all

# 5. Verify health
MDP> health-check

# 6. Check logs
MDP> logs influxdb
MDP> logs grafana
MDP> logs redis

# 7. View service status
MDP> status

# 8. Access services
# InfluxDB: http://localhost:8086
# Grafana: http://localhost:3000
# Redis: localhost:6379
```

---

## Environment Variables

Set before launching CLI:

```bash
# Force specific runtime
export MDP_CONTAINER_RUNTIME=docker

# Set service ports
export INFLUXDB_PORT=8086
export GRAFANA_PORT=3000
export REDIS_PORT=6379

# Enable debug logging
export MDP_DEBUG=1
```

---

## Integration with Deployment Files

The enhanced CLI reads from:
- `config/gateways.yaml` - Gateway configurations
- `config/influxdb.yaml` - InfluxDB settings
- `config/zmq_topics.yaml` - ZMQ topic definitions
- `config/research_config.yaml` - Research settings

Configure these files for custom deployments.

---

## Summary

| Feature | Status |
|---------|--------|
| **Docker support** | ✅ Complete |
| **Podman support** | ✅ Complete |
| **LXC support** | ✅ Complete |
| **Auto-detection** | ✅ Complete |
| **Service management** | ✅ Complete (install, start, stop, logs, health) |
| **Configuration** | ✅ Complete |
| **Tmux integration** | ✅ Complete |
| **Command grouping** | ✅ Complete |
| **Best practices** | ✅ Complete |

---

**Version**: 2.0.0 (Enhanced)  
**Last Updated**: 2024-01-16  
**Status**: Production Ready ✅
