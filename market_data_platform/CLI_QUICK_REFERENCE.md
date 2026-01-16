# CLI Enhancement Quick Reference Card

## 🚀 Quick Commands

### Container Runtime & Services Status
```bash
status              # Show detected runtime and running services
health-check        # Check health of all or specific services
health-check <svc>  # Check specific service: influxdb, grafana, redis, parquet
```

### Installation & Deployment
```bash
install all                    # Install all services (auto-detects runtime)
install <service>              # Install specific service
install all --runtime docker   # Force Docker deployment
install all --runtime podman   # Force Podman deployment
install all --runtime lxc      # Force LXC deployment

start <service>               # Start service (docker run / podman run / systemctl)
start all                     # Start all services
stop <service>                # Stop service
stop all                      # Stop all services
restart <service>             # Restart with 2-second delay
restart all                   # Restart all services
```

### Runtime Switching
```bash
deploy-docker <service>       # Deploy service to Docker
deploy-docker all             # Deploy all to Docker
deploy-podman <service>       # Deploy service to Podman
deploy-lxc <service>          # Deploy service to LXC
```

### Service Monitoring
```bash
logs <service>                # Show latest service logs
logs <service> --lines 100    # Show 100 lines of logs
logs influxdb                 # Example: InfluxDB logs
logs grafana                  # Example: Grafana logs
```

### Service Configuration
```bash
configure-service influxdb    # Show InfluxDB config template
configure-service grafana     # Show Grafana config template
```

---

## 📊 Service Details

| Service   | Docker Image         | Port  | Type            |
|-----------|----------------------|-------|-----------------|
| InfluxDB  | influxdb:2.7-alpine  | 8086  | Time-Series DB  |
| Grafana   | grafana/grafana      | 3000  | Visualization   |
| Redis     | redis:7-alpine       | 6379  | Cache/Message   |
| Parquet   | ubuntu:22.04         | 9090  | Analytics       |
| ZMQ       | Native               | 5555  | Messaging       |

---

## 🎯 Container Runtime Capabilities

### Docker
- **Best for**: Production, widest compatibility
- **Detection**: `docker --version`
- **Port mapping**: Full support
- **Volume management**: Complete
- **Resource limits**: Advanced options

### Podman
- **Best for**: Rootless operation, security
- **Detection**: `podman --version`
- **Port mapping**: Full support
- **Volume management**: Compatible
- **Resource limits**: Supported

### LXC
- **Best for**: System isolation, performance
- **Detection**: `lxc --version`
- **Port mapping**: Via nat
- **Volume management**: Via mount
- **Resource limits**: cgroup native

---

## 🪟 Tmux Window Groups

```bash
windows deployment            # Plan deployment window layout
windows gateways              # Plan gateways window layout
windows all                   # Show all 5 window groups
```

### Default Layout
```
┌─ Window 1: DEPLOYMENT ───┬─ Window 2: GATEWAYS ──┐
│ install, start, stop      │ connect, stream       │
│ logs, health-check, status│ gateway-status        │
├─ Window 3: DATA ──────────┼─ Window 4: ANALYTICS ─┤
│ price, ohlc, history      │ sentiment, indicators │
│ export, import, query     │ correlation, backtest │
└─ Window 5: ADMIN ─────────────────────────────────┘
  config, backup, upgrade, security, help
```

---

## 📋 Command Groups

```
🚀 DEPLOYMENT       → install, start, stop, restart, status, logs, health-check
🔗 GATEWAYS        → connect, disconnect, list-gateways, stream, test-gateway
📊 DATA            → price, ohlc, history, orderbook, export, import, query
📈 ANALYTICS       → sentiment, correlation, indicators, backtest, alert
⚙️  ADMIN          → config, backup, restore, upgrade, security, help
```

---

## 💡 Common Workflows

### Fresh Installation (Recommended)
```bash
MDP> status                   # Check runtime
MDP> install all              # Install all services
MDP> start all                # Start all services
MDP> health-check             # Verify all healthy
MDP> logs influxdb            # Verify InfluxDB started
```

### Switch Runtimes
```bash
MDP> deploy-podman all        # Switch to Podman
MDP> health-check             # Verify all services
```

### Troubleshooting Service
```bash
MDP> status                   # Check overall status
MDP> logs <service>           # View service logs
MDP> health-check <service>   # Check specific service
MDP> restart <service>        # Restart service
```

### Development Setup
```bash
MDP> deploy-docker influxdb   # Docker for core DB
MDP> deploy-podman redis      # Podman for cache
MDP> health-check             # Verify setup
MDP> status                   # Show active services
```

---

## 🔍 Help & Discovery

```bash
help                          # Show all commands grouped by category
help install                  # Show help for 'install' command
help <command>                # Show detailed help for any command
status                        # Show current deployment status
windows                       # Show tmux window layout planning
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Service won't start** | `logs <service>` to see error, then `restart <service>` |
| **Docker not found** | `apt-get install docker.io` or `deploy-podman <svc>` |
| **Port already in use** | Check existing containers or services on that port |
| **Permission denied** | May need sudo, or use `deploy-podman` for rootless |
| **Health check fails** | Review service logs: `logs <service>` |

---

## 📌 Environment Detection

The CLI automatically detects available runtimes in this order:
1. **Docker** (`docker` command)
2. **Podman** (`podman` command)
3. **LXC** (`lxc` command)
4. **Auto** (manual specification if none found)

Display detected runtime:
```bash
MDP> status
```

Force specific runtime:
```bash
MDP> deploy-docker all
MDP> deploy-podman all
MDP> deploy-lxc all
```

---

## 📝 File Locations

| File | Purpose |
|------|---------|
| `/root/rf_env/market_data_platform/cli/terminal.py` | Main CLI application |
| `/root/rf_env/market_data_platform/config/gateways.yaml` | Gateway configurations |
| `/root/rf_env/market_data_platform/config/influxdb.yaml` | InfluxDB settings |
| `results/` | Test and execution logs |
| `docker/` | Docker compose files |

---

## 🎓 Learning Path

1. **Start**: `status` → See detected runtime
2. **Install**: `install all` → Deploy all services
3. **Monitor**: `health-check` → Verify all healthy
4. **Explore**: `help <service>` → Discover available operations
5. **Operate**: Use service-specific commands
6. **Manage**: Use admin commands for backup/restore

---

## 🚀 Performance Tips

- **Use Docker** for maximum speed (native container runtime)
- **Use Podman** for security-first deployments (rootless)
- **Use LXC** for system-level isolation (VM-like security)
- **Health checks** before production operations
- **Monitor logs** during first run of new services

---

**CLI Version**: 2.0.0 Enhanced  
**Status**: ✅ Production Ready  
**Last Updated**: 2024-01-16
