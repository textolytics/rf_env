# CLI Visual Command Reference

## Command Quick Access Map

```
┌─────────────────────────────────────────────────────────────────┐
│ Market Data Platform CLI - Command Reference Map                │
└─────────────────────────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════════════╗
║ 🚀 DEPLOYMENT & INSTALLATION (Group 1)                         ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  install         Install services                              ║
║  ├─ install all              Install all services              ║
║  ├─ install influxdb         Install InfluxDB                 ║
║  ├─ install grafana          Install Grafana                  ║
║  ├─ install redis            Install Redis                    ║
║  ├─ install parquet          Install Parquet                  ║
║  └─ --runtime docker|podman|lxc                               ║
║                                                                 ║
║  start           Start services                                ║
║  ├─ start all                Start all services               ║
║  ├─ start <service>          Start specific service           ║
║  └─ Check running via status                                   ║
║                                                                 ║
║  stop            Stop running services                         ║
║  ├─ stop <service>           Stop specific service            ║
║  └─ stop all                 Stop all services                ║
║                                                                 ║
║  restart         Restart services (stop → wait 2s → start)   ║
║  ├─ restart <service>        Restart specific service         ║
║  └─ restart all              Restart all services             ║
║                                                                 ║
║  status          Show deployment status                        ║
║  ├─ Display: current runtime                                   ║
║  ├─ Display: running services                                  ║
║  └─ Display: available services                                ║
║                                                                 ║
║  logs            View service logs                             ║
║  ├─ logs <service>           Show latest logs                 ║
║  ├─ logs <service> --lines N Show N lines of logs              ║
║  └─ Example: logs influxdb --lines 50                         ║
║                                                                 ║
║  health-check    Check service health                          ║
║  ├─ health-check             Check all services               ║
║  ├─ health-check <service>   Check specific service           ║
║  └─ Returns: health status                                     ║
║                                                                 ║
║  deploy-docker   Deploy to Docker runtime                      ║
║  ├─ deploy-docker <service>  Deploy one service               ║
║  └─ deploy-docker all        Deploy all services              ║
║                                                                 ║
║  deploy-podman   Deploy to Podman runtime                      ║
║  ├─ deploy-podman <service>  Deploy one service               ║
║  └─ deploy-podman all        Deploy all services              ║
║                                                                 ║
║  deploy-lxc      Deploy to LXC runtime                         ║
║  ├─ deploy-lxc <service>     Deploy one service               ║
║  └─ deploy-lxc all           Deploy all services              ║
║                                                                 ║
║  configure-service  Show service configuration                 ║
║  ├─ configure-service influxdb                                ║
║  ├─ configure-service grafana                                 ║
║  └─ Shows config template for current runtime                 ║
║                                                                 ║
║  Ctrl+B 1        (Tmux) Switch to Deployment window           ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
```

---

```
╔═════════════════════════════════════════════════════════════════╗
║ 🔗 GATEWAY & CONNECTION MANAGEMENT (Group 2)                  ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  connect         Connect to gateway                            ║
║  ├─ connect freedx            Connect to Freedx               ║
║  ├─ connect gate.io           Connect to Gate.io              ║
║  ├─ connect oanda             Connect to OANDA               ║
║  ├─ connect kraken            Connect to Kraken               ║
║  └─ connect betfair           Connect to Betfair              ║
║                                                                 ║
║  disconnect      Disconnect from gateway                       ║
║  ├─ disconnect <gateway>      Disconnect specific             ║
║  └─ disconnect all            Disconnect all                  ║
║                                                                 ║
║  list-gateways   List available gateways                       ║
║  └─ Shows: gateway name, type, status, URL                    ║
║                                                                 ║
║  gateway-status  Check gateway connection status               ║
║  ├─ gateway-status <gateway>  Check specific                  ║
║  └─ gateway-status all        Check all                       ║
║                                                                 ║
║  stream          Stream data from gateway                      ║
║  ├─ stream <gateway>.<symbol> Stream specific symbol           ║
║  ├─ stream oanda.eurusd                                        ║
║  └─ stream kraken.btcusd                                       ║
║                                                                 ║
║  stop-stream     Stop streaming data                           ║
║  ├─ stop-stream <gateway>                                      ║
║  └─ stop-stream all           Stop all streams                ║
║                                                                 ║
║  test-gateway    Test gateway connectivity                     ║
║  ├─ test-gateway <gateway>    Test specific gateway           ║
║  └─ test-gateway all          Test all gateways              ║
║                                                                 ║
║  Ctrl+B 2        (Tmux) Switch to Gateways window             ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
```

---

```
╔═════════════════════════════════════════════════════════════════╗
║ 📊 DATA & MARKET OPERATIONS (Group 3)                         ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  price           Get current price                             ║
║  ├─ price <symbol>           Get current price                ║
║  └─ price eurusd             Get EURUSD price                 ║
║                                                                 ║
║  ohlc            Get OHLC data                                 ║
║  ├─ ohlc <symbol>            Get default 1h OHLC             ║
║  ├─ ohlc <symbol> --timeframe 1h                              ║
║  └─ --timeframe 1m|5m|1h|4h|1d                                ║
║                                                                 ║
║  history         Get historical data                           ║
║  ├─ history <symbol>         Get last 100 candles             ║
║  ├─ history <symbol> --limit N                                ║
║  └─ history eurusd --limit 1000                               ║
║                                                                 ║
║  orderbook       Get order book                                ║
║  ├─ orderbook <symbol>       Show order book                  ║
║  └─ orderbook eurusd         EURUSD order book                ║
║                                                                 ║
║  depth           Get depth chart                               ║
║  ├─ depth <symbol>           Show depth chart                 ║
║  └─ depth btcusd             Bitcoin depth chart              ║
║                                                                 ║
║  export          Export data to file                           ║
║  ├─ export json <file>       Export as JSON                   ║
║  ├─ export csv <file>        Export as CSV                    ║
║  └─ export parquet <file>    Export as Parquet               ║
║                                                                 ║
║  import          Import data from file                         ║
║  ├─ import json <file>       Import from JSON                 ║
║  ├─ import csv <file>        Import from CSV                  ║
║  └─ import parquet <file>    Import from Parquet             ║
║                                                                 ║
║  query           Query database                                ║
║  ├─ query <sql>              Execute SQL query                ║
║  └─ Example: query SELECT * FROM prices                       ║
║                                                                 ║
║  aggregate       Aggregate data                                ║
║  ├─ aggregate <symbol> --period 1d                            ║
║  └─ aggregate <symbol> --function avg|min|max                 ║
║                                                                 ║
║  Ctrl+B 3        (Tmux) Switch to Data window                 ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
```

---

```
╔═════════════════════════════════════════════════════════════════╗
║ 📈 ANALYTICS & ANALYSIS (Group 4)                             ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  sentiment       Analyze sentiment                             ║
║  ├─ sentiment <asset>        Analyze sentiment               ║
║  ├─ sentiment crypto         Crypto sentiment                 ║
║  └─ sentiment equity         Equity sentiment                 ║
║                                                                 ║
║  correlation     Calculate correlation                         ║
║  ├─ correlation <sym1> <sym2>                                 ║
║  ├─ correlation eurusd gbpusd                                 ║
║  └─ correlation btcusd ethusd                                 ║
║                                                                 ║
║  indicators      Calculate technical indicators                ║
║  ├─ indicators <symbol>      Calculate indicators             ║
║  ├─ indicators eurusd                                          ║
║  └─ Returns: RSI, MACD, Bollinger, etc.                       ║
║                                                                 ║
║  backtest        Backtest strategy                             ║
║  ├─ backtest <strategy>      Run backtest                     ║
║  ├─ backtest eurusd_strategy                                   ║
║  └─ Returns: P&L, Sharpe, drawdown                            ║
║                                                                 ║
║  portfolio       Analyze portfolio                             ║
║  ├─ portfolio <name>         Analyze portfolio               ║
║  ├─ portfolio my_portfolio                                     ║
║  └─ Returns: allocation, performance                           ║
║                                                                 ║
║  risk-analysis   Perform risk analysis                         ║
║  ├─ risk-analysis <portfolio>                                 ║
║  ├─ risk-analysis my_portfolio                                ║
║  └─ Returns: VaR, CVaR, beta                                  ║
║                                                                 ║
║  alert           Set/manage alerts                             ║
║  ├─ alert set <symbol> <condition> <value>                    ║
║  ├─ alert set eurusd >= 1.1000                                ║
║  ├─ alert list                Show all alerts                 ║
║  └─ alert delete <id>        Delete alert                     ║
║                                                                 ║
║  Ctrl+B 4        (Tmux) Switch to Analytics window            ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
```

---

```
╔═════════════════════════════════════════════════════════════════╗
║ ⚙️  ADMINISTRATION & CONFIG (Group 5)                          ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  config          Show/edit configuration                       ║
║  ├─ config show              Show all config                  ║
║  ├─ config show <section>    Show section                     ║
║  ├─ config set <key> <val>   Set config value                 ║
║  └─ config reset             Reset to defaults                ║
║                                                                 ║
║  settings        Show/edit settings                            ║
║  ├─ settings show            Show all settings                ║
║  ├─ settings update <key> <val>                               ║
║  └─ settings reset           Reset settings                   ║
║                                                                 ║
║  backup          Create backup                                 ║
║  ├─ backup                   Full backup                      ║
║  ├─ backup --database        Database only                    ║
║  ├─ backup --config          Config only                      ║
║  └─ backup --destination <path>                               ║
║                                                                 ║
║  restore         Restore from backup                           ║
║  ├─ restore <backup_file>    Restore backup                  ║
║  ├─ restore --database <file>                                 ║
║  └─ restore --config <file>                                   ║
║                                                                 ║
║  upgrade         Upgrade system                                ║
║  ├─ upgrade                  Upgrade all                      ║
║  ├─ upgrade <component>      Upgrade component                ║
║  └─ upgrade --check          Check for updates                ║
║                                                                 ║
║  security        Security operations                           ║
║  ├─ security status          Show security status             ║
║  ├─ security audit           Run security audit               ║
║  ├─ security certificate     Manage certificates              ║
║  └─ security firewall        Configure firewall               ║
║                                                                 ║
║  performance     Performance operations                        ║
║  ├─ performance status       Show performance metrics         ║
║  ├─ performance optimize     Optimize system                  ║
║  ├─ performance profile      Run profiling                    ║
║  └─ performance report       Generate report                  ║
║                                                                 ║
║  help            Show help                                     ║
║  ├─ help                     Show all commands                ║
║  ├─ help <command>           Show command help                ║
║  └─ help install             Help for install                 ║
║                                                                 ║
║  exit            Exit CLI                                      ║
║  ├─ exit                     Exit interactive mode             ║
║  ├─ quit                     Alias for exit                   ║
║  └─ Ctrl+D                   Exit (keyboard shortcut)         ║
║                                                                 ║
║  Ctrl+B 5        (Tmux) Switch to Admin window                ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
```

---

## Common Command Patterns

### Service Management Pattern
```
MDP> [action] [service] [options]

Actions:     install, start, stop, restart, logs, status
Services:    influxdb, grafana, redis, parquet, all
Options:     --runtime docker|podman|lxc, --lines N

Examples:
  MDP> install all                        # Install everything
  MDP> start grafana                      # Start Grafana
  MDP> logs influxdb --lines 50          # Show 50 lines
  MDP> deploy-docker all                 # Deploy to Docker
```

### Data Operations Pattern
```
MDP> [operation] [symbol] [options]

Operations:  price, ohlc, history, export, import
Symbols:     eurusd, gbpusd, btcusd, ethusd, etc.
Options:     --timeframe 1m|5m|1h|4h|1d, --limit N, --format json|csv

Examples:
  MDP> price eurusd                       # Get EURUSD price
  MDP> ohlc eurusd --timeframe 1h        # Get 1h OHLC
  MDP> export json /tmp/data.json        # Export to JSON
```

### Gateway Pattern
```
MDP> [gateway-action] [gateway|symbol]

Actions:     connect, disconnect, stream, test-gateway
Gateways:    freedx, gate.io, oanda, kraken, betfair
Symbols:     gateway.symbol format (e.g., oanda.eurusd)

Examples:
  MDP> connect oanda                      # Connect to OANDA
  MDP> stream oanda.eurusd               # Stream EURUSD
  MDP> test-gateway all                  # Test all gateways
```

### Configuration Pattern
```
MDP> config [action] [section] [key] [value]

Actions:     show, set, reset
Sections:    database, gateways, services, analytics, etc.

Examples:
  MDP> config show                        # Show all config
  MDP> config set database host localhost # Set value
  MDP> config reset                       # Reset to defaults
```

---

## Command Discovery

### Get Help
```bash
MDP> help                      # Show all commands by group
MDP> help install              # Show help for install
MDP> help <any-command>        # Show help for any command
```

### Show Status
```bash
MDP> status                    # Show current status
MDP> health-check              # Check health of services
```

### List Resources
```bash
MDP> list-gateways             # Show all gateways
MDP> config show               # Show all configuration
```

---

## Keyboard Shortcuts

### Tmux Navigation
```
Ctrl+B 1          Switch to Deployment window
Ctrl+B 2          Switch to Gateways window
Ctrl+B 3          Switch to Data window
Ctrl+B 4          Switch to Analytics window
Ctrl+B 5          Switch to Admin window

Ctrl+B N          Next window
Ctrl+B P          Previous window
Ctrl+B W          List windows
Ctrl+B D          Detach from session
```

### CLI Shortcuts
```
Up Arrow          Previous command
Down Arrow        Next command
Ctrl+A            Start of line
Ctrl+E            End of line
Ctrl+D            Exit CLI
Tab               Command completion
?                 Show help
```

---

## Service Configuration Matrix

```
┌──────────┬──────────────┬──────────────┬──────────────┐
│ Service  │ Docker       │ Podman       │ LXC          │
├──────────┼──────────────┼──────────────┼──────────────┤
│InfluxDB  │:8086→8086    │:8086→8086    │:8086 via nat │
│          │image:2.7     │image:2.7     │pkg:influxdb2 │
├──────────┼──────────────┼──────────────┼──────────────┤
│Grafana   │:3000→3000    │:3000→3000    │:3000 via nat │
│          │image:latest  │image:latest  │pkg:grafana   │
├──────────┼──────────────┼──────────────┼──────────────┤
│Redis     │:6379→6379    │:6379→6379    │:6379 via nat │
│          │image:7-alpine│image:7-alpine│pkg:redis-srv │
├──────────┼──────────────┼──────────────┼──────────────┤
│Parquet   │:9090→9090    │:9090→9090    │:9090 via nat │
│          │ubuntu:22.04  │ubuntu:22.04  │python-pyarrow│
└──────────┴──────────────┴──────────────┴──────────────┘
```

---

## Success Indicators

### Service Started Successfully
```
✓ Service installed
✓ Service started
✓ Health check passed
→ Service ready for use
```

### Full System Ready
```
✓ Docker detected and running
✓ InfluxDB started
✓ Grafana started
✓ Redis started
✓ All health checks passed
→ System ready for data operations
```

---

## Troubleshooting Quick Reference

```
Problem                     Solution
─────────────────────────────────────────────────────
Service won't start        → logs <service>
Port already in use        → Check status, restart
Docker not found           → apt-get install docker.io
Permission denied          → Use podman or sudo
Health check fails         → logs <service> --lines 100
Configuration issue        → config show
Low disk space             → backup cleanup, compress

Quick Checks:
  MDP> status              ← Check runtime & services
  MDP> health-check        ← Verify all services
  MDP> logs <svc>         ← View service logs
  MDP> deploy-podman all  ← Switch to alternate runtime
```

---

## Pro Tips

1. **Auto-completion**: Press `Tab` for command completion
2. **Command history**: Use up/down arrows for previous commands
3. **Parallel operations**: Open multiple Tmux windows for concurrent work
4. **Health monitoring**: Set up `watch` for continuous monitoring:
   ```bash
   watch -n 5 'MDP> status'
   ```
5. **Log streaming**: Keep logs open in one window:
   ```bash
   MDP> logs influxdb  # Runs continuously
   ```
6. **Batch operations**: Use scripts to automate workflows
   ```bash
   echo -e "install all\nstart all\nhealth-check" | MDP
   ```

---

**Reference Version**: 2.0.0  
**Last Updated**: January 16, 2024  
**Status**: ✅ Ready for Production
