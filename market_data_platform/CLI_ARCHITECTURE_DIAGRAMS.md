# CLI Architecture & Workflow Diagrams

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Market Data Platform CLI                     │
│                      (terminal.py v2.0)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
            ┌───────▼────────┐  ┌───────▼────────┐
            │ Runtime Layer  │  │ Service Layer  │
            ├────────────────┤  ├────────────────┤
            │ • Docker       │  │ • InfluxDB     │
            │ • Podman       │  │ • Grafana      │
            │ • LXC          │  │ • Redis        │
            │ • Auto-detect  │  │ • Parquet      │
            └────────────────┘  └────────────────┘
                    │                   │
        ┌───────────┴───────────┬──────┴────────────┐
        │                       │                   │
   ┌────▼────┐        ┌────────▼───────┐    ┌─────▼──────┐
   │ Commands │        │ Configuration  │    │   State    │
   ├─────────┤        ├────────────────┤    ├────────────┤
   │ install │        │ SERVICE_CONFIG │    │ running_   │
   │ start   │        │ s (per runtime)│    │ services   │
   │ stop    │        │ COMMAND_GROUPS │    │ tmux_      │
   │ restart │        │                │    │ session    │
   │ logs    │        │ PORT mapping   │    │ container_ │
   │ status  │        │ ENV variables  │    │ runtime    │
   └─────────┘        │ VOLUME binds   │    └────────────┘
                      └────────────────┘
```

---

## Command Execution Flow

```
User Input
    │
    ▼
┌──────────────────┐
│ Parse Command    │
│ (cmd.Cmd)        │
└────────┬─────────┘
         │
         ▼
    ┌────────────────────────────────┐
    │ Extract Runtime & Service      │
    │ (default or specified)         │
    └────────┬───────────────────────┘
             │
             ▼
    ┌────────────────────────────────┐
    │ Get Service Configuration      │
    │ SERVICE_CONFIGS[service]       │
    │ [runtime][option]              │
    └────────┬───────────────────────┘
             │
             ▼
    ┌────────────────────────────────┐
    │ Build Command String           │
    │ docker run / podman run / etc  │
    │ with env, ports, volumes       │
    └────────┬───────────────────────┘
             │
             ▼
    ┌────────────────────────────────┐
    │ Execute via subprocess.run()   │
    │ or subprocess.Popen()          │
    └────────┬───────────────────────┘
             │
             ▼
    ┌────────────────────────────────┐
    │ Capture Output/Result          │
    │ stdout, stderr, returncode     │
    └────────┬───────────────────────┘
             │
             ▼
    ┌────────────────────────────────┐
    │ Update Internal State          │
    │ running_services[], variables  │
    └────────┬───────────────────────┘
             │
             ▼
    Print Result to User
```

---

## Container Runtime Selection

```
                    detect_container_runtime()
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                    ▼         ▼         ▼
              shutil.which()  ...  ...
              /usr/bin/docker /usr/bin/podman /usr/bin/lxc
                    │         │         │
                    ▼         ▼         ▼
              [DOCKER]   [PODMAN]    [LXC]
                    │         │         │
                    └─────────┼─────────┘
                              │
                   ┌──────────▼──────────┐
                   │ Selected Runtime    │
                   │ (returned to caller)│
                   └─────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌──────────────┐    ┌──────────────┐
            │ Use docker   │    │ Use podman   │
            │ (if found)   │    │ (if docker   │
            │              │    │  not found)  │
            └──────────────┘    └──────────────┘
```

---

## Service Installation Workflow

```
    user: install influxdb --runtime docker
                    │
                    ▼
    ┌──────────────────────────┐
    │ do_install(arg)          │
    │ Parse: service, runtime  │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Get SERVICE_CONFIGS      │
    │ ["influxdb"]["docker"]   │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Build docker run cmd     │
    │ - image                  │
    │ - port mapping           │
    │ - env variables          │
    │ - volume mounts          │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Execute subprocess       │
    │ $ docker run ...         │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ Update running_services  │
    │ ['influxdb'] = running   │
    └────────┬─────────────────┘
             │
             ▼
    ✅ "InfluxDB installed successfully"
```

---

## Multi-Runtime Deployment Scenario

```
Target: Deploy different services to different runtimes

┌─────────────────────────────────────────────────────┐
│ MDP> deploy-docker influxdb grafana                │
│ MDP> deploy-podman redis                           │
│ MDP> deploy-lxc parquet                            │
└─────────────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    DOCKER      PODMAN         LXC
        │           │           │
    ┌───┴──┐     ┌───┴─────┐  ┌─┴───┐
    │      │     │         │  │     │
    ▼      ▼     ▼         ▼  ▼     ▼
InfluxDB Grafana Redis    (none)  Parquet
    │      │     │              │
    └──────┴─────┴──────────────┘
           │
    ┌──────▼──────────────┐
    │ Unified CLI View    │
    │ "all services OK"   │
    │ Different runtimes  │
    │ but managed as one  │
    └─────────────────────┘
```

---

## Tmux Window Organization

```
╔═══════════════════════════════════════════════════════════════════╗
║  Market Data Platform - Tmux Session (mdp)                        ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  [1] Deployment  [2] Gateways  [3] Data  [4] Analytics [5] Admin ║
║  ┌──────────┐   ┌──────────┐  ┌─────┐   ┌──────────┐  ┌────────┐ ║
║  │ install  │   │ connect  │  │price│   │sentiment │  │config  │ ║
║  │ start    │   │ stream   │  │ohlc │   │backtest  │  │backup  │ ║
║  │ stop     │   │ gateway- │  │hist │   │correlate │  │upgrade │ ║
║  │ logs     │   │ status   │  │ory  │   │indicators│  │restore │ ║
║  │ status   │   │ test-gw  │  │orderbook│ portfolio  │ security│ ║
║  │ restart  │   │          │  │export│  │risk      │  │help    │ ║
║  │ health-  │   │          │  │import│  │alert     │  │        │ ║
║  │ check    │   │          │  │     │   │          │  │        │ ║
║  └──────────┘   └──────────┘  └─────┘   └──────────┘  └────────┘ ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    Ctrl+B 1         Ctrl+B 2      Ctrl+B 3   Ctrl+B 4      Ctrl+B 5
    (window 1)       (window 2)    (window 3) (window 4)    (window 5)
```

---

## Service Configuration Hierarchy

```
                        SERVICE_CONFIGS
                              │
                ┌─────────────┼─────────────┐
                │             │             │
            ┌───▼──┐      ┌───▼──┐      ┌──▼───┐
            │  influxdb   │ grafana    │ redis
            ├──────┤      ├─────────┤  ├───────┤
            │      │      │         │  │       │
        ┌───▼─┐┌──▼──┐┌───▼┐      │   │       │
        │docker││podman││lxc │      │   │       │
        ├──────┤├─────┤├────┤      │   │       │
        │Image ││Image││Pkg │      │   │       │
        │Port  ││Port ││Port│      │   │       │
        │Env   ││Env  ││Env │      │   │       │
        │Vol   ││Vol  ││Vol │      │   │       │
        └──────┘└─────┘└────┘      │   │       │
                                   │   │       │
        ┌──────────────────────────┘   │       │
        │                              │       │
    [Same pattern for]            [Same pattern]
    Grafana & Parquet               for Redis
```

---

## Command Group Organization

```
                    COMMAND_GROUPS
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    ┌───▼────┐        ┌───▼────┐      ┌────▼─────┐
    │deployment    │ gateways │      │    data
    ├─────────┤    ├─────────┤      ├──────────┤
    │• install│    │• connect │      │• price   │
    │• start  │    │• disconnect   │• ohlc    │
    │• stop   │    │• list-gateways   │• history│
    │• restart│    │• gateway-status  │• export │
    │• logs   │    │• stream   │      │• import │
    │• status │    │• test-gateway    │• query  │
    └─────────┘    └─────────┘      └──────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    ┌───▼─────────┐   ┌───▼──────┐    ┌────▼──┐
    │  analytics  │   │   admin
    ├─────────────┤   ├──────────┤
    │• sentiment  │   │• config  │
    │• correlation    │• backup  │
    │• indicators │   │• restore │
    │• backtest   │   │• upgrade │
    │• portfolio  │   │• security│
    │• risk       │   │• help    │
    │• alert      │   │• exit    │
    └─────────────┘   └──────────┘
```

---

## Service Lifecycle State Machine

```
    Not Installed
        │
        │ install
        ▼
    ┌──────────────┐
    │  Installed   │◄─────────────┐
    │ (ready state)│              │
    └────┬─────────┘              │
         │                        │
         │ start                  │
         ▼                        │
    ┌──────────────┐              │
    │   Running    │──► stop ─────┘
    │ (active)     │
    └────┬─────────┘
         │
         │ logs, health-check
         ▼
    ┌──────────────┐
    │  Monitoring  │
    │ (status view)│
    └──────────────┘
         │
         │ restart (stop → wait 2s → start)
         └─────────────────────┬──────────────┐
                               │              │
                         ┌─────▼─────┐   ┌───▼──────┐
                         │ Restarting│   │  Stopped │
                         │  (running)│   │(offline) │
                         └───────────┘   └──────────┘
```

---

## CLI Initialization Sequence

```
1. Detect Container Runtime
   └─→ docker? podman? lxc? → AUTO
       │
       ▼
2. Initialize Service State
   └─→ running_services = {}
       │
       ▼
3. Load Service Configurations
   └─→ SERVICE_CONFIGS loaded from enums
       │
       ▼
4. Load Command Groups
   └─→ COMMAND_GROUPS with all 50+ commands
       │
       ▼
5. Initialize CLI Prompt
   └─→ Show intro, set prompt
       │
       ▼
6. Start Interactive Loop
   └─→ cmdloop() ready for user input
```

---

## Configuration Override Priority

```
    User Input (highest priority)
         │
         ▼
    --runtime flag
         │
         ▼
    Environment Variable
    (MDP_CONTAINER_RUNTIME)
         │
         ▼
    Auto-Detected Runtime
         │
         ▼
    Default (AUTO) (lowest priority)
```

---

## Error Handling Flow

```
    Execute Command
         │
         ├─ Success ──→ Update state, print success
         │
         └─ Failure
             │
             ├─ Docker not found?
             │  └─→ Try Podman
             │      └─ Not found? Try LXC
             │
             ├─ Service not found?
             │  └─→ "Service not available"
             │
             ├─ Port already in use?
             │  └─→ "Check 'status' for conflicts"
             │
             └─ Command execution error?
                 └─→ Print stderr, offer troubleshooting
```

---

## Multi-Window Workflow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     Tmux Session: mdp                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Window 1: Deployment              Window 2: Gateways          │
│  ┌────────────────────────┐       ┌─────────────────────────┐  │
│  │ install influxdb       │       │ connect oanda           │  │
│  │ start all              │       │ stream oanda.eurusd     │  │
│  │ health-check           │ Ctrl+B│ gateway-status          │  │
│  │ status                 │   2   │ test-gateway oanda      │  │
│  └────────────────────────┘───────┴─────────────────────────┘  │
│                                                                  │
│  Window 3: Data                    Window 4: Analytics         │
│  ┌────────────────────────┐       ┌─────────────────────────┐  │
│  │ price EURUSD           │       │ sentiment crypto        │  │
│  │ ohlc EURUSD 1h         │ Ctrl+B│ correlation EURUSD EUR │  │
│  │ history EURUSD         │   4   │ indicators EURUSD       │  │
│  │ export json prices.json│       │ backtest strategy1      │  │
│  └────────────────────────┘───────┴─────────────────────────┘  │
│                                                                  │
│                  Window 5: Admin                                │
│                  ┌─────────────────────┐                       │
│                  │ config show         │                       │
│                  │ backup --full       │                       │
│                  │ logs all            │                       │
│                  │ security status     │                       │
│                  └─────────────────────┘                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                        Ctrl+B <number>
                     to switch windows
```

---

## Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    Market Data Platform                         │
└─────────────────────────────────────────────────────────────────┘
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐
    │  Container      │ │  Gateway     │ │  Data Storage   │
    │  Runtime        │ │  Managers    │ │  & Analytics    │
    ├─────────────────┤ ├──────────────┤ ├─────────────────┤
    │• Docker         │ │• Freedx      │ │• InfluxDB       │
    │• Podman         │ │• Gate.io     │ │• Grafana        │
    │• LXC            │ │• OANDA       │ │• Redis          │
    │                 │ │• Kraken      │ │• Parquet        │
    │                 │ │• Betfair     │ │• ZMQ Broker     │
    └─────────────────┘ └──────────────┘ └─────────────────┘
              │              │              │
              └──────────────┬───────────────┘
                             │
                    ┌────────▼─────────┐
                    │  CLI Terminal    │
                    │  (Unified View)  │
                    └──────────────────┘
```

---

**Diagram Set**: Complete CLI Architecture  
**Version**: 2.0 Enhanced  
**Status**: ✅ Ready for Reference
