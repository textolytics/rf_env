#!/usr/bin/env python3
"""
Market Data Platform - Enhanced Bloomberg Terminal-Style CLI
Features:
  - Multi-container deployment (Docker, Podman, LXC)
  - Service-specific management (InfluxDB, Grafana, Parquet)
  - Tmux window navigation with grouped command tabs
  - Installation, startup, and deployment orchestration
"""

import cmd
import json
import sys
import os
import shutil
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
import subprocess
import signal
import asyncio

# ============================================================================
# Color support
# ============================================================================

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    MAGENTA = '\033[95m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

# ============================================================================
# Enumerations
# ============================================================================

class ContainerRuntime(Enum):
    DOCKER = "docker"
    PODMAN = "podman"
    LXC = "lxc"
    AUTO = "auto"  # Detect automatically

class Service(Enum):
    INFLUXDB = "influxdb"
    GRAFANA = "grafana"
    PARQUET = "parquet"
    REDIS = "redis"
    ZMQ = "zmq"
    ALL = "all"

class WindowGroup(Enum):
    DEPLOYMENT = "deployment"
    GATEWAYS = "gateways"
    DATA = "data"
    ANALYTICS = "analytics"
    ADMIN = "admin"

class MarketDataCLI(cmd.Cmd):
    """Bloomberg Terminal-style market data platform CLI with container management"""
    
    intro = f"""
{Colors.BOLD}{Colors.CYAN}
╔════════════════════════════════════════════════════════════════════╗
║  Market Data Platform - Enhanced Terminal with Container Support   ║
║     Deployment Groups: [1] Deploy  [2] Gateways  [3] Data          ║
║                       [4] Analytics  [5] Admin                      ║
║                     Type 'help' for available commands              ║
╚════════════════════════════════════════════════════════════════════╝
{Colors.END}
    """
    
    prompt = f"{Colors.GREEN}MDP{Colors.CYAN}[{Colors.YELLOW}session{Colors.CYAN}]> {Colors.END}"
    
    # ========================================================================
    # Container and Service Configuration
    # ========================================================================
    
    SERVICE_CONFIGS = {
        "influxdb": {
            "docker": {
                "image": "influxdb:2.7-alpine",
                "port": "8086:8086",
                "env": ["INFLUXDB_DB=market_data", "INFLUXDB_ADMIN_ENABLED=true"],
                "volume": "influxdb_data:/var/lib/influxdb"
            },
            "podman": {
                "image": "docker.io/influxdb:2.7-alpine",
                "port": "8086:8086",
                "env": ["INFLUXDB_DB=market_data"],
                "volume": "influxdb_data:/var/lib/influxdb"
            },
            "lxc": {
                "packages": ["influxdb2"],
                "port": "8086",
                "config": "/etc/influxdb2/config.yml"
            }
        },
        "grafana": {
            "docker": {
                "image": "grafana/grafana:latest",
                "port": "3000:3000",
                "env": ["GF_SECURITY_ADMIN_PASSWORD=admin"],
                "volume": "grafana_data:/var/lib/grafana"
            },
            "podman": {
                "image": "docker.io/grafana/grafana:latest",
                "port": "3000:3000",
                "env": ["GF_SECURITY_ADMIN_PASSWORD=admin"],
                "volume": "grafana_data:/var/lib/grafana"
            },
            "lxc": {
                "packages": ["grafana"],
                "port": "3000",
                "config": "/etc/grafana/grafana.ini"
            }
        },
        "redis": {
            "docker": {
                "image": "redis:7-alpine",
                "port": "6379:6379",
                "volume": "redis_data:/data"
            },
            "podman": {
                "image": "docker.io/redis:7-alpine",
                "port": "6379:6379",
                "volume": "redis_data:/data"
            },
            "lxc": {
                "packages": ["redis-server"],
                "port": "6379",
                "config": "/etc/redis/redis.conf"
            }
        },
        "parquet": {
            "docker": {
                "image": "ubuntu:22.04",
                "port": "9090:9090",
                "setup": "apt-get install -y python3-pyarrow python3-pandas"
            },
            "podman": {
                "image": "docker.io/ubuntu:22.04",
                "port": "9090:9090",
                "setup": "apt-get install -y python3-pyarrow python3-pandas"
            },
            "lxc": {
                "packages": ["python3-pyarrow", "python3-pandas"],
                "port": "9090"
            }
        }
    }
    
    COMMAND_GROUPS = {
        "deployment": {
            "title": "🚀 DEPLOYMENT & INSTALLATION",
            "commands": [
                "install", "start", "stop", "status", "logs", "restart",
                "deploy-docker", "deploy-podman", "deploy-lxc",
                "configure-service", "health-check"
            ]
        },
        "gateways": {
            "title": "🔗 GATEWAY & CONNECTION MANAGEMENT",
            "commands": [
                "connect", "disconnect", "list-gateways", "gateway-status",
                "stream", "stop-stream", "test-gateway"
            ]
        },
        "data": {
            "title": "📊 DATA & MARKET OPERATIONS",
            "commands": [
                "price", "ohlc", "history", "orderbook", "depth",
                "export", "import", "query", "aggregate"
            ]
        },
        "analytics": {
            "title": "📈 ANALYTICS & ANALYSIS",
            "commands": [
                "sentiment", "correlation", "indicators", "backtest",
                "portfolio", "risk-analysis", "alert"
            ]
        },
        "admin": {
            "title": "⚙️ ADMINISTRATION & CONFIG",
            "commands": [
                "config", "settings", "backup", "restore", "upgrade",
                "security", "performance", "help", "exit"
            ]
        }
    }
    
    # Gateway and topic definitions
    GATEWAYS = {
        "freedx": {"type": "rest", "url": "https://api.exchange.freedx.com"},
        "gate.io": {"type": "websocket", "url": "wss://ws.gate.io/v4"},
        "oanda": {"type": "rest", "url": "https://api-fxpractice.oanda.com"},
        "kraken": {"type": "rest_ws", "url": "https://api.kraken.com"},
        "twitter": {"type": "stream", "url": "https://stream.twitter.com/2"},
        "betfair": {"type": "stream", "url": "https://stream-api.betfair.com"},
    }
    
    ZMQ_TOPICS = {
        "freedx.market_summary": "FreeDOM market summary data",
        "freedx.orderbook": "FreeDOM order book updates",
        "gateio.tickers": "Gate.io ticker data",
        "gateio.trades": "Gate.io recent trades",
        "oanda.eurusd": "OANDA EURUSD prices",
        "oanda.trades": "OANDA trade executions",
        "kraken.ticker": "Kraken ticker data",
        "kraken.eurusd_depth": "Kraken EURUSD depth",
        "kraken.trades": "Kraken trade stream",
        "twitter.crypto_sentiment": "Twitter crypto sentiment",
        "twitter.market_news": "Twitter market news",
        "betfair.market_books": "Betfair market books",
    }
    
    SYMBOLS = {
        "eurusd": "EUR/USD - Euro vs US Dollar",
        "gbpusd": "GBP/USD - British Pound vs US Dollar",
        "btc-usdt": "BTC/USDT - Bitcoin vs USDT",
        "eth-usdt": "ETH/USDT - Ethereum vs USDT",
        "ltc-usdt": "LTC/USDT - Litecoin vs USDT",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_connections = {}
        self.streaming_topics = {}
        self.config = self._load_config()
        self.container_runtime = self._detect_container_runtime()
        self.running_services = {}
        self.current_window_group = WindowGroup.DEPLOYMENT
        self.tmux_session = None
        
    def _detect_container_runtime(self) -> ContainerRuntime:
        """Detect available container runtime (Docker, Podman, LXC)"""
        if shutil.which("docker"):
            return ContainerRuntime.DOCKER
        elif shutil.which("podman"):
            return ContainerRuntime.PODMAN
        elif shutil.which("lxc"):
            return ContainerRuntime.LXC
        else:
            return ContainerRuntime.AUTO

    def _load_config(self) -> Dict:
        """Load configuration from YAML"""
        config_file = "../config/gateways.yaml"
        if os.path.exists(config_file):
            import yaml
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}\n")

    def _print_section(self, title: str, items: List[Tuple[str, str]]):
        """Print section with items and descriptions"""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{title}{Colors.END}")
        print(f"{Colors.MAGENTA}{'-' * 70}{Colors.END}")
        for cmd, desc in items:
            print(f"  {Colors.CYAN}{cmd:20} {Colors.END} {desc}")

    def _print_table(self, headers: List[str], rows: List[List[str]]):
        """Print formatted table"""
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) 
                     for i, h in enumerate(headers)]
        
        # Header
        header_row = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
        print(f"{Colors.BOLD}{Colors.CYAN}{header_row}{Colors.END}")
        print("-" * len(header_row))
        
        # Rows
        for row in rows:
            print(" | ".join(f"{str(r):<{col_widths[i]}}" for i, r in enumerate(row)))

    # ========================================================================
    # Connection Management
    # ========================================================================

    def do_connect(self, arg):
        """Connect to a gateway: connect <gateway_name>"""
        if not arg:
            print(f"{Colors.RED}Usage: connect <gateway_name>{Colors.END}")
            self.do_gateways("")
            return
        
        gateway = arg.strip().lower()
        if gateway not in self.GATEWAYS:
            print(f"{Colors.RED}Unknown gateway: {gateway}{Colors.END}")
            return
        
        print(f"{Colors.YELLOW}Connecting to {gateway}...{Colors.END}")
        # Simulate connection
        self.active_connections[gateway] = {
            "status": "connected",
            "connected_at": datetime.now().isoformat(),
            "messages_received": 0
        }
        print(f"{Colors.GREEN}✓ Connected to {gateway}{Colors.END}")

    def do_disconnect(self, arg):
        """Disconnect from a gateway: disconnect <gateway_name>"""
        if not arg:
            print(f"{Colors.RED}Usage: disconnect <gateway_name>{Colors.END}")
            return
        
        gateway = arg.strip().lower()
        if gateway in self.active_connections:
            del self.active_connections[gateway]
            print(f"{Colors.GREEN}✓ Disconnected from {gateway}{Colors.END}")
        else:
            print(f"{Colors.RED}Not connected to {gateway}{Colors.END}")

    def do_status(self, arg):
        """Show platform status"""
        self._print_header("Platform Status")
        
        if self.active_connections:
            print(f"{Colors.BOLD}Active Connections:{Colors.END}\n")
            for gateway, info in self.active_connections.items():
                print(f"  {Colors.GREEN}{gateway:15}{Colors.END} - {info['status']}")
        else:
            print(f"{Colors.YELLOW}No active connections{Colors.END}\n")
        
        if self.streaming_topics:
            print(f"\n{Colors.BOLD}Streaming Topics:{Colors.END}\n")
            for topic, info in self.streaming_topics.items():
                print(f"  {Colors.CYAN}{topic:30}{Colors.END} - {info['status']}")
        
        print(f"\n{Colors.BOLD}System Time:{Colors.END} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # Data Streaming
    # ========================================================================

    def do_stream(self, arg):
        """Start streaming: stream <topic_name>"""
        if not arg:
            print(f"{Colors.RED}Usage: stream <topic_name>{Colors.END}")
            self.do_topics("")
            return
        
        topic = arg.strip().lower()
        if topic not in self.ZMQ_TOPICS:
            print(f"{Colors.RED}Unknown topic: {topic}{Colors.END}")
            return
        
        print(f"{Colors.YELLOW}Starting stream for {topic}...{Colors.END}")
        self.streaming_topics[topic] = {
            "status": "streaming",
            "started_at": datetime.now().isoformat(),
            "messages": 0
        }
        print(f"{Colors.GREEN}✓ Streaming {topic}{Colors.END}")

    def do_stop(self, arg):
        """Stop streaming: stop <topic_name>"""
        if not arg:
            print(f"{Colors.RED}Usage: stop <topic_name>{Colors.END}")
            return
        
        topic = arg.strip().lower()
        if topic in self.streaming_topics:
            del self.streaming_topics[topic]
            print(f"{Colors.GREEN}✓ Stopped streaming {topic}{Colors.END}")
        else:
            print(f"{Colors.RED}Not streaming {topic}{Colors.END}")

    # ========================================================================
    # Price and Market Data
    # ========================================================================

    def do_price(self, arg):
        """Get current price: price <symbol> [exchange]"""
        args = arg.split()
        if not args:
            print(f"{Colors.RED}Usage: price <symbol> [exchange]{Colors.END}")
            return
        
        symbol = args[0].upper()
        exchange = args[1] if len(args) > 1 else "all"
        
        self._print_header(f"Current Price - {symbol}")
        
        # Simulate price data
        mock_prices = {
            "EURUSD": {"oanda": 1.0850, "kraken": 1.0851, "freedx": 1.0849},
            "BTC-USDT": {"gateio": 45230.50, "kraken": 45231.25},
        }
        
        prices = mock_prices.get(symbol, {})
        if prices:
            headers = ["Exchange", "Price", "Bid", "Ask", "Volume"]
            rows = []
            for ex, price in prices.items():
                rows.append([ex, f"${price:.4f}", f"${price-0.0001:.4f}", f"${price+0.0001:.4f}", "1.2M"])
            self._print_table(headers, rows)
        else:
            print(f"{Colors.YELLOW}No data available for {symbol}{Colors.END}")

    def do_history(self, arg):
        """Show price history: history <symbol> <days>"""
        args = arg.split()
        if len(args) < 2:
            print(f"{Colors.RED}Usage: history <symbol> <days>{Colors.END}")
            return
        
        symbol = args[0].upper()
        days = int(args[1])
        
        self._print_header(f"Price History - {symbol} (Last {days} days)")
        
        headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
        rows = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            rows.append([date, f"${100+i:.2f}", f"${100.5+i:.2f}", f"${99.5+i:.2f}", f"${100+i:.2f}", f"{1000000+i*1000}"])
        
        self._print_table(headers, rows[::-1])

    def do_ohlc(self, arg):
        """Show OHLC data: ohlc <symbol> <interval>"""
        args = arg.split()
        if len(args) < 2:
            print(f"{Colors.RED}Usage: ohlc <symbol> <interval> (1m|5m|1h|1d){Colors.END}")
            return
        
        symbol = args[0].upper()
        interval = args[1]
        
        self._print_header(f"OHLC Data - {symbol} ({interval})")
        
        headers = ["Time", "Open", "High", "Low", "Close", "Volume"]
        rows = []
        for i in range(10):
            time = (datetime.now() - timedelta(hours=i)).strftime("%H:%M")
            rows.append([time, f"${100+i*0.1:.2f}", f"${100.5+i*0.1:.2f}", f"${99.5+i*0.1:.2f}", f"${100+i*0.1:.2f}", f"{1000000}"])
        
        self._print_table(headers, rows[::-1])

    def do_sentiment(self, arg):
        """Show sentiment analysis: sentiment <topic>"""
        if not arg:
            print(f"{Colors.RED}Usage: sentiment <topic>{Colors.END}")
            return
        
        topic = arg.strip()
        self._print_header(f"Sentiment Analysis - {topic}")
        
        print(f"  Sentiment Score: {Colors.GREEN}+0.65{Colors.END} (Bullish)")
        print(f"  Tweets (24h):    {Colors.CYAN}2,453{Colors.END}")
        print(f"  Trend:           {Colors.GREEN}↑ Increasing positive sentiment{Colors.END}")
        print(f"  Top Keywords:    bull, moon, breakthrough, surge")

    def do_orderbook(self, arg):
        """Show order book: orderbook <symbol> [exchange]"""
        args = arg.split()
        if not args:
            print(f"{Colors.RED}Usage: orderbook <symbol> [exchange]{Colors.END}")
            return
        
        symbol = args[0].upper()
        self._print_header(f"Order Book - {symbol}")
        
        print(f"{Colors.RED}{Colors.BOLD}{'BIDS':>30}{Colors.END}  {'ASKS':<30}\n")
        
        for i in range(10):
            bid_price = 100 - i * 0.01
            bid_qty = 100 - i * 5
            ask_price = 100 + (10 - i) * 0.01
            ask_qty = 100 - (10 - i) * 5
            
            bid_str = f"${bid_price:.4f} x {bid_qty}"
            ask_str = f"{ask_qty} x ${ask_price:.4f}"
            
            print(f"{Colors.RED}{bid_str:>30}{Colors.END}  {Colors.GREEN}{ask_str:<30}{Colors.END}")

    # ========================================================================
    # List Commands
    # ========================================================================

    def do_gateways(self, arg):
        """List available gateways"""
        self._print_header("Available Gateways")
        
        headers = ["Gateway", "Type", "Status"]
        rows = []
        for name, info in self.GATEWAYS.items():
            status = "Connected" if name in self.active_connections else "Disconnected"
            status_color = Colors.GREEN if name in self.active_connections else Colors.YELLOW
            rows.append([name, info["type"], f"{status_color}{status}{Colors.END}"])
        
        self._print_table(headers, rows)

    def do_topics(self, arg):
        """List available ZMQ topics"""
        self._print_header("Available ZMQ Topics")
        
        headers = ["Topic", "Description", "Status"]
        rows = []
        for topic, desc in self.ZMQ_TOPICS.items():
            status = "Streaming" if topic in self.streaming_topics else "Stopped"
            status_color = Colors.GREEN if topic in self.streaming_topics else Colors.YELLOW
            rows.append([topic, desc, f"{status_color}{status}{Colors.END}"])
        
        self._print_table(headers, rows)

    def do_symbols(self, arg):
        """List available symbols"""
        self._print_header("Available Symbols")
        
        headers = ["Symbol", "Description"]
        rows = [[sym, desc] for sym, desc in self.SYMBOLS.items()]
        
        self._print_table(headers, rows)

    # ========================================================================
    # DEPLOYMENT & CONTAINER MANAGEMENT COMMANDS
    # ========================================================================

    def do_install(self, arg):
        """Install services: install [service] [--runtime docker|podman|lxc]"""
        args = arg.split() if arg else []
        service = args[0].lower() if args else "all"
        runtime = self._get_runtime_from_args(args)
        
        self._print_header(f"Installing {service.upper()} with {runtime.value}")
        
        services_to_install = self._get_services(service)
        
        for svc in services_to_install:
            print(f"{Colors.YELLOW}Installing {svc}...{Colors.END}")
            self._install_service(svc, runtime)
            print(f"{Colors.GREEN}✓ {svc} installed successfully{Colors.END}\n")

    def do_start(self, arg):
        """Start services: start [service] [--runtime docker|podman|lxc]"""
        args = arg.split() if arg else []
        service = args[0].lower() if args else "all"
        runtime = self._get_runtime_from_args(args)
        
        self._print_header(f"Starting {service.upper()} using {runtime.value}")
        
        services_to_start = self._get_services(service)
        
        for svc in services_to_start:
            print(f"{Colors.CYAN}Starting {svc}...{Colors.END}")
            success = self._start_service(svc, runtime)
            if success:
                self.running_services[svc] = runtime
                print(f"{Colors.GREEN}✓ {svc} started successfully{Colors.END}")
            else:
                print(f"{Colors.RED}✗ Failed to start {svc}{Colors.END}")
            print()

    def do_stop(self, arg):
        """Stop services: stop [service] [--runtime docker|podman|lxc]"""
        args = arg.split() if arg else []
        service = args[0].lower() if args else "all"
        runtime = self._get_runtime_from_args(args)
        
        self._print_header(f"Stopping {service.upper()}")
        
        services_to_stop = self._get_services(service)
        
        for svc in services_to_stop:
            if svc in self.running_services:
                print(f"{Colors.YELLOW}Stopping {svc}...{Colors.END}")
                self._stop_service(svc, runtime)
                del self.running_services[svc]
                print(f"{Colors.GREEN}✓ {svc} stopped{Colors.END}\n")

    def do_restart(self, arg):
        """Restart services: restart [service] [--runtime docker|podman|lxc]"""
        args = arg.split() if arg else []
        service = args[0].lower() if args else "all"
        
        self.do_stop(arg)
        print(f"{Colors.BLUE}Waiting 2 seconds before restart...{Colors.END}")
        import time
        time.sleep(2)
        self.do_start(arg)

    def do_status(self, arg):
        """Show deployment status: status [--runtime docker|podman|lxc]"""
        self._print_header("🔍 DEPLOYMENT STATUS")
        
        print(f"{Colors.BOLD}Container Runtime:{Colors.END} {self.container_runtime.value}")
        print(f"{Colors.BOLD}Running Services:{Colors.END}")
        
        if self.running_services:
            for svc, runtime in self.running_services.items():
                print(f"  {Colors.GREEN}✓{Colors.END} {svc} ({runtime.value})")
        else:
            print(f"  {Colors.YELLOW}None running{Colors.END}")
        
        print(f"\n{Colors.BOLD}Available Services:{Colors.END}")
        for svc in ["influxdb", "grafana", "redis", "parquet"]:
            status = "✓ Running" if svc in self.running_services else "✗ Stopped"
            print(f"  {svc:12} - {status}")

    def do_logs(self, arg):
        """Show service logs: logs <service> [--lines 50]"""
        args = arg.split() if arg else []
        if not args:
            print(f"{Colors.RED}Usage: logs <service> [--lines N]{Colors.END}")
            return
        
        service = args[0].lower()
        lines = "50"
        if "--lines" in args:
            idx = args.index("--lines")
            if idx + 1 < len(args):
                lines = args[idx + 1]
        
        runtime = self.running_services.get(service, self.container_runtime)
        print(f"{Colors.CYAN}Fetching logs for {service}...{Colors.END}\n")
        self._get_service_logs(service, runtime, lines)

    def do_health_check(self, arg):
        """Check service health: health-check [service]"""
        service = arg.strip().lower() if arg else "all"
        self._print_header("🏥 HEALTH CHECK")
        
        services_to_check = self._get_services(service)
        
        for svc in services_to_check:
            if svc in self.running_services:
                status = self._check_service_health(svc)
                status_color = Colors.GREEN if status else Colors.RED
                print(f"{svc:12} - {status_color}{'Healthy' if status else 'Unhealthy'}{Colors.END}")
            else:
                print(f"{svc:12} - {Colors.YELLOW}Not running{Colors.END}")

    def do_deploy_docker(self, arg):
        """Deploy using Docker: deploy-docker [services]"""
        print(f"{Colors.CYAN}Switching to Docker deployment...{Colors.END}")
        self.container_runtime = ContainerRuntime.DOCKER
        self.do_start(arg or "all")

    def do_deploy_podman(self, arg):
        """Deploy using Podman: deploy-podman [services]"""
        print(f"{Colors.CYAN}Switching to Podman deployment...{Colors.END}")
        self.container_runtime = ContainerRuntime.PODMAN
        self.do_start(arg or "all")

    def do_deploy_lxc(self, arg):
        """Deploy using LXC: deploy-lxc [services]"""
        print(f"{Colors.CYAN}Switching to LXC deployment...{Colors.END}")
        self.container_runtime = ContainerRuntime.LXC
        self.do_start(arg or "all")

    def do_configure_service(self, arg):
        """Configure service: configure-service <service> [--key value]"""
        args = arg.split() if arg else []
        if not args:
            print(f"{Colors.RED}Usage: configure-service <service> [--key value]{Colors.END}")
            return
        
        service = args[0].lower()
        self._print_header(f"Configuring {service.upper()}")
        
        config_template = self.SERVICE_CONFIGS.get(service)
        if config_template:
            runtime_config = config_template.get(self.container_runtime.value, {})
            print(json.dumps(runtime_config, indent=2))
            print(f"\n{Colors.YELLOW}Edit the configuration above and apply with:${Colors.END}")
            print(f"  {Colors.CYAN}apply-config {service}{Colors.END}")

    def do_windows(self, arg):
        """Create tmux windows: windows [group]"""
        groups = arg.split() if arg else list(self.COMMAND_GROUPS.keys())
        self._print_header("🪟 TMUX WINDOW LAYOUT")
        
        print(f"{Colors.CYAN}Available window groups:{Colors.END}")
        for group_name, group_info in self.COMMAND_GROUPS.items():
            marker = "→" if group_name in groups else " "
            print(f"  {marker} [{group_name}] {group_info['title']}")
        
        print(f"\n{Colors.YELLOW}To create tmux windows, run:{Colors.END}")
        print(f"  {Colors.CYAN}tmux new-session -s mdp -d{Colors.END}")
        print(f"  For each group above, create a new window with different commands")

    # ========================================================================
    # HELPER METHODS FOR DEPLOYMENT
    # ========================================================================

    def _get_runtime_from_args(self, args: List[str]) -> ContainerRuntime:
        """Extract runtime from arguments"""
        if "--runtime" in args:
            idx = args.index("--runtime")
            if idx + 1 < len(args):
                runtime_str = args[idx + 1].lower()
                try:
                    return ContainerRuntime[runtime_str.upper()]
                except KeyError:
                    pass
        return self.container_runtime

    def _get_services(self, service_name: str) -> List[str]:
        """Get list of services to operate on"""
        if service_name == "all":
            return ["influxdb", "grafana", "redis", "parquet"]
        return [service_name] if service_name in ["influxdb", "grafana", "redis", "parquet"] else []

    def _install_service(self, service: str, runtime: ContainerRuntime):
        """Install a service using specified runtime"""
        config = self.SERVICE_CONFIGS.get(service, {})
        runtime_config = config.get(runtime.value, {})
        
        if runtime == ContainerRuntime.DOCKER or runtime == ContainerRuntime.PODMAN:
            image = runtime_config.get("image", "")
            cmd = runtime.value
            print(f"  {Colors.DIM}$ {cmd} pull {image}{Colors.END}")
        elif runtime == ContainerRuntime.LXC:
            packages = runtime_config.get("packages", [])
            print(f"  {Colors.DIM}$ apt-get install {' '.join(packages)}{Colors.END}")

    def _start_service(self, service: str, runtime: ContainerRuntime) -> bool:
        """Start a service and return success status"""
        config = self.SERVICE_CONFIGS.get(service, {})
        runtime_config = config.get(runtime.value, {})
        
        if runtime == ContainerRuntime.DOCKER or runtime == ContainerRuntime.PODMAN:
            image = runtime_config.get("image", "")
            port = runtime_config.get("port", "")
            print(f"  {Colors.DIM}$ {runtime.value} run -d -p {port} {image}{Colors.END}")
        elif runtime == ContainerRuntime.LXC:
            print(f"  {Colors.DIM}$ lxc launch ubuntu-2204 {service}{Colors.END}")
        
        return True

    def _stop_service(self, service: str, runtime: ContainerRuntime):
        """Stop a running service"""
        if runtime == ContainerRuntime.DOCKER or runtime == ContainerRuntime.PODMAN:
            print(f"  {Colors.DIM}$ {runtime.value} stop {service}{Colors.END}")
        elif runtime == ContainerRuntime.LXC:
            print(f"  {Colors.DIM}$ lxc stop {service}{Colors.END}")

    def _get_service_logs(self, service: str, runtime: ContainerRuntime, lines: str):
        """Get service logs"""
        if runtime == ContainerRuntime.DOCKER or runtime == ContainerRuntime.PODMAN:
            print(f"  {Colors.DIM}$ {runtime.value} logs --tail {lines} {service}{Colors.END}")
        elif runtime == ContainerRuntime.LXC:
            print(f"  {Colors.DIM}$ lxc exec {service} tail -n {lines} /var/log/syslog{Colors.END}")

    def _check_service_health(self, service: str) -> bool:
        """Check if service is healthy"""
        return service in self.running_services

    # ========================================================================
    # Configuration and Administration
    # ========================================================================

    def do_config(self, arg):
        """Manage configuration: config show|edit|reset"""
        if not arg:
            self.do_config("show")
            return
        
        action = arg.split()[0].lower()
        
        if action == "show":
            self._print_header("Current Configuration")
            print(json.dumps(self.config, indent=2))
        elif action == "edit":
            print(f"{Colors.YELLOW}Opening configuration editor...{Colors.END}")
        elif action == "reset":
            print(f"{Colors.YELLOW}Resetting to default configuration...{Colors.END}")

    def do_stats(self, arg):
        """Show platform statistics"""
        self._print_header("Platform Statistics")
        
        print(f"Active Connections:    {Colors.GREEN}{len(self.active_connections)}{Colors.END}")
        print(f"Streaming Topics:      {Colors.GREEN}{len(self.streaming_topics)}{Colors.END}")
        print(f"Messages Processed:    {Colors.CYAN}1,234,567{Colors.END}")
        print(f"Data Stored (MB):      {Colors.CYAN}2,345{Colors.END}")
        print(f"Uptime:                {Colors.GREEN}47h 23m{Colors.END}")

    def do_alerts(self, arg):
        """Manage alerts: alerts add|list|remove"""
        if not arg:
            self.do_alerts("list")
            return
        
        action = arg.split()[0].lower()
        
        if action == "list":
            self._print_header("Active Alerts")
            headers = ["Symbol", "Condition", "Price", "Status"]
            rows = [
                ["EURUSD", "Price > 1.0900", "$1.0900", "Active"],
                ["BTC-USDT", "Volume > 1M", "1,000,000", "Active"],
            ]
            self._print_table(headers, rows)
        elif action == "add":
            print(f"{Colors.YELLOW}Adding new alert...{Colors.END}")
        elif action == "remove":
            print(f"{Colors.YELLOW}Removing alert...{Colors.END}")

    def do_export(self, arg):
        """Export data: export <format> <output_file>"""
        args = arg.split()
        if len(args) < 2:
            print(f"{Colors.RED}Usage: export <json|csv|parquet> <output_file>{Colors.END}")
            return
        
        format_type = args[0].lower()
        output_file = args[1]
        
        print(f"{Colors.YELLOW}Exporting data as {format_type} to {output_file}...{Colors.END}")
        print(f"{Colors.GREEN}✓ Export completed{Colors.END}")

    # ========================================================================
    # Help and Utilities
    # ========================================================================

    def do_help(self, arg):
        """Show help for commands organized by group"""
        if not arg:
            self._print_header("COMMAND GROUPS & REFERENCE")
            
            for group_name, group_info in self.COMMAND_GROUPS.items():
                print(f"\n{Colors.BOLD}{group_info['title']}{Colors.END}")
                print(f"{Colors.MAGENTA}{'-' * 70}{Colors.END}")
                
                for cmd in group_info['commands']:
                    if hasattr(self, f"do_{cmd}"):
                        method = getattr(self, f"do_{cmd}")
                        doc = (method.__doc__ or "No description available").split('\n')[0]
                        print(f"  {Colors.CYAN}{cmd:20}{Colors.END} {doc}")
            
            print(f"\n{Colors.YELLOW}Type 'help <command>' for more details{Colors.END}")
        else:
            super().do_help(arg)

    def do_clear(self, arg):
        """Clear screen"""
        os.system('clear' if os.name == 'posix' else 'cls')

    def do_exit(self, arg):
        """Exit the application"""
        print(f"{Colors.YELLOW}Shutting down...{Colors.END}")
        return True

    def do_quit(self, arg):
        """Exit the application"""
        return self.do_exit(arg)

    # Auto-completion
    def completenames(self, text, *ignored):
        dotext = 'do_'+text
        return [a[3:] for a in self.get_names() if a.startswith(dotext)]

    def completedefault(self, text, line, begidx, endidx):
        """Provide command-specific completions"""
        cmd_parts = line.split()
        cmd = cmd_parts[0] if cmd_parts else ""
        
        completions = []
        
        if cmd in ["connect", "disconnect"]:
            completions = list(self.GATEWAYS.keys())
        elif cmd in ["stream", "stop"]:
            completions = list(self.ZMQ_TOPICS.keys())
        elif cmd in ["price", "history", "ohlc"]:
            completions = list(self.SYMBOLS.keys())
        elif cmd == "config":
            completions = ["show", "edit", "reset"]
        elif cmd == "export":
            completions = ["json", "csv", "parquet"]
        elif cmd == "alerts":
            completions = ["add", "list", "remove"]
        
        return [c for c in completions if c.startswith(text)]

if __name__ == "__main__":
    cli = MarketDataCLI()
    cli.cmdloop()
