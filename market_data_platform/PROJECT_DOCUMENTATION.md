# Market Data Platform - Project Documentation

## Overview

The Market Data Platform is a sophisticated, multi-gateway market data aggregation system designed for traders, researchers, and market analysts. It provides a **Bloomberg Terminal-style interface** with real-time streaming, historical data storage, sentiment analysis, and comprehensive CLI navigation.

### Key Capabilities

- **Multi-Gateway Integration**: FreeDOM Exchange, Gate.io, OANDA, Kraken, Twitter, Betfair
- **Real-Time Streaming**: ZMQ-based pub/sub messaging (high-performance message distribution)
- **Time-Series Storage**: InfluxDB for metrics, Apache Parquet for batch analytics
- **Visualization**: Grafana dashboards for market monitoring
- **Sentiment Analysis**: Twitter sentiment tracking for market mood
- **CLI Interface**: Command-line terminal with auto-completion, suggestions
- **Testing Framework**: Robot Framework integration tests with comprehensive coverage
- **Rust/C++ Connectivity**: High-performance modules for gateway connections

---

## Project Structure

```
market_data_platform/
├── connectivity/              # Gateway and API connectivity modules
│   ├── freedx_module.rs      # Rust: FreeDOM Exchange REST connector
│   ├── gateio_module.rs      # Rust: Gate.io WebSocket connector
│   ├── oanda_module.rs       # Rust: OANDA Forex API
│   ├── kraken_module.rs      # Rust: Kraken hybrid REST+WS
│   ├── twitter_module.cpp    # C++: Twitter Sentiment extraction
│   ├── betfair_module.cpp    # C++: Betfair streaming
│   └── zmq_broker.py         # Python: ZMQ message broker & distributor
│
├── storage/                   # Data persistence layer
│   ├── influxdb_client.py    # InfluxDB time-series client
│   ├── parquet_writer.py     # Parquet batch storage
│   ├── schema_definitions.py # Data schemas (OHLC, Trades, Sentiment)
│   └── retention_policies.py # Data lifecycle management
│
├── research/                  # Research and analysis modules
│   ├── eurusd_analyzer.py    # EURUSD-specific analysis
│   ├── technical_indicators.py
│   ├── correlation_engine.py # Portfolio correlation analysis
│   ├── sentiment_aggregator.py
│   └── notebooks/            # Jupyter notebooks for exploration
│
├── cli/                       # Command-Line Interface
│   ├── terminal.py           # Main CLI (cmd.Cmd + typer)
│   ├── commands/             # Command implementations
│   ├── completers.py         # Auto-completion suggestions
│   └── themes.py             # Output formatting & colors
│
├── testing/                   # Test suite
│   ├── integration_tests.robot   # RF integration tests
│   ├── keywords/
│   │   ├── gateway_keywords.robot
│   │   ├── storage_keywords.robot
│   │   └── cli_keywords.robot
│   └── conftest.py           # Pytest fixtures
│
├── config/                    # Configuration files
│   ├── gateways.yaml         # Gateway credentials & endpoints
│   ├── influxdb.yaml         # InfluxDB settings
│   ├── zmq_topics.yaml       # ZMQ topic definitions
│   └── research_config.yaml  # Research module settings
│
├── docker/                    # Docker setup
│   ├── docker-compose.yaml   # InfluxDB + Grafana services
│   ├── influxdb.dockerfile   # InfluxDB with pre-loaded configs
│   └── grafana.dockerfile    # Grafana with market dashboards
│
└── docs/                      # Documentation
    ├── ARCHITECTURE.md       # System design
    ├── API_REFERENCE.md      # Gateway API docs
    ├── SETUP_GUIDE.md        # Installation & setup
    └── EXAMPLES.md           # Usage examples
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Rust 1.70+ (for connectivity modules)
- C++ compiler (g++ or clang)
- Robot Framework 7.4.1+
- Grafana 10.0+ (via Docker)

### Installation

#### 1. Clone and Setup Python Environment

```bash
cd market_data_platform
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

#### 2. Start Infrastructure Services

```bash
cd docker
docker-compose up -d

# Verify services
docker-compose ps
```

This starts:
- **InfluxDB** on `http://localhost:8086` (org: market_data, bucket: market_data_bucket)
- **Grafana** on `http://localhost:3000` (admin/admin)
- **ZMQ Broker** on `127.0.0.1:5555`

#### 3. Configure Gateway Credentials

Create `config/gateways.yaml`:

```yaml
freedx:
  api_url: "https://api.exchange.freedx.com"
  api_key: "YOUR_KEY"
  api_secret: "YOUR_SECRET"

gateio:
  ws_url: "wss://ws.gate.io/v4"
  api_key: "YOUR_KEY"
  api_secret: "YOUR_SECRET"

oanda:
  base_url: "https://api-fxpractice.oanda.com"
  account_id: "YOUR_ACCOUNT"
  token: "YOUR_TOKEN"

kraken:
  rest_url: "https://api.kraken.com"
  ws_url: "wss://ws.kraken.com"
  api_key: "YOUR_KEY"
  api_secret: "YOUR_SECRET"

twitter:
  bearer_token: "YOUR_BEARER_TOKEN"
  search_query: "crypto OR bitcoin OR ethereum"

betfair:
  username: "YOUR_USERNAME"
  password: "YOUR_PASSWORD"
  app_key: "YOUR_APP_KEY"
```

#### 4. Initialize InfluxDB

```bash
./scripts/init_influxdb.sh
```

This creates:
- Organization: `market_data`
- Bucket: `market_data_bucket` (infinite retention)
- Measurement: `market_ticks`, `trades`, `sentiment`, `ohlc_candles`

---

## Usage

### 1. Start the CLI

```bash
python cli/terminal.py
```

Interactive commands:

```
MDP> connect freedx          # Connect to gateway
MDP> price EURUSD            # Get current price
MDP> stream freedx.market_summary  # Stream live data
MDP> sentiment crypto        # View sentiment analysis
MDP> ohlc EURUSD --timeframe 1h   # View candles
MDP> export json /tmp/data.json   # Export data
MDP> help                    # Show all commands
```

### 2. Run Research Analysis

```bash
# EURUSD-specific analysis
python research/eurusd_analyzer.py --lookback 30 --strategy breakout

# Correlation analysis
python research/correlation_engine.py --symbols EURUSD,GBPUSD,USDJPY

# Sentiment aggregation
python research/sentiment_aggregator.py --source twitter --topic crypto
```

### 3. Execute Integration Tests

```bash
# Run all tests
robot testing/integration_tests.robot

# Run specific suite
robot --include gateway testing/integration_tests.robot

# Run with coverage
robot --collect-only testing/integration_tests.robot
```

### 4. View Dashboards

Open Grafana: `http://localhost:3000`

Pre-configured dashboards:
- **Market Overview**: Real-time pricing, volume, sentiment
- **EURUSD Deep Dive**: Bid/ask spreads, depth, correlation with other pairs
- **Portfolio Performance**: P&L, correlations, risk metrics
- **Sentiment Analysis**: Social media mood, trending topics

---

## Architecture Components

### Connectivity Layer (`connectivity/`)

Each gateway has dedicated modules:

| Gateway | Type | Module | Protocol |
|---------|------|--------|----------|
| FreeDOM Exchange | REST | `freedx_module.rs` | HTTP/REST |
| Gate.io | WebSocket | `gateio_module.rs` | WS + REST |
| OANDA | REST | `oanda_module.rs` | HTTP/REST |
| Kraken | Hybrid | `kraken_module.rs` | REST + WS |
| Twitter | Stream | `twitter_module.cpp` | REST + Stream |
| Betfair | Stream | `betfair_module.cpp` | HTTP Streaming |

All modules forward data via **ZMQ Pub/Sub** to centralized broker.

### Message Flow

```
[Gateway 1]--\
[Gateway 2]--+---> [ZMQ Broker] ---> [Storage Layer]
[Gateway N]--/                  ---> [CLI/Dashboards]
```

### Storage Layer (`storage/`)

- **InfluxDB**: Hot storage for recent market ticks (1-7 days)
- **Parquet**: Cold storage for historical analysis (archival)
- **Python Clients**: Abstracted APIs for data writing and querying

### Research Layer (`research/`)

Specialized analysis modules:
- **Technical Indicators**: SMA, EMA, RSI, Bollinger Bands, MACD
- **Correlation Engine**: Portfolio-level correlation matrices
- **Sentiment Aggregation**: Twitter/Reddit mood synthesis
- **EURUSD Analysis**: Forex-specific strategies and patterns

---

## ZMQ Topic Reference

Published topics:

```
freedx.market_summary       # OHLC, volume, spread
freedx.depth                # Order book (bid/ask levels)
gateio.tickers              # Crypto tickers
gateio.trades               # Recent trades
oanda.eurusd                # EURUSD prices & spreads
oanda.trades                # Trade executions
kraken.eurusd_depth         # EURUSD order book
kraken.eurusd_tick          # EURUSD tick data
twitter.sentiment           # Aggregated sentiment scores
betfair.market_books        # Betting market data
```

Subscribe to any topic:

```python
# Python subscriber
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5555")
socket.subscribe(b"oanda.eurusd")

while True:
    message = socket.recv_json()
    print(message)
```

---

## API Reference

### CLI Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `connect` | `<gateway>` | Connect to data gateway |
| `disconnect` | `<gateway>` | Disconnect |
| `price` | `<symbol>` `[--gateway]` | Current price |
| `stream` | `<topic>` `[--duration]` | Start streaming |
| `stop` | `<topic>` | Stop streaming |
| `sentiment` | `<topic>` `[--source]` | Sentiment analysis |
| `ohlc` | `<symbol>` `--timeframe` | OHLC candles |
| `depth` | `<symbol>` `[--levels]` | Order book depth |
| `export` | `<format>` `<output>` | Export data |
| `status` | | Platform status |
| `config` | `show\|set\|reset` | Configuration management |

### InfluxDB Queries

```flux
# Recent market ticks
from(bucket: "market_data_bucket")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "market_ticks" and r.symbol == "EURUSD")

# OHLC candles
from(bucket: "market_data_bucket")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "ohlc_candles" and r.timeframe == "1h")

# Sentiment timeline
from(bucket: "market_data_bucket")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement == "sentiment" and r.topic == "crypto")
```

---

## Configuration

### `config/gateways.yaml`

Credentials, endpoints, and retry policies for each gateway.

### `config/influxdb.yaml`

InfluxDB host, token, org, bucket settings.

### `config/zmq_topics.yaml`

Topic subscriptions, message formats, retention policies.

### Environment Variables

```bash
# InfluxDB
INFLUXDB_URL=http://localhost:8086
INFLUXDB_ORG=market_data
INFLUXDB_BUCKET=market_data_bucket
INFLUXDB_TOKEN=your_token

# ZMQ
ZMQ_BROKER_HOST=127.0.0.1
ZMQ_BROKER_PORT=5555

# API Keys (load from .env)
OANDA_API_TOKEN=your_token
TWITTER_BEARER_TOKEN=your_token
```

---

## Testing

### Robot Framework Integration Tests

```bash
# All tests
robot testing/integration_tests.robot

# Specific tags
robot --include connectivity testing/integration_tests.robot
robot --include zmq testing/integration_tests.robot
robot --include storage testing/integration_tests.robot

# Generate reports
robot --output results/output.xml testing/integration_tests.robot
rebot results/output.xml
```

### Test Coverage

- **Gateway Connectivity**: Connection, disconnect, data fetch
- **ZMQ Messaging**: Pub/sub, topics, throughput
- **Storage**: Write, query, retention, backup
- **CLI**: Commands, completion, output formatting
- **Data Quality**: Price validation, volume consistency
- **Performance**: Latency, throughput, stress tests

---

## Performance Tuning

### ZMQ Optimization

```yaml
# config/zmq_topics.yaml
high_frequency:
  topic: "kraken.eurusd_tick"
  buffer_size: 10000
  hwm: 100  # High-water mark
```

### InfluxDB Optimization

- Use batch writes: collect 1000+ records before flush
- Compression: snappy (default)
- Shard retention: 7 days for hot storage
- Downsample old data to lower resolution

### CLI Performance

- Command caching for auto-completion
- Lazy-load heavy modules
- Connection pooling for gateway APIs

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| InfluxDB connection refused | Check `docker-compose ps`, restart container |
| ZMQ "Address already in use" | Change port in config or kill existing process |
| Gateway auth fails | Verify API keys in `config/gateways.yaml` |
| Slow CLI response | Check gateway latency with `mcp status` |
| Robot tests fail | Ensure InfluxDB + ZMQ running: `docker-compose up -d` |
| Memory leak in streaming | Check subscriber cleanup in `zmq_broker.py` |

---

## Contributing

1. Create feature branch: `git checkout -b feature/new-gateway`
2. Develop and test locally
3. Submit PR with test coverage
4. CI/CD runs full test suite before merge

---

## License

MIT License - See LICENSE file for details

---

## Support

- **Documentation**: [docs/](./docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@marketdata.example.com

---

## Roadmap

### Q1 2024
- [ ] Native browser UI (React.js)
- [ ] Multi-portfolio support
- [ ] Advanced alerting system

### Q2 2024
- [ ] Machine learning prediction module
- [ ] Options pricing engine
- [ ] Risk analytics dashboard

### Q3 2024
- [ ] Backtesting framework
- [ ] Strategy optimization
- [ ] Paper trading integration

### Q4 2024
- [ ] Live trading connectors
- [ ] Risk management automation
- [ ] Regulatory reporting

---

**Last Updated**: 2024-01-15  
**Version**: 1.0.0  
**Maintainers**: Market Data Team
