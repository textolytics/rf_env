# Market Data Platform - System Deployment and Testing Guide

## Quick Start

### 1. System Verification
```bash
cd /root/rf_env
bash bin/verify_services.sh
```

### 2. Install Dependencies
```bash
# Install ZMQ development libraries
sudo apt-get install -y libzmq3-dev libzmq5 pkg-config

# For Go gateway
go mod download
go mod tidy

# For Rust validator
cd rust && cargo build --release
```

### 3. Start All Services (Gracefully)
```bash
bash bin/start.sh
```

This will:
- Start PostgreSQL (initialization + schema)
- Start Redis (cache layer)
- Start InfluxDB (time-series storage)
- Start Grafana (visualization dashboard)
- Start Prometheus (metrics collection)
- Compile and start C ZMQ services
- Start Python API (port 8000)
- Start Go Gateway (port 8080)
- Start Rust Validator (internal)
- Start Nginx (reverse proxy)

### 4. Monitor Services
```bash
# Watch service status
docker-compose ps

# Check logs in real-time
tail -f logs/publisher.log
tail -f logs/subscriber.log

# Service health checks
bash bin/verify_services.sh
```

### 5. Graceful Shutdown
```bash
bash bin/stop.sh
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│         External Data Sources                               │
│  ┌─────────────────┐    ┌──────────────────┐                │
│  │   FreeDX API    │    │   Gate.io API    │                │
│  └────────┬────────┘    └────────┬─────────┘                │
└───────────┼──────────────────────┼──────────────────────────┘
            │                      │
            └──────────┬───────────┘
                       │ (HTTP/WebSocket)
            ┌──────────▼──────────┐
            │   Go Gateway        │ (Port 8080)
            │ - Fetch market data │
            │ - Store to Redis    │
            │ - Route to ZMQ      │
            └──────────┬──────────┘
                       │
                       │ (ZMQ PUB)
            ┌──────────▼──────────┐
            │  C Publisher (5555) │
            │ - Broadcast market  │
            │   data via ZMQ PUB  │
            └──────────┬──────────┘
                       │
                       │ (ZMQ SUB)
            ┌──────────▼──────────┐
            │  C Subscriber (5556)│
            │ - Route messages    │
            │ - Load distribution │
            └──────────┬──────────┘
                       │
                       │ (ZMQ SUB)
            ┌──────────▼──────────┐
            │ Rust Validator      │
            │ - Validate data     │
            │ - Schema check      │
            │ - Outlier detect    │
            └──────────┬──────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼───────┐ ┌──▼─────────┐ ┌──▼────────────┐
   │   Redis    │ │ InfluxDB   │ │ PostgreSQL   │
   │  (Cache)   │ │(Time-series)│ │ (Relational) │
   └────┬───────┘ └──┬─────────┘ └──┬────────────┘
        │            │              │
        └────────────┼──────────────┘
                     │
            ┌────────▼────────┐
            │   Grafana       │ (Port 3000)
            │  - Dashboard    │ (admin/admin)
            │  - Real-time    │
            │    visualization│
            └─────────────────┘
```

---

## Component Details

### C ZMQ Core Services

**Publisher** (`c/zmq_core/publisher.c`)
- Broadcasts market data via ZMQ PUB socket
- Listens on `tcp://127.0.0.1:5555`
- Signal handlers for graceful shutdown (SIGINT, SIGTERM)
- High-throughput message envelope pattern
- High water mark (HWM) optimization: 10000 messages

**Subscriber/Router** (`c/zmq_core/subscriber.c`)
- Subscribes to publisher via ZMQ SUB socket
- Routes to clients via ZMQ ROUTER socket
- Listens on `tcp://127.0.0.1:5556`
- 4-thread worker pool for parallel processing
- Load distribution across workers

### Go Gateway Layer

**Location**: `go/cmd/gateway/main.go`

Features:
- Connects to FreeDX and Gate.io APIs
- Fetches market data (BTC/USD, ETH/USD, etc.)
- Stores to Redis cache (TTL: 30 seconds)
- Maintains historical data (last 1000 records)
- Publishes to C Publisher via ZMQ
- Health check: `GET http://localhost:8080/health`
- Metrics: `GET http://localhost:8080/metrics`

### Rust Data Validator

**Location**: `rust/src/bin/validator.rs`

Features:
- Validates incoming market data
- Schema validation (symbol format, price, volume)
- Timestamp validation (±24 hour window)
- Threshold checking (price < $1M, volume < 1B)
- Publishes validated data to `tcp://127.0.0.1:5557`
- Error tracking and logging
- 100% reliability for critical data paths

### Data Storage

**Redis** (Port 6379)
- Live market data: `market:BTC/USD` → JSON
- Historical data: `history:BTC/USD` → List (last 1000)
- Session storage
- Cache with TTL

**InfluxDB** (Port 8086)
- Time-series market data storage
- 30-day retention policy
- Query language: InfluxQL/Flux
- Designed for time-range queries

**PostgreSQL** (Port 5432)
- User data
- Session management
- Historical analysis
- Reporting

---

## Service Endpoints

| Service      | Endpoint              | Purpose              |
|-------------|----------------------|----------------------|
| API         | http://localhost:8000| Python REST API      |
| Gateway     | http://localhost:8080| Go data collection   |
| Grafana     | http://localhost:3000| Visualization        |
| Prometheus  | http://localhost:9090| Metrics              |
| InfluxDB    | http://localhost:8086| Time-series DB       |
| Redis       | localhost:6379       | Cache layer          |
| PostgreSQL  | localhost:5432       | Relational DB        |

---

## Data Flow Testing

### 1. Start Services
```bash
bash bin/start.sh
```

### 2. Verify ZMQ Message Flow
```bash
# In terminal 1: Start publisher
./c/zmq_core/publisher

# In terminal 2: Start subscriber (should see messages)
./c/zmq_core/subscriber
```

### 3. Check Gateway Data
```bash
# View current market data
curl http://localhost:8080/metrics | jq .

# Check gateway health
curl http://localhost:8080/health | jq .
```

### 4. Verify Redis
```bash
# Connect to Redis
redis-cli

# View keys
KEYS market:*
LLEN history:BTC/USD
```

### 5. Query InfluxDB
```bash
# Connect to InfluxDB UI
http://localhost:8086

# Or use CLI
influx query 'from(bucket:"market_data") |> range(start: -1h)'
```

### 6. Dashboard Verification
```bash
# Open Grafana
http://localhost:3000

# Login: admin / admin
# Create dashboard for market data topics
```

---

## Graceful Shutdown Procedure

Services shut down in proper order:

1. Application services (Python API, Go Gateway, Rust Validator)
2. ZMQ services (Publisher, Subscriber)
3. Infrastructure (Nginx, Prometheus, Grafana)
4. Storage services (InfluxDB, Redis, PostgreSQL)

```bash
bash bin/stop.sh
```

This ensures:
- Connections are closed gracefully
- Buffers are flushed
- No data loss
- Clean shutdown logs

---

## Performance Tuning

### C Services (publisher/subscriber)
- Compiled with `-O3` optimization flag
- High water mark (HWM) set to 10000 for buffering
- Multi-threaded worker pool (4 threads) in subscriber

### Go Gateway
- Concurrent data fetching (2 goroutines)
- Connection pooling to Redis and ZMQ
- 5-second data refresh interval

### Rust Validator
- Compiled in release mode (`cargo build --release`)
- Real-time data validation with minimal overhead
- Efficient memory management via Rust ownership

---

## Monitoring and Metrics

### Prometheus Targets
- PostgreSQL exporter
- Redis exporter
- Go metrics
- Python metrics
- System metrics

### Grafana Dashboards
Create dashboards for:
- Market data by symbol
- Volume trends
- Price movements
- Data freshness
- System health
- Service uptime

---

## Troubleshooting

### Services Won't Start
```bash
# Check logs
tail -f logs/*.log

# Verify dependencies
bash bin/install.sh

# Check ports
netstat -tuln | grep -E ':(8000|8080|5555|5556|6379|8086|3000|9090)'
```

### No Data in Grafana
1. Check Go gateway is running: `curl http://localhost:8080/health`
2. Verify Redis has data: `redis-cli KEYS market:*`
3. Check InfluxDB has measurements: `influx bucket list`
4. Verify Grafana datasource configuration

### Signal Handling Issues
```bash
# Test graceful shutdown
kill -SIGTERM <service_pid>

# Check for proper cleanup in logs
tail -f logs/publisher.log
tail -f logs/subscriber.log
```

---

## Docker Compose Services

```yaml
Services defined in docker-compose.yml:
- postgres: PostgreSQL database
- redis: Redis cache
- influxdb: Time-series database
- grafana: Visualization dashboard
- prometheus: Metrics collection
- python-api: Python API server
- go-gateway: Go data collection
- rust-processor: Rust data validation
- nginx: Reverse proxy
```

Start individual services:
```bash
docker-compose up -d postgres
docker-compose up -d redis
docker-compose up -d influxdb
docker-compose up -d grafana
```

---

## API Examples

### Python API
```bash
# Get API health
curl http://localhost:8000/health

# Get market summary
curl http://localhost:8000/api/market/summary

# Get symbol data
curl http://localhost:8000/api/market/BTC/USD
```

### Go Gateway
```bash
# Get gateway health
curl http://localhost:8080/health

# Get current metrics
curl http://localhost:8080/metrics | jq .
```

---

## Regression Testing

Run comprehensive tests:
```bash
# Python tests
pytest tests/ -v

# Go tests
go test ./... -v

# Rust tests
cd rust && cargo test --release
```

---

## Next Steps

1. ✅ Components created and compiled
2. ✅ Scripts for start/stop/verify
3. 🟡 Start full system: `bash bin/start.sh`
4. 🟡 Configure Grafana dashboards
5. 🟡 Run regression tests
6. 🟡 Verify data flow end-to-end
7. 🟡 Monitor for 24 hours
8. 🟡 Deploy to Kubernetes

