# Market Data Platform - Quick Reference & Command Guide

## Quick Start (3 steps)

### 1. Start System
```bash
cd /root/rf_env
bash bin/start.sh
```

### 2. Verify Services
```bash
bash bin/verify_services.sh
```

### 3. Access Dashboard
```bash
# Open in browser: http://localhost:3000
# Username: admin
# Password: admin
```

---

## Essential Commands

### Management
```bash
# Start all services (gracefully)
bash bin/start.sh

# Stop all services (gracefully)
bash bin/stop.sh

# Verify service health
bash bin/verify_services.sh

# Install dependencies
bash bin/install.sh

# Using Makefile
make help                 # Show all targets
make start                # Start services
make stop                 # Stop services
make verify               # Health check
```

### Building
```bash
# Compile C services
cd c/zmq_core
gcc -O3 -Wall publisher.c -o publisher -lzmq
gcc -O3 -Wall subscriber.c -o subscriber -lzmq -lpthread

# Build Go gateway
cd go/cmd/gateway
go build -o gateway

# Build Rust validator
cd rust
cargo build --release
```

### Monitoring
```bash
# Follow logs
tail -f logs/publisher.log
tail -f logs/subscriber.log

# Docker compose logs
docker-compose logs -f postgres
docker-compose logs -f redis
docker-compose logs -f influxdb

# Check running services
docker-compose ps
```

### Data Access
```bash
# Connect to Redis
redis-cli

# Query InfluxDB
influx query 'from(bucket:"market_data") |> range(start: -1h)'

# Connect to PostgreSQL
psql -h localhost -U postgres -d market_data
```

---

## Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| Prometheus | http://localhost:9090 | Metrics browser |
| InfluxDB | http://localhost:8086 | Time-series DB |
| Python API | http://localhost:8000 | REST API |
| Go Gateway | http://localhost:8080 | Data collection |
| Redis | localhost:6379 | Cache/sessions |
| PostgreSQL | localhost:5432 | Main database |

---

## API Examples

### Go Gateway
```bash
# Health check
curl http://localhost:8080/health

# Get current market data
curl http://localhost:8080/metrics | jq .
```

### Python API
```bash
# Health check
curl http://localhost:8000/health

# Market summary
curl http://localhost:8000/api/market/summary
```

---

## ZMQ Message Patterns

### Publisher (5555) - Broadcasting
```
Message Format: [topic] [data]
Topics: freedx:BTC/USD, gateio:ETH/USD
Data: JSON with {symbol, price, volume, timestamp}
```

### Subscriber (5556) - Receiving
```
Subscribes to: tcp://127.0.0.1:5555
Distributes via ROUTER: tcp://127.0.0.1:5556
Load balanced across 4 worker threads
```

### Validator (5557) - Validated Data
```
Subscribes to: tcp://127.0.0.1:5555
Validates and republishes to: tcp://127.0.0.1:5557
Adds validation status and warnings
```

---

## Storage Query Examples

### Redis
```bash
# Get current BTC price
redis-cli GET "market:BTC/USD"

# Get price history (last 1000)
redis-cli LRANGE "history:BTC/USD" 0 -1

# Get all market keys
redis-cli KEYS "market:*"
```

### InfluxDB
```bash
# Query last hour of data
from(bucket: "market_data")
  |> range(start: -1h)
  |> filter(fn: (r) => r.symbol == "BTC/USD")

# Query with aggregation
from(bucket: "market_data")
  |> range(start: -24h)
  |> filter(fn: (r) => r._field == "price")
  |> aggregateWindow(every: 1h, fn: mean)
```

### PostgreSQL
```sql
-- Get user data
SELECT * FROM users;

-- Get trade history
SELECT * FROM trades WHERE user_id = 1 ORDER BY created_at DESC;

-- Get market metadata
SELECT * FROM market_metadata WHERE symbol = 'BTC/USD';
```

---

## Common Troubleshooting

### Services won't start
```bash
# Check logs
tail -f logs/*.log

# Verify dependencies
bash bin/install.sh

# Check ports
netstat -tuln | grep -E ':(8000|8080|5555|5556|6379|8086|3000|9090)'
```

### No data in Grafana
```bash
# Check Go gateway is running
docker-compose ps go-gateway

# Verify Redis has data
redis-cli KEYS market:*

# Check InfluxDB
curl http://localhost:8086/health
```

### C services compilation fails
```bash
# Install ZMQ dev libraries
sudo apt-get install libzmq3-dev libzmq5 pkg-config

# Try compilation again
cd c/zmq_core
gcc -O3 -Wall publisher.c -o publisher -lzmq
```

---

## System Architecture

```
External APIs (FreeDX, Gate.io)
        ↓
    Go Gateway (8080)
        ├→ Redis (Cache)
        ├→ InfluxDB (Storage)
        └→ C Publisher (5555)
              ↓
        C Subscriber (5556)
              ↓
        Rust Validator
              ↓
    InfluxDB / Redis / PostgreSQL
              ↓
        Grafana Dashboard (3000)
```

---

## Performance Metrics

- **Latency**: < 500µs (end-to-end)
- **Throughput**: 1M+ events/hour
- **Data points**: 1000+ symbols
- **Redis TTL**: 30 seconds
- **InfluxDB Retention**: 30 days
- **PostgreSQL**: Permanent storage

---

## Configuration Files

Key configuration locations:
- `config/application/` - Service configs
- `config/database/` - DB settings
- `config/prometheus.yml` - Metrics
- `config/grafana/provisioning/` - Dashboard configs
- `config/nginx/` - Reverse proxy
- `config/env/` - Environment variables

---

## Docker Compose Stack

Services running:
- PostgreSQL (postgres)
- Redis (redis)
- InfluxDB (influxdb)
- Grafana (grafana)
- Prometheus (prometheus)
- Python API (python-api)
- Go Gateway (go-gateway)
- Rust Processor (rust-processor)
- ZMQ Broker (zmq-broker)
- Nginx (nginx)

---

## Useful Docker Commands

```bash
# View all services
docker-compose ps

# View service logs
docker-compose logs <service_name>

# Follow service logs
docker-compose logs -f <service_name>

# Restart service
docker-compose restart <service_name>

# Execute command in container
docker-compose exec postgres psql -U postgres -d market_data

# Stop all services
docker-compose stop

# Start specific service
docker-compose up -d <service_name>
```

---

## Database Management

```bash
# Initialize schema
docker-compose exec postgres psql -U postgres -d market_data < database/schema.sql

# Backup database
docker-compose exec postgres pg_dump -U postgres market_data > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres -d market_data < backup.sql

# Connect to DB
docker-compose exec postgres psql -U postgres -d market_data
```

---

## Performance Tuning

### C Services (publisher/subscriber)
- Compiled with `-O3` optimization
- High water mark (HWM): 10000
- Multi-threaded: 4 worker threads

### Go Gateway
- Connection pooling
- Concurrent data fetching
- 5-second refresh interval

### Python API
- Async request handling
- Connection pooling to databases
- Caching layer via Redis

---

## Scaling

### Horizontal
- Add more Go gateway instances
- Scale Rust validators
- Add Python API replicas
- Use Kubernetes auto-scaling

### Vertical
- Increase container resources
- Tune database connection pools
- Adjust ZMQ HWM values
- Optimize query patterns

---

## Deployment to Kubernetes

```bash
# Apply manifests
kubectl apply -f kubernetes/

# Check deployment
kubectl get pods -n market-data

# View service
kubectl get svc -n market-data

# Scale deployment
kubectl scale deployment market-data-gateway --replicas=3
```

---

## System Requirements

**Minimum**:
- 2 CPU cores
- 4GB RAM
- 20GB disk

**Recommended**:
- 4 CPU cores
- 16GB RAM
- 100GB disk

**Network**:
- ZMQ ports: 5555, 5556, 5557
- HTTP ports: 8000, 8080, 3000, 9090, 8086

---

## Support & Resources

- Check logs: `tail -f logs/*.log`
- Run health check: `bash bin/verify_services.sh`
- Review docs: `DEPLOYMENT_AND_TESTING_GUIDE.md`
- Architecture: `MARKET_DATA_PLATFORM_DELIVERY.md`
- Quick ref: `TERMINAL_QUICK_REFERENCE.md`

---

## Next Steps

1. ✅ Start system: `bash bin/start.sh`
2. ✅ Verify health: `bash bin/verify_services.sh`
3. ✅ Access Grafana: `http://localhost:3000`
4. ✅ Check data: `redis-cli KEYS market:*`
5. ✅ Create dashboard: Use Grafana UI
6. ✅ Monitor metrics: `http://localhost:9090`
7. ✅ Setup alerts: Grafana alerting
8. ✅ Deploy to K8s: `kubectl apply -f kubernetes/`

