# Market Data Platform - Deployment & Operations Guide

## Deployment Architecture

The Market Data Platform is designed for **production deployment** across three layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   CLI Tool   │  │   Grafana    │  │  Python API  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  APPLICATION LAYER                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Market Data Platform (Python/Rust/C++)             │  │
│  │  ┌─────────────┐  ┌─────────────┐ ┌────────────┐   │  │
│  │  │  CLI Cmd    │  │ Research    │ │ Adapters   │   │  │
│  │  │  Execution  │  │ Modules     │ │ (REST API) │   │  │
│  │  └─────────────┘  └─────────────┘ └────────────┘   │  │
│  │              ↓                           ↓          │  │
│  │         ┌────────────────────────────────┐         │  │
│  │         │  ZMQ Broker (Pub/Sub)          │         │  │
│  │         └────────────────────────────────┘         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   DATA LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  InfluxDB    │  │  Parquet     │  │  Redis      │      │
│  │  (Hot)       │  │  (Cold)      │  │  (Cache)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 GATEWAY LAYER                               │
│  FreeDOM │ Gate.io │ OANDA │ Kraken │ Twitter │ Betfair   │
└─────────────────────────────────────────────────────────────┘
```

---

## Docker Deployment

### 1. Build Images

```bash
cd market_data_platform/docker

# Build InfluxDB image
docker build -f influxdb.dockerfile -t market-influxdb:1.0 .

# Build Grafana image with pre-configured dashboards
docker build -f grafana.dockerfile -t market-grafana:1.0 .

# Build application image
docker build -f app.dockerfile -t market-data-platform:1.0 ..
```

### 2. Docker Compose - Full Stack

```yaml
# docker-compose.yml
version: '3.9'

services:
  # InfluxDB - Time-series database
  influxdb:
    image: market-influxdb:1.0
    container_name: market-influxdb
    ports:
      - "8086:8086"
    environment:
      INFLUXDB_DB: market_data
      INFLUXDB_ADMIN_USER: admin
      INFLUXDB_ADMIN_PASSWORD: ${INFLUXDB_PASSWORD}
      INFLUXDB_RETENTION: 30d
    volumes:
      - influxdb_data:/var/lib/influxdb
    restart: unless-stopped
    networks:
      - market-network

  # Grafana - Visualization
  grafana:
    image: market-grafana:1.0
    container_name: market-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_INSTALL_PLUGINS: grafana-worldmap-panel
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - influxdb
    restart: unless-stopped
    networks:
      - market-network

  # Redis - Caching layer
  redis:
    image: redis:7-alpine
    container_name: market-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - market-network

  # Market Data Platform
  market-app:
    image: market-data-platform:1.0
    container_name: market-app
    ports:
      - "5000:5000"      # REST API
      - "5555:5555"      # ZMQ Broker
    environment:
      INFLUXDB_URL: http://influxdb:8086
      REDIS_URL: redis://redis:6379
      LOG_LEVEL: INFO
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
    depends_on:
      - influxdb
      - redis
    restart: unless-stopped
    networks:
      - market-network

volumes:
  influxdb_data:
  grafana_data:
  redis_data:

networks:
  market-network:
    driver: bridge
```

### 3. Run Stack

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f market-app

# Monitor resources
docker stats

# Verify services
docker-compose ps
```

---

## Kubernetes Deployment

### 1. Helm Chart Structure

```
market-data-helm/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── statefulset-influxdb.yaml
│   ├── ingress.yaml
│   └── hpa.yaml
```

### 2. Deploy to Kubernetes

```bash
# Install Helm chart
helm install market-data ./market-data-helm \
  --namespace market-data \
  --create-namespace \
  --values values-production.yaml

# Verify deployment
kubectl get pods -n market-data
kubectl get services -n market-data

# Check logs
kubectl logs -n market-data deployment/market-data-platform

# Port forwarding for local access
kubectl port-forward -n market-data svc/market-data-platform 5000:5000
```

### 3. values-production.yaml

```yaml
replicaCount: 3

image:
  repository: market-data-platform
  tag: 1.0
  pullPolicy: IfNotPresent

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

influxdb:
  enabled: true
  storage:
    size: 100Gi
  retention: 30d

ingress:
  enabled: true
  hosts:
    - market-data.example.com

redis:
  enabled: true
  replicas: 2
```

---

## Performance Optimization

### 1. ZMQ Optimization

```python
# config/zmq_config.py
ZMQ_CONFIG = {
    "broker": {
        "host": "0.0.0.0",
        "port": 5555,
        "io_threads": 4,
        "hwm": {  # High Water Mark
            "publisher": 100000,
            "subscriber": 50000
        }
    },
    "topics": {
        "high_frequency": {
            "oanda.eurusd": {"buffer": 10000, "priority": 10},
            "kraken.eurusd_depth": {"buffer": 5000, "priority": 9},
        },
        "low_frequency": {
            "twitter.sentiment": {"buffer": 100, "priority": 1},
        }
    }
}
```

### 2. InfluxDB Tuning

```bash
# InfluxDB configuration
[http]
  bind-address = "0.0.0.0:8086"
  max-body-size = 536870912  # 512MB
  max-connections = 1000

[data]
  cache-max-memory-bytes = 1073741824  # 1GB
  cache-snapshot-memory-bytes = 26214400  # 25MB

[retention]
  check-interval = "10m"

[shard-precreation]
  advance-period = "30m"
  check-interval = "10m"
```

### 3. Connection Pooling

```python
# storage/influxdb_client.py
from influxdb_client import InfluxDBClient
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

class OptimizedInfluxDBClient:
    def __init__(self, url, token, org, pool_size=10):
        self.client = InfluxDBClient(
            url=url,
            token=token,
            org=org,
            enable_gzip=True
        )
        
        # Connection pooling
        session = self.client.api_client.configuration.proxy_headers
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_size,
            pool_maxsize=pool_size
        )
```

### 4. Caching Strategy

```python
# caching/redis_cache.py
import redis
from functools import wraps
import json

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_result(ttl=300):
    """Cache frequently accessed data"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Check cache
            result = redis_client.get(cache_key)
            if result:
                return json.loads(result)
            
            # Compute and cache
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache_result(ttl=60)
def get_current_price(symbol):
    """Cache price for 60 seconds"""
    # Fetch from gateway
    ...
```

---

## Monitoring & Alerting

### 1. Prometheus Metrics

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Message counts
messages_processed = Counter(
    'market_data_messages_total',
    'Total messages processed',
    ['gateway', 'topic']
)

# Latency tracking
gateway_latency = Histogram(
    'market_data_gateway_latency_seconds',
    'Gateway response latency',
    ['gateway'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

# Active connections
active_connections = Gauge(
    'market_data_active_connections',
    'Active gateway connections',
    ['gateway']
)

# Storage metrics
influxdb_writes = Counter(
    'market_data_influxdb_writes_total',
    'InfluxDB writes',
    ['measurement', 'status']
)
```

### 2. Alerting Rules

```yaml
# prometheus-rules.yml
groups:
  - name: market-data
    interval: 30s
    rules:
      # Gateway connection loss
      - alert: GatewayConnectionLost
        expr: market_data_active_connections == 0
        for: 5m
        annotations:
          summary: "Gateway {{ $labels.gateway }} connection lost"

      # High latency
      - alert: HighGatewayLatency
        expr: histogram_quantile(0.95, gateway_latency) > 1.0
        for: 5m
        annotations:
          summary: "High latency on {{ $labels.gateway }}"

      # Storage failures
      - alert: InfluxDBWriteFailure
        expr: rate(market_data_influxdb_writes_total{status="error"}[5m]) > 0.1
        for: 10m
        annotations:
          summary: "InfluxDB write errors detected"
```

### 3. Grafana Dashboards

Pre-configured dashboards:

- **System Health**: CPU, memory, disk usage
- **Message Flow**: Throughput, latency, errors
- **Data Quality**: Missing ticks, duplicate messages
- **Storage**: InfluxDB performance, retention

---

## Backup & Disaster Recovery

### 1. InfluxDB Backup

```bash
#!/bin/bash
# backup_influxdb.sh

BACKUP_DIR="/backups/influxdb"
RETENTION_DAYS=30

# Create backup
influxd backup \
  --bucket market_data_bucket \
  --output-path "$BACKUP_DIR/$(date +%Y%m%d_%H%M%S)"

# Clean old backups
find "$BACKUP_DIR" -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} \;

# Upload to S3
aws s3 sync "$BACKUP_DIR" s3://market-data-backups/influxdb/
```

### 2. Parquet Archive

```python
# storage/archive_manager.py
import pyarrow.parquet as pq
from datetime import datetime, timedelta

class ArchiveManager:
    def archive_old_data(self, days=7):
        """Archive data older than N days to Parquet"""
        cutoff = datetime.now() - timedelta(days=days)
        
        # Query InfluxDB
        query = f"""
        from(bucket: "market_data_bucket")
          |> range(start: -90d, stop: {cutoff.isoformat()})
          |> to(csv: true)
        """
        
        df = influxdb_to_dataframe(query)
        
        # Write Parquet with compression
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            f"s3://market-data-archive/market_data_{cutoff.date()}.parquet",
            compression='snappy',
            compression_level=10
        )
```

### 3. Disaster Recovery Plan

| Scenario | Recovery Time | Recovery Point | Procedure |
|----------|---------------|-----------------|-----------|
| Single node failure | 5 min | Latest replica | Kubernetes auto-restart |
| InfluxDB data loss | 30 min | Last backup (daily) | Restore from S3 backup |
| Total system failure | 2 hours | Latest checkpoint | Redeploy from infrastructure-as-code |

---

## Security

### 1. API Authentication

```python
# security/auth.py
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthCredential

security = HTTPBearer()

def verify_token(credentials: HTTPAuthCredential = Depends(security)) -> str:
    token = credentials.credentials
    
    # Verify JWT token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["sub"]
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 2. Data Encryption

```bash
# InfluxDB TLS Configuration
[http]
  https-enabled = true
  https-certificate = "/etc/ssl/certs/influxdb.crt"
  https-private-key = "/etc/ssl/private/influxdb.key"

# Enforce HTTPS
http-tls-enabled = true
```

### 3. Network Security

```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: market-data-netpolicy
spec:
  podSelector:
    matchLabels:
      app: market-data-platform
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: market-data
      ports:
        - protocol: TCP
          port: 5000
  egress:
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 8086  # InfluxDB
        - protocol: TCP
          port: 6379  # Redis
        - protocol: TCP
          port: 443   # External APIs
```

---

## Maintenance

### 1. Health Checks

```bash
#!/bin/bash
# health_check.sh

echo "Checking InfluxDB..."
curl -f http://localhost:8086/health || exit 1

echo "Checking Grafana..."
curl -f http://localhost:3000/api/health || exit 1

echo "Checking ZMQ Broker..."
timeout 2 python -c "import zmq; zmq.Context().socket(zmq.REQ).connect('tcp://127.0.0.1:5555')" || exit 1

echo "All systems operational"
```

### 2. Database Optimization

```sql
-- InfluxDB maintenance
-- Compact shards
OPTIMIZE

-- Rebalance data
REBALANCE

-- Check shard stats
SHOW SHARDS
```

### 3. Log Aggregation

```yaml
# ELK Stack integration
filebeat:
  inputs:
    - type: log
      enabled: true
      paths:
        - /var/log/market-data-platform/*.log
        - /var/log/influxdb/*.log

elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "market-data-%{+yyyy.MM.dd}"
```

---

## Upgrades

### 1. Rolling Update

```bash
# Update Docker image tag
docker pull market-data-platform:1.1

# Update docker-compose
sed -i 's/market-data-platform:1.0/market-data-platform:1.1/' docker-compose.yml

# Perform rolling update
docker-compose up -d --no-deps --build market-app
```

### 2. Database Migration

```python
# migrations/migrate_v1_0_to_v1_1.py
from influxdb_client import InfluxDBClient

def migrate_data():
    """Migration script for data schema changes"""
    client = InfluxDBClient(...)
    
    # Migrate old measurements to new schema
    query = 'from(bucket: "market_data_bucket") |> range(start: -365d)'
    
    # Process and re-write with new schema
    ...
```

---

## Support & Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues and solutions.

---

**Last Updated**: 2024-01-15  
**Deployment Version**: 1.0.0
