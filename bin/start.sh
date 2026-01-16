#!/usr/bin/env bash
# Graceful startup script - starts all services in proper order
# Ensures dependencies are ready before starting dependent services

set -euo pipefail

PROJECT_ROOT="/root/rf_env"
cd "$PROJECT_ROOT" || exit 1

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}→${NC} $1"; }
done() { echo -e "${GREEN}✓${NC} $1"; }

trap 'echo "Startup interrupted"; exit 1' SIGINT SIGTERM

# Start Docker services first
log "Starting Docker services (PostgreSQL, Redis, InfluxDB, Grafana, Prometheus)..."
docker-compose up -d postgres redis influxdb grafana prometheus
done "Docker services starting (waiting for health checks)..."

# Wait for critical services
log "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker-compose exec -T postgres pg_isready -U postgres >/dev/null 2>&1; then
        done "PostgreSQL ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "PostgreSQL failed to start"; exit 1
    fi
    sleep 1
done

log "Waiting for Redis to be ready..."
for i in {1..20}; do
    if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
        done "Redis ready"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "Redis failed to start"; exit 1
    fi
    sleep 1
done

log "Waiting for InfluxDB to be ready..."
for i in {1..20}; do
    if curl -s http://localhost:8086/health >/dev/null 2>&1; then
        done "InfluxDB ready"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "InfluxDB failed to start"; exit 1
    fi
    sleep 1
done

# Initialize database schema
log "Initializing database schema..."
if [ -f "database/schema.sql" ]; then
    docker-compose exec -T postgres psql -U postgres -d market_data < database/schema.sql || true
    done "Database schema initialized"
else
    echo "Warning: schema.sql not found"
fi

# Initialize InfluxDB
log "Initializing InfluxDB..."
docker-compose exec -T influxdb influx bucket create -n market_data --retention 30d -o influxdata -t "$(docker-compose exec -T influxdb influx auth create --org influxdata --description 'Market data token' --write-bucket 'd0d2df2b9e3cc000' --token market-data-token 2>/dev/null | grep -oP 'REDACTED' || echo 'local-token')" >/dev/null 2>&1 || true
done "InfluxDB initialized"

# Start ZMQ core services
log "Compiling C ZMQ services..."
if [ ! -f "c/zmq_core/publisher" ]; then
    gcc -O3 -Wall c/zmq_core/publisher.c -o c/zmq_core/publisher -lzmq 2>/dev/null || {
        echo "C compilation failed - check libzmq installation"; exit 1
    }
fi
if [ ! -f "c/zmq_core/subscriber" ]; then
    gcc -O3 -Wall c/zmq_core/subscriber.c -o c/zmq_core/subscriber -lzmq -lpthread 2>/dev/null || {
        echo "C compilation failed - check libzmq installation"; exit 1
    }
fi
done "C services compiled"

# Start ZMQ services in background
log "Starting ZMQ publisher..."
nohup c/zmq_core/publisher > logs/publisher.log 2>&1 &
echo $! > .pids/publisher.pid
sleep 1
done "ZMQ publisher started (PID: $(cat .pids/publisher.pid))"

log "Starting ZMQ subscriber..."
nohup c/zmq_core/subscriber > logs/subscriber.log 2>&1 &
echo $! > .pids/subscriber.pid
sleep 1
done "ZMQ subscriber started (PID: $(cat .pids/subscriber.pid))"

# Start Python services
log "Starting Python API server..."
docker-compose up -d python-api
sleep 3
done "Python API server started"

# Start Go services
log "Starting Go gateway..."
if [ -f "go/cmd/gateway/main.go" ]; then
    docker-compose up -d go-gateway
    sleep 2
    done "Go gateway started"
else
    echo "Go gateway not yet implemented"
fi

# Start Rust services
log "Starting Rust processor..."
if [ -f "rust/src/bin/validator.rs" ]; then
    docker-compose up -d rust-processor
    sleep 2
    done "Rust processor started"
else
    echo "Rust processor not yet implemented"
fi

# Start Nginx (reverse proxy)
log "Starting Nginx..."
docker-compose up -d nginx
sleep 1
done "Nginx started"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "All services started successfully!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Running services:"
docker-compose ps
echo ""
echo "Dashboards & Services:"
echo "  Grafana:     http://localhost:3000 (admin/admin)"
echo "  Prometheus:  http://localhost:9090"
echo "  API:         http://localhost:8000"
echo "  Gateway:     http://localhost:8080"
echo ""
echo "To view logs: tail -f logs/*.log"
echo "To stop services: bash bin/stop.sh"
echo ""
