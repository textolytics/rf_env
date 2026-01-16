#!/usr/bin/env bash
# Graceful startup script - Enhanced with component management
# Starts services by component group with health validation

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
cd "$PROJECT_ROOT" || exit 1

# Source component manager
source lib/component_manager.sh

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${BLUE}→${NC} $1"; }
done() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

trap 'echo "Startup interrupted"; exit 1' SIGINT SIGTERM

# Create necessary directories
mkdir -p logs .pids database

log "════════════════════════════════════════════════════════════════════"
log "Market Data Platform - Graceful Component Startup"
log "════════════════════════════════════════════════════════════════════"
log ""

# Start database layer (dependencies for everything)
log "Step 1: Starting Database Layer"
log "Starting PostgreSQL and Redis..."
start_component "database" || { error "Failed to start database"; exit 1; }
echo ""

# Initialize database schema
log "Initializing database schema..."
if [ -f "database/schema.sql" ]; then
    docker-compose exec -T postgres psql -U mdp_user -d market_data < database/schema.sql 2>/dev/null || true
    done "Database schema initialized"
else
    warn "schema.sql not found - using docker defaults"
fi

# Start storage layer
log "Step 2: Starting Storage Layer"
log "Starting InfluxDB..."
start_component "storage" || { warn "Storage layer startup had issues"; }
echo ""

# Start monitoring layer
log "Step 3: Starting Monitoring Layer"
log "Starting Prometheus and Grafana..."
start_component "monitoring" || { warn "Monitoring layer startup had issues"; }
echo ""

# Start messaging layer (ZMQ)
log "Step 4: Starting Messaging Layer"
log "Compiling and starting ZMQ services..."
start_component "messaging" || { error "Failed to start messaging"; exit 1; }
    }
fi
if [ ! -f "c/zmq_core/subscriber" ]; then
    gcc -O3 -Wall c/zmq_core/subscriber.c -o c/zmq_core/subscriber -lzmq -lpthread 2>/dev/null || {
        echo "C compilation failed - check libzmq installation"; exit 1
    }
fi
done "C services compiled"

# Start ZMQ services in background
log "Step 4: Starting Messaging Layer"
log "Compiling and starting ZMQ services..."
start_component "messaging" || { error "Failed to start messaging"; exit 1; }
echo ""

# Start application layer
log "Step 5: Starting Application Services"
log "Starting Python API..."
start_component "api" || { warn "API startup had issues"; }
echo ""

log "Starting Go Gateway..."
start_component "gateway" || { warn "Gateway startup had issues"; }
echo ""

log "Starting Rust Processor..."
start_component "processor" || { warn "Processor startup had issues"; }
echo ""

# Start proxy layer
log "Step 6: Starting Proxy Layer"
log "Starting Nginx reverse proxy..."
start_component "proxy" || { warn "Proxy startup had issues"; }
echo ""

# Final status
log "Startup complete!"
echo ""
status_all_components

# Show connectivity information
log "Service Connectivity:"
log "════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Dashboards & Monitoring:"
echo "  • Grafana:      http://localhost:3000 (admin/admin)"
echo "  • Prometheus:   http://localhost:9090"
echo ""
echo "🔌 API Services:"
echo "  • Python API:   http://localhost:8000"
echo "  • API Docs:     http://localhost:8000/docs"
echo "  • Go Gateway:   http://localhost:8080"
echo ""
echo "📝 Data Access:"
echo "  • Redis CLI:    redis-cli"
echo "  • PostgreSQL:   psql -h localhost -U mdp_user -d market_data"
echo "  • InfluxDB:     http://localhost:8086"
echo ""
echo "🔄 Messaging (ZMQ):"
echo "  • Publisher:    tcp://127.0.0.1:5555"
echo "  • Subscriber:   tcp://127.0.0.1:5556"
echo ""
echo "📋 Log Files:"
echo "  • tail -f logs/publisher.log"
echo "  • tail -f logs/subscriber.log"
echo "  • docker-compose logs -f <service>"
echo ""
echo "🛑 To stop all services: bash bin/stop.sh"
echo "📊 To check status: bash bin/component_manager.sh status"
echo ""
