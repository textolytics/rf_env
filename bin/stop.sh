#!/usr/bin/env bash
# Graceful shutdown script - stops all services in proper order
# Sends SIGTERM signals and waits for graceful shutdown

set -euo pipefail

PROJECT_ROOT="/root/rf_env"
cd "$PROJECT_ROOT" || exit 1

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}→${NC} $1"; }
done() { echo -e "${GREEN}✓${NC} $1"; }

# Create PID directory if needed
mkdir -p .pids

# Stop application services first (give them time to gracefully shutdown)
log "Stopping application services..."

# Stop ZMQ services
if [ -f ".pids/publisher.pid" ] && kill -0 "$(cat .pids/publisher.pid)" 2>/dev/null; then
    log "  Stopping ZMQ publisher (PID: $(cat .pids/publisher.pid))..."
    kill -TERM "$(cat .pids/publisher.pid)" 2>/dev/null || true
    for i in {1..5}; do
        if ! kill -0 "$(cat .pids/publisher.pid)" 2>/dev/null; then
            done "  ZMQ publisher stopped"
            break
        fi
        sleep 1
    done
    rm -f .pids/publisher.pid
fi

if [ -f ".pids/subscriber.pid" ] && kill -0 "$(cat .pids/subscriber.pid)" 2>/dev/null; then
    log "  Stopping ZMQ subscriber (PID: $(cat .pids/subscriber.pid))..."
    kill -TERM "$(cat .pids/subscriber.pid)" 2>/dev/null || true
    for i in {1..5}; do
        if ! kill -0 "$(cat .pids/subscriber.pid)" 2>/dev/null; then
            done "  ZMQ subscriber stopped"
            break
        fi
        sleep 1
    done
    rm -f .pids/subscriber.pid
fi

# Stop Docker services
log "Stopping Docker services..."
log "  Stopping application containers..."
docker-compose stop python-api go-gateway rust-processor nginx 2>/dev/null || true

log "  Stopping infrastructure services..."
docker-compose stop grafana prometheus influxdb redis postgres 2>/dev/null || true

# Give services time to flush buffers
sleep 2

# Remove containers
log "Removing containers..."
docker-compose down 2>/dev/null || true

# Clear process IDs
rm -f .pids/*.pid
mkdir -p .pids

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "All services stopped gracefully"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Show Docker status
if command -v docker &> /dev/null; then
    remaining=$(docker ps --filter "name=market_data|name=mdp" --format "{{.Names}}" 2>/dev/null | wc -l)
    if [ "$remaining" -gt 0 ]; then
        echo -e "${RED}Warning: $remaining containers still running${NC}"
        docker ps --filter "name=market_data|name=mdp" --format "table {{.Names}}\t{{.Status}}"
    else
        echo -e "${GREEN}✓ All containers stopped${NC}"
    fi
fi
