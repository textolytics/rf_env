#!/usr/bin/env bash
# Graceful shutdown script - Enhanced with component management
# Stops services in reverse order with graceful shutdown

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
cd "$PROJECT_ROOT" || exit 1

# Source component manager
source lib/component_manager.sh

trap 'echo "Stop interrupted"; exit 1' SIGINT SIGTERM

log "════════════════════════════════════════════════════════════════════"
log "Market Data Platform - Graceful Component Shutdown"
log "════════════════════════════════════════════════════════════════════"
log ""

# Stop components in reverse order (dependencies last)
log "Initiating graceful shutdown of all components..."
log ""

log "Step 1: Stopping Proxy Layer"
stop_component "proxy" || true
echo ""

log "Step 2: Stopping Application Services"
stop_component "processor" || true
stop_component "gateway" || true
stop_component "api" || true
echo ""

log "Step 3: Stopping Messaging Layer"
stop_component "messaging" || true
echo ""

log "Step 4: Stopping Monitoring Layer"
stop_component "monitoring" || true
echo ""

log "Step 5: Stopping Storage Layer"
stop_component "storage" || true
echo ""

log "Step 6: Stopping Database Layer"
stop_component "database" || true
echo ""

# Final cleanup
log "Performing final cleanup..."
docker-compose down 2>/dev/null || true
sleep 1

# Clear PID directory
rm -f .pids/*.pid

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "All services stopped gracefully"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

status_all_components

# Show remaining Docker status
if command -v docker &> /dev/null; then
    remaining=$(docker ps --filter "name=mdp" --format "{{.Names}}" 2>/dev/null | wc -l)
    if [ "$remaining" -gt 0 ]; then
        warn "⚠ $remaining Docker containers still running"
        docker ps --filter "name=mdp" --format "table {{.Names}}\t{{.Status}}"
    else
        echo -e "${GREEN}✓ All containers stopped${NC}"
    fi
fi

echo ""
echo "✅ Graceful shutdown complete"
echo ""
