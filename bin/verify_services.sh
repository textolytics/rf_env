#!/usr/bin/env bash
# Comprehensive service verification and health check script
# Verifies all components: ZMQ, Redis, InfluxDB, Grafana, PostgreSQL, services

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_pass() { echo -e "${GREEN}✓${NC} $1"; }
log_fail() { echo -e "${RED}✗${NC} $1"; }
log_info() { echo -e "${BLUE}→${NC} $1"; }
log_warn() { echo -e "${YELLOW}!${NC} $1"; }

PROJECT_ROOT="/root/rf_env"
cd "$PROJECT_ROOT" || exit 1

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Market Data Platform - Service Verification"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

FAILED=0

# 1. Check ZMQ
log_info "Checking ZMQ infrastructure..."
if command -v zmq_version &> /dev/null || netstat -tuln 2>/dev/null | grep -q ":5555\|:5556"; then
    log_pass "ZMQ ports available"
else
    log_warn "ZMQ not running (will start with docker-compose)"
fi

# 2. Check Redis
log_info "Checking Redis..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping >/dev/null 2>&1; then
        redis_info=$(redis-cli info server | grep redis_version | cut -d':' -f2)
        log_pass "Redis running (version: $redis_info)"
    else
        log_warn "Redis not responding - check docker-compose"
        FAILED=$((FAILED + 1))
    fi
else
    log_warn "redis-cli not installed"
fi

# 3. Check PostgreSQL
log_info "Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    if psql -h localhost -U postgres -d market_data -c "SELECT 1" >/dev/null 2>&1; then
        log_pass "PostgreSQL connected"
    else
        log_warn "PostgreSQL not responding - check docker-compose or configuration"
        FAILED=$((FAILED + 1))
    fi
else
    log_warn "psql not installed"
fi

# 4. Check InfluxDB
log_info "Checking InfluxDB..."
if curl -s http://localhost:8086/health >/dev/null 2>&1; then
    log_pass "InfluxDB running"
else
    log_warn "InfluxDB not responding at http://localhost:8086"
    FAILED=$((FAILED + 1))
fi

# 5. Check Grafana
log_info "Checking Grafana..."
if curl -s http://localhost:3000 >/dev/null 2>&1; then
    log_pass "Grafana running at http://localhost:3000"
else
    log_warn "Grafana not responding at http://localhost:3000"
    FAILED=$((FAILED + 1))
fi

# 6. Check Prometheus
log_info "Checking Prometheus..."
if curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
    log_pass "Prometheus running at http://localhost:9090"
else
    log_warn "Prometheus not responding at http://localhost:9090"
fi

# 7. Check Python services
log_info "Checking Python services..."
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    log_pass "Python API running at http://localhost:8000"
else
    log_warn "Python API not responding at http://localhost:8000"
fi

# 8. Check Go gateway
log_info "Checking Go gateway..."
if curl -s http://localhost:8080/health >/dev/null 2>&1; then
    log_pass "Go gateway running at http://localhost:8080"
else
    log_warn "Go gateway not responding at http://localhost:8080"
fi

# 9. Database tables
log_info "Checking database tables..."
if command -v psql &> /dev/null; then
    table_count=$(psql -h localhost -U postgres -d market_data -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null || echo "0")
    if [ "$table_count" -gt 0 ]; then
        log_pass "Database tables initialized ($table_count tables)"
    else
        log_warn "Database tables not initialized"
    fi
fi

# 10. Check data flow
log_info "Checking data flow..."
if command -v redis-cli &> /dev/null; then
    key_count=$(redis-cli dbsize 2>/dev/null | grep -oP '\d+' || echo "0")
    if [ "$key_count" -gt 0 ]; then
        log_pass "Data in Redis ($key_count keys)"
    else
        log_warn "No data in Redis - check gateways"
    fi
fi

# 11. Check Docker containers
log_info "Checking Docker containers..."
if command -v docker &> /dev/null; then
    running=$(docker ps --filter "status=running" --format "{{.Names}}" 2>/dev/null | wc -l)
    if [ "$running" -gt 0 ]; then
        log_pass "$running Docker containers running"
        docker ps --filter "status=running" --format "table {{.Names}}\t{{.Status}}" | sed 's/^/  /'
    else
        log_warn "No Docker containers running"
    fi
fi

# 12. Check logs
log_info "Checking service logs..."
if [ -d "logs" ]; then
    log_count=$(find logs -type f -name "*.log" 2>/dev/null | wc -l)
    log_pass "$log_count log files in logs/ directory"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Verification Summary"
echo "═══════════════════════════════════════════════════════════════════"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All systems operational!${NC}"
else
    echo -e "${YELLOW}! $FAILED components need attention${NC}"
fi

echo ""
echo "Service URLs:"
echo "  API:         http://localhost:8000"
echo "  Gateway:     http://localhost:8080"
echo "  Grafana:     http://localhost:3000 (admin/admin)"
echo "  Prometheus:  http://localhost:9090"
echo "  InfluxDB:    http://localhost:8086"
echo "  Redis:       localhost:6379"
echo "  PostgreSQL:  localhost:5432"
echo ""

exit $FAILED
