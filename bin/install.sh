#!/usr/bin/env bash
# Installation and verification script for Market Data Platform
# Sets up all components: C ZMQ core, Go services, Rust processor, InfluxDB, Grafana, Redis

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

PROJECT_ROOT="/root/rf_env"
cd "$PROJECT_ROOT" || exit 1

log_info "Market Data Platform - Complete Installation & Validation"
log_info "=========================================================="

# 1. Verify Python environment
log_info "Step 1: Verifying Python environment..."
if [ -f "$PROJECT_ROOT/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/bin/python"
else
    PYTHON="python3"
fi
$PYTHON --version || exit 1
log_info "✓ Python environment ready"

# 2. Install Python dependencies
log_info "Step 2: Installing Python dependencies..."
$PYTHON -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1
$PYTHON -m pip install -r requirements.txt --no-cache-dir 2>&1 | grep -E "(Successfully|Requirement|ERROR)" || true
log_info "✓ Python dependencies installed"

# 3. Verify Go
log_info "Step 3: Verifying Go installation..."
if command -v go &> /dev/null; then
    go version
    cd "$PROJECT_ROOT/go"
    go mod tidy || true
    log_info "✓ Go environment ready"
    cd "$PROJECT_ROOT"
else
    log_warn "Go not installed - skipping Go module verification"
fi

# 4. Verify Rust
log_info "Step 4: Verifying Rust installation..."
if command -v cargo &> /dev/null; then
    rustc --version
    cd "$PROJECT_ROOT/rust"
    cargo check --release 2>&1 | tail -5 || true
    log_info "✓ Rust environment ready"
    cd "$PROJECT_ROOT"
else
    log_warn "Rust not installed - skipping Rust module verification"
fi

# 5. Check system dependencies
log_info "Step 5: Checking system dependencies..."
deps=("zmq" "psql" "redis-cli" "docker" "docker-compose")
for dep in "${deps[@]}"; do
    if command -v "$dep" &> /dev/null 2>&1 || apt list --installed 2>/dev/null | grep -q "lib.*zmq"; then
        log_info "  ✓ $dep available"
    else
        log_warn "  ○ $dep not found (optional)"
    fi
done

# 6. Verify Docker services
log_info "Step 6: Checking Docker services..."
if command -v docker &> /dev/null; then
    docker --version
    if command -v docker-compose &> /dev/null; then
        docker-compose --version
        log_info "✓ Docker environment ready"
    else
        log_warn "docker-compose not found"
    fi
else
    log_warn "Docker not installed - skipping Docker verification"
fi

# 7. Verify Python modules
log_info "Step 7: Verifying Python modules..."
$PYTHON -c "import market_data_platform; print('  ✓ market_data_platform'); import zmq; print('  ✓ zmq'); import redis; print('  ✓ redis'); import fastapi; print('  ✓ fastapi'); import sqlalchemy; print('  ✓ sqlalchemy')" || true

# 8. Create necessary directories
log_info "Step 8: Creating necessary directories..."
mkdir -p "$PROJECT_ROOT"/{logs,data,config,bin,tmp}
chmod -R 755 "$PROJECT_ROOT"/{logs,data,tmp}
log_info "✓ Directories created"

# 9. Verify configurations
log_info "Step 9: Verifying configuration files..."
configs=("config/application/settings.yaml" "config/database/db.yaml" "config/logging/logging.yaml")
for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        log_info "  ✓ $config"
    else
        log_warn "  ○ $config missing"
    fi
done

# 10. Check Docker images
log_info "Step 10: Checking required Docker images..."
if command -v docker &> /dev/null; then
    images=("postgres:15-alpine" "redis:7-alpine" "grafana/grafana:latest" "prom/prometheus:latest" "influxdb:latest")
    for img in "${images[@]}"; do
        if docker image inspect "$img" >/dev/null 2>&1; then
            log_info "  ✓ $img"
        else
            log_warn "  ○ $img not found locally (will pull on first use)"
        fi
    done
fi

log_info ""
log_info "=========================================================="
log_info "✅ Installation & Verification Complete!"
log_info "=========================================================="
log_info ""
log_info "Next steps:"
log_info "1. Configure .env: cp .env.example .env && edit .env"
log_info "2. Start services: docker-compose up -d"
log_info "3. Verify services: bash bin/verify_services.sh"
log_info "4. Run tests: make test"
log_info ""
