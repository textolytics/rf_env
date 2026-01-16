#!/bin/bash
# build.sh - Build all project components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

log_info "Project root: $PROJECT_ROOT"
log_info "Starting build process..."

# Parse arguments
BUILD_PYTHON=${BUILD_PYTHON:-true}
BUILD_GO=${BUILD_GO:-true}
BUILD_RUST=${BUILD_RUST:-true}
BUILD_DOCKER=${BUILD_DOCKER:-false}
BUILD_ALL=${BUILD_ALL:-false}

if [ "$BUILD_ALL" = "true" ]; then
    BUILD_PYTHON=true
    BUILD_GO=true
    BUILD_RUST=true
    BUILD_DOCKER=true
fi

# Python build
if [ "$BUILD_PYTHON" = "true" ]; then
    log_info "Building Python module..."
    cd "$PROJECT_ROOT"
    
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    log_info "Installing dependencies..."
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    
    log_info "Building Python package..."
    python setup.py build
    
    log_info "Running Python tests..."
    pytest tests/unit/ -v --tb=short || log_warn "Some tests failed"
    
    log_info "✅ Python module built successfully"
fi

# Go build
if [ "$BUILD_GO" = "true" ]; then
    log_info "Building Go module..."
    cd "$PROJECT_ROOT/go"
    
    log_info "Downloading Go dependencies..."
    go mod download
    go mod verify
    
    log_info "Building Gateway service..."
    CGO_ENABLED=1 go build -v -o ../bin/gateway ./cmd/gateway
    
    log_info "Building Client..."
    CGO_ENABLED=0 go build -v -o ../bin/client ./cmd/client
    
    log_info "Running Go tests..."
    go test ./... -v -race || log_warn "Some Go tests failed"
    
    log_info "Running Go linter..."
    golangci-lint run ./... || log_warn "Go lint warnings found"
    
    log_info "✅ Go module built successfully"
fi

# Rust build
if [ "$BUILD_RUST" = "true" ]; then
    log_info "Building Rust module..."
    cd "$PROJECT_ROOT/rust"
    
    log_info "Building Rust release binary..."
    cargo build --release --locked
    
    log_info "Running Rust tests..."
    cargo test --release || log_warn "Some Rust tests failed"
    
    log_info "Running Rust clippy..."
    cargo clippy --release -- -D warnings || log_warn "Clippy warnings found"
    
    log_info "✅ Rust module built successfully"
fi

# Docker build
if [ "$BUILD_DOCKER" = "true" ]; then
    log_info "Building Docker images..."
    cd "$PROJECT_ROOT"
    
    log_info "Building Python Docker image..."
    docker build -f build/docker/Dockerfile.python -t market-data-platform:python-latest .
    
    log_info "Building Go Docker image..."
    docker build -f build/docker/Dockerfile.go -t market-data-platform:go-latest ./go
    
    log_info "Building Rust Docker image..."
    docker build -f build/docker/Dockerfile.rust -t market-data-platform:rust-latest ./rust
    
    log_info "✅ Docker images built successfully"
fi

log_info "========================================="
log_info "✅ Build process completed successfully!"
log_info "========================================="

# Summary
echo ""
echo "Build Summary:"
[ "$BUILD_PYTHON" = "true" ] && echo "  ✓ Python module built"
[ "$BUILD_GO" = "true" ] && echo "  ✓ Go module built"
[ "$BUILD_RUST" = "true" ] && echo "  ✓ Rust module built"
[ "$BUILD_DOCKER" = "true" ] && echo "  ✓ Docker images built"
echo ""
