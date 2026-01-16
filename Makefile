# Makefile - Project Build and Development Commands

.PHONY: help install build test clean run docker

help:
	@echo "Market Data Platform - Build Commands"
	@echo "====================================="
	@echo "make install          - Install all dependencies"
	@echo "make build            - Build all modules"
	@echo "make build-python     - Build Python module"
	@echo "make build-go         - Build Go module"
	@echo "make build-rust       - Build Rust module"
	@echo "make test             - Run all tests"
	@echo "make test-python      - Run Python tests"
	@echo "make test-go          - Run Go tests"
	@echo "make test-rust        - Run Rust tests"
	@echo "make test-integration - Run integration tests"
	@echo "make run              - Run the platform"
	@echo "make docker-build     - Build Docker images"
	@echo "make docker-up        - Start Docker containers"
	@echo "make docker-down      - Stop Docker containers"
	@echo "make clean            - Clean build artifacts"
	@echo "make lint             - Run linters"
	@echo "make format           - Format code"

# Installation targets
install:
	pip install -r requirements.txt
	cd go && go mod download
	cd rust && cargo fetch

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt
	cd go && go mod download
	cd rust && cargo fetch

# Build targets
build: build-python build-go build-rust
	@echo "✅ All modules built successfully"

build-python:
	@echo "Building Python module..."
	python setup.py build
	@echo "✅ Python module built"

build-go:
	@echo "Building Go module..."
	cd go && go build -o ../bin/gateway ./cmd/gateway
	cd go && go build -o ../bin/client ./cmd/client
	@echo "✅ Go module built"

build-rust:
	@echo "Building Rust module..."
	cd rust && cargo build --release
	@echo "✅ Rust module built"

# Test targets
test: test-python test-go test-rust test-integration
	@echo "✅ All tests passed"

test-python:
	@echo "Running Python tests..."
	pytest tests/ -v --cov=market_data_platform

test-go:
	@echo "Running Go tests..."
	cd go && go test ./... -v

test-rust:
	@echo "Running Rust tests..."
	cd rust && cargo test --release

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

test-robot:
	@echo "Running Robot Framework tests..."
	robot -d results robot_framework/test_suites

# Linting and formatting
lint:
	@echo "Running linters..."
	pylint market_data_platform tests
	cd go && golangci-lint run
	cd rust && cargo clippy -- -D warnings

format:
	@echo "Formatting code..."
	black market_data_platform tests
	isort market_data_platform tests
	cd go && go fmt ./...
	cd rust && cargo fmt --all

# Docker targets
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Run targets
run:
	python -m market_data_platform

run-cli:
	python -m market_data_platform.cli.unified_terminal_launcher

run-api:
	uvicorn market_data_platform.api:app --reload

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	cd go && go clean
	cd rust && cargo clean
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

# Development
dev-setup: install-dev
	pre-commit install

migrate-db:
	@echo "Running database migrations..."
	psql -U postgres -d market_data < config/database/schema.sql

init-cache:
	@echo "Initializing cache..."
	redis-cli FLUSHALL

init-zmq:
	@echo "Initializing ZMQ..."
	# ZMQ initialization commands

# Utility targets
status:
	docker-compose ps

logs:
	docker-compose logs -f

shell:
	python -i -c "from market_data_platform import *"

version:
	@cat VERSION

.DEFAULT_GOAL := help
