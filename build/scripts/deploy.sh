#!/bin/bash
# deploy.sh - Deploy services to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KUBE_DIR="$PROJECT_ROOT/build/kubernetes"

# Variables
NAMESPACE=${NAMESPACE:-market-data}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-docker.io}
IMAGE_TAG=${IMAGE_TAG:-latest}
ENVIRONMENT=${ENVIRONMENT:-production}

log_info "Kubernetes Deployment Script"
log_info "============================"
log_info "Namespace: $NAMESPACE"
log_info "Environment: $ENVIRONMENT"
log_info "Docker Registry: $DOCKER_REGISTRY"
log_info "Image Tag: $IMAGE_TAG"

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    log_error "docker is not installed"
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

log_info "✓ Prerequisites satisfied"

# Create namespace
log_info "Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
log_info "✓ Namespace ready"

# Apply namespace configuration
log_info "Applying namespace configuration..."
kubectl apply -f "$KUBE_DIR/namespace.yaml"
log_info "✓ Namespace configuration applied"

# Create secrets
log_info "Creating secrets..."
kubectl apply -f "$KUBE_DIR/python/deployment.yaml" --dry-run=client -o yaml | \
    grep "kind: Secret" || kubectl apply -f "$KUBE_DIR/python/deployment.yaml"
log_info "✓ Secrets created"

# Deploy database
log_info "Deploying PostgreSQL..."
kubectl apply -f "$KUBE_DIR/database/postgres.yaml"
log_warn "Waiting for PostgreSQL to be ready (this may take a while)..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s || log_warn "PostgreSQL not ready yet"
log_info "✓ PostgreSQL deployed"

# Deploy cache
log_info "Deploying Redis..."
kubectl apply -f "$KUBE_DIR/cache/redis.yaml"
log_warn "Waiting for Redis to be ready..."
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s || log_warn "Redis not ready yet"
log_info "✓ Redis deployed"

# Deploy Python API
log_info "Deploying Python API..."
kubectl apply -f "$KUBE_DIR/python/deployment.yaml"
log_warn "Waiting for Python API to be ready..."
kubectl wait --for=condition=ready pod -l app=python-api -n $NAMESPACE --timeout=300s || log_warn "Python API not ready yet"
log_info "✓ Python API deployed"

# Deploy Go Gateway
log_info "Deploying Go Gateway..."
kubectl apply -f "$KUBE_DIR/go/deployment.yaml"
log_warn "Waiting for Go Gateway to be ready..."
kubectl wait --for=condition=ready pod -l app=go-gateway -n $NAMESPACE --timeout=300s || log_warn "Go Gateway not ready yet"
log_info "✓ Go Gateway deployed"

# Deploy Rust Processor
log_info "Deploying Rust Processor..."
kubectl apply -f "$KUBE_DIR/rust/deployment.yaml"
log_info "✓ Rust Processor deployed"

# Display deployment status
log_info "Deployment Status:"
log_info "=================="
kubectl get all -n $NAMESPACE
echo ""

# Get service endpoints
log_info "Service Endpoints:"
log_info "=================="
kubectl get svc -n $NAMESPACE -o wide
echo ""

# Show deployment details
log_info "Deployment Details:"
log_info "==================="
for deployment in $(kubectl get deployments -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}'); do
    log_info "Deployment: $deployment"
    kubectl rollout status deployment/$deployment -n $NAMESPACE --timeout=300s || log_warn "Rollout timeout for $deployment"
done

log_info "✅ Deployment completed!"
echo ""
echo "To view logs:"
echo "  kubectl logs -f deployment/python-api -n $NAMESPACE"
echo "  kubectl logs -f deployment/go-gateway -n $NAMESPACE"
echo "  kubectl logs -f deployment/rust-processor -n $NAMESPACE"
echo ""
echo "To access services:"
echo "  kubectl port-forward svc/python-api-service 8000:8000 -n $NAMESPACE"
echo "  kubectl port-forward svc/go-gateway-service 8080:8080 -n $NAMESPACE"
echo ""
