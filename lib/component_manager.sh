#!/usr/bin/env bash
# Component Manager - Manages graceful start/stop of service groups
# Organizes services into logical component groups with dependencies

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
COMPONENTS_DIR="${PROJECT_ROOT}/config/components"
PIDS_DIR="${PROJECT_ROOT}/.pids"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${BLUE}→${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }

mkdir -p "${PIDS_DIR}"

###############################################################################
# COMPONENT DEFINITIONS
###############################################################################

# Define component groups: name:type:services or binaries
declare -A COMPONENTS
COMPONENTS[database]="docker:postgres redis"
COMPONENTS[monitoring]="docker:prometheus grafana"
COMPONENTS[storage]="docker:influxdb"
COMPONENTS[messaging]="binary:zmq-publisher zmq-subscriber"
COMPONENTS[api]="docker:python-api"
COMPONENTS[gateway]="docker:go-gateway"
COMPONENTS[processor]="docker:rust-processor"
COMPONENTS[proxy]="docker:nginx"

# Define component dependencies (must start in order)
declare -A DEPENDS_ON
DEPENDS_ON[monitoring]="database"
DEPENDS_ON[api]="database"
DEPENDS_ON[gateway]="database"
DEPENDS_ON[processor]="database messaging"
DEPENDS_ON[proxy]="api gateway"

# Define health checks for components
declare -A HEALTH_CHECKS
HEALTH_CHECKS[database]="postgres:pg_isready -U mdp_user"
HEALTH_CHECKS[database]="redis:redis-cli ping"
HEALTH_CHECKS[storage]="curl:http://localhost:8086/health"
HEALTH_CHECKS[monitoring]="curl:http://localhost:9090/-/healthy"
HEALTH_CHECKS[api]="curl:http://localhost:8000/health"
HEALTH_CHECKS[gateway]="curl:http://localhost:8080/health"
HEALTH_CHECKS[proxy]="curl:http://localhost/health"

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

# Get component type (docker or binary)
get_component_type() {
    local component=$1
    local def="${COMPONENTS[$component]}"
    echo "${def%%:*}"
}

# Get services/binaries for component
get_component_services() {
    local component=$1
    local def="${COMPONENTS[$component]}"
    echo "${def#*:}"
}

# Get dependencies for component
get_component_dependencies() {
    local component=$1
    echo "${DEPENDS_ON[$component]:-}"
}

# Check if component is running
is_component_running() {
    local component=$1
    local type=$(get_component_type "$component")
    
    case "$type" in
        docker)
            local services=$(get_component_services "$component")
            for service in $services; do
                if docker-compose ps --services --filter "status=running" | grep -q "^${service}$"; then
                    return 0
                fi
            done
            return 1
            ;;
        binary)
            local pids=$(find "$PIDS_DIR" -name "${component}*.pid" -exec cat {} \; 2>/dev/null)
            [ -n "$pids" ] && return 0 || return 1
            ;;
    esac
}

# Validate component health
validate_component_health() {
    local component=$1
    local max_retries=${2:-30}
    local retry_count=0
    
    log "Validating health of component: ${component}"
    
    while [ $retry_count -lt $max_retries ]; do
        case "$component" in
            database)
                if docker-compose exec -T postgres pg_isready -U mdp_user >/dev/null 2>&1 && \
                   docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
                    success "Database component healthy"
                    return 0
                fi
                ;;
            storage)
                if curl -s http://localhost:8086/health >/dev/null 2>&1; then
                    success "Storage component healthy"
                    return 0
                fi
                ;;
            monitoring)
                if curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
                    success "Monitoring component healthy"
                    return 0
                fi
                ;;
            api)
                if curl -s http://localhost:8000/health >/dev/null 2>&1; then
                    success "API component healthy"
                    return 0
                fi
                ;;
            gateway)
                if curl -s http://localhost:8080/health >/dev/null 2>&1; then
                    success "Gateway component healthy"
                    return 0
                fi
                ;;
            proxy)
                if curl -s http://localhost/health >/dev/null 2>&1; then
                    success "Proxy component healthy"
                    return 0
                fi
                ;;
            messaging)
                # ZMQ check - verify processes are running
                if is_component_running "messaging"; then
                    success "Messaging component healthy"
                    return 0
                fi
                ;;
        esac
        
        ((retry_count++))
        if [ $retry_count -lt $max_retries ]; then
            sleep 1
        fi
    done
    
    error "Component health check failed: ${component}"
    return 1
}

###############################################################################
# START FUNCTIONS
###############################################################################

# Start a single component
start_component() {
    local component=$1
    local type=$(get_component_type "$component")
    
    if is_component_running "$component"; then
        warn "Component already running: ${component}"
        return 0
    fi
    
    # Start dependencies first
    local deps=$(get_component_dependencies "$component")
    for dep in $deps; do
        if ! is_component_running "$dep"; then
            log "Starting dependency: ${dep}"
            start_component "$dep" || return 1
        fi
    done
    
    log "Starting component: ${component}"
    
    case "$type" in
        docker)
            local services=$(get_component_services "$component")
            docker-compose up -d $services || return 1
            sleep 2
            ;;
        binary)
            case "$component" in
                messaging)
                    start_messaging_component
                    ;;
            esac
            ;;
    esac
    
    # Validate health
    if ! validate_component_health "$component"; then
        error "Component failed health check: ${component}"
        return 1
    fi
    
    success "Component started: ${component}"
    return 0
}

# Start messaging (ZMQ) component
start_messaging_component() {
    local project_root="${PROJECT_ROOT:-.}"
    
    # Compile C services if needed
    if [ ! -f "$project_root/c/zmq_core/publisher" ]; then
        log "  Compiling ZMQ publisher..."
        gcc -O3 -Wall "$project_root/c/zmq_core/publisher.c" -o "$project_root/c/zmq_core/publisher" -lzmq 2>/dev/null || {
            error "Failed to compile publisher"
            return 1
        }
    fi
    
    if [ ! -f "$project_root/c/zmq_core/subscriber" ]; then
        log "  Compiling ZMQ subscriber..."
        gcc -O3 -Wall "$project_root/c/zmq_core/subscriber.c" -o "$project_root/c/zmq_core/subscriber" -lzmq -lpthread 2>/dev/null || {
            error "Failed to compile subscriber"
            return 1
        }
    fi
    
    # Start publisher
    log "  Starting ZMQ publisher..."
    nohup "$project_root/c/zmq_core/publisher" > "$project_root/logs/publisher.log" 2>&1 &
    echo $! > "$PIDS_DIR/zmq-publisher.pid"
    sleep 1
    
    # Start subscriber
    log "  Starting ZMQ subscriber..."
    nohup "$project_root/c/zmq_core/subscriber" > "$project_root/logs/subscriber.log" 2>&1 &
    echo $! > "$PIDS_DIR/zmq-subscriber.pid"
    sleep 1
    
    return 0
}

# Start multiple components (or all if empty)
start_components() {
    local components=("$@")
    
    if [ ${#components[@]} -eq 0 ]; then
        # Start all in proper order
        components=(database monitoring storage messaging api gateway processor proxy)
    fi
    
    for component in "${components[@]}"; do
        start_component "$component" || return 1
    done
    
    return 0
}

###############################################################################
# STOP FUNCTIONS
###############################################################################

# Stop a single component
stop_component() {
    local component=$1
    local type=$(get_component_type "$component")
    
    if ! is_component_running "$component"; then
        warn "Component not running: ${component}"
        return 0
    fi
    
    log "Stopping component: ${component}"
    
    case "$type" in
        docker)
            local services=$(get_component_services "$component")
            docker-compose stop $services 2>/dev/null || true
            ;;
        binary)
            case "$component" in
                messaging)
                    stop_messaging_component
                    ;;
            esac
            ;;
    esac
    
    success "Component stopped: ${component}"
    return 0
}

# Stop messaging (ZMQ) component
stop_messaging_component() {
    # Stop publisher
    if [ -f "$PIDS_DIR/zmq-publisher.pid" ]; then
        local pid=$(cat "$PIDS_DIR/zmq-publisher.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log "  Sending SIGTERM to ZMQ publisher (PID: $pid)"
            kill -TERM "$pid" 2>/dev/null || true
            
            # Wait for graceful shutdown
            for i in {1..5}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    success "  ZMQ publisher stopped"
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                log "  Force killing ZMQ publisher"
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PIDS_DIR/zmq-publisher.pid"
    fi
    
    # Stop subscriber
    if [ -f "$PIDS_DIR/zmq-subscriber.pid" ]; then
        local pid=$(cat "$PIDS_DIR/zmq-subscriber.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log "  Sending SIGTERM to ZMQ subscriber (PID: $pid)"
            kill -TERM "$pid" 2>/dev/null || true
            
            # Wait for graceful shutdown
            for i in {1..5}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    success "  ZMQ subscriber stopped"
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                log "  Force killing ZMQ subscriber"
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PIDS_DIR/zmq-subscriber.pid"
    fi
}

# Stop multiple components (or all if empty)
stop_components() {
    local components=("$@")
    
    if [ ${#components[@]} -eq 0 ]; then
        # Stop all in reverse order
        components=(proxy processor gateway api messaging storage monitoring database)
    fi
    
    for component in "${components[@]}"; do
        stop_component "$component"
    done
    
    return 0
}

###############################################################################
# STATUS FUNCTIONS
###############################################################################

# Get status of all components
status_all_components() {
    echo ""
    echo "Component Status:"
    echo "═══════════════════════════════════════════"
    
    for component in database monitoring storage messaging api gateway processor proxy; do
        if is_component_running "$component"; then
            echo -e "${GREEN}✓${NC} ${component}"
        else
            echo -e "${RED}✗${NC} ${component}"
        fi
    done
    
    echo "═══════════════════════════════════════════"
    echo ""
}

# Get detailed status
status_detailed() {
    echo ""
    echo "Docker Services:"
    docker-compose ps
    
    echo ""
    echo "Running Processes:"
    if [ -d "$PIDS_DIR" ]; then
        ls -la "$PIDS_DIR" 2>/dev/null || echo "  No process files"
    fi
    
    echo ""
}

###############################################################################
# MAIN ENTRY POINT
###############################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    action="${1:-}"
    shift || true
    
    case "$action" in
        start)
            start_components "$@"
            status_all_components
            ;;
        stop)
            stop_components "$@"
            status_all_components
            ;;
        status)
            status_all_components
            status_detailed
            ;;
        *)
            echo "Usage: $0 {start|stop|status} [component1 component2 ...]"
            echo ""
            echo "Available components:"
            echo "  database, monitoring, storage, messaging, api, gateway, processor, proxy"
            exit 1
            ;;
    esac
fi
