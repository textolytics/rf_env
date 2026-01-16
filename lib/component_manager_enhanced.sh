#!/usr/bin/env bash
# Enhanced Component Manager - Manages graceful start/stop/install/uninstall with dependencies
# Provides dependency resolution, state tracking, and rich status display

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
COMPONENTS_DIR="${PROJECT_ROOT}/config/components"
STATE_FILE="${PROJECT_ROOT}/.component_state.json"
PIDS_DIR="${PROJECT_ROOT}/.pids"
LOGS_DIR="${PROJECT_ROOT}/logs"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions
log() { echo -e "${BLUE}→${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }
info() { echo -e "${CYAN}ℹ${NC} $1"; }

# Create required directories
mkdir -p "${PIDS_DIR}" "${LOGS_DIR}" "${COMPONENTS_DIR}"

###############################################################################
# STATE MANAGEMENT
###############################################################################

init_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo '{"components": {}, "services": {}, "last_updated": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"}' > "$STATE_FILE"
    fi
}

get_service_state() {
    local service=$1
    python3 -c "import json; f=json.load(open('$STATE_FILE')); print(f.get('services', {}).get('$service', {}).get('state', 'unknown'))" 2>/dev/null || echo "unknown"
}

set_service_state() {
    local service=$1
    local state=$2
    python3 << EOF
import json
from datetime import datetime

with open('$STATE_FILE', 'r') as f:
    data = json.load(f)

if 'services' not in data:
    data['services'] = {}

data['services']['$service'] = {
    'state': '$state',
    'updated_at': datetime.utcnow().isoformat() + 'Z'
}

data['last_updated'] = datetime.utcnow().isoformat() + 'Z'

with open('$STATE_FILE', 'w') as f:
    json.dump(data, f, indent=2)
EOF
}

get_all_states() {
    python3 -c "import json; print(json.dumps(json.load(open('$STATE_FILE')).get('services', {}), indent=2))" 2>/dev/null || echo "{}"
}

###############################################################################
# SERVICE CONFIGURATION
###############################################################################

load_service_config() {
    local service=$1
    python3 << EOF
import yaml
try:
    with open('${PROJECT_ROOT}/config/services.yml', 'r') as f:
        config = yaml.safe_load(f)
        svc = config.get('services', {}).get('$service', {})
        print(f"type={svc.get('type', 'unknown')}")
        print(f"container={svc.get('container', '$service')}")
        print(f"port={svc.get('port', 'N/A')}")
        print(f"startup_cmd={svc.get('startup_cmd', '')}")
        print(f"shutdown_cmd={svc.get('shutdown_cmd', '')}")
        print(f"health_check={svc.get('health_check', '')}")
        deps=",".join(svc.get('depends_on', []))
        print(f"depends_on={deps}")
except Exception as e:
    print(f"error={e}")
EOF
}

get_service_type() {
    local service=$1
    eval "$(load_service_config "$service" | grep "^type=")"
    echo "${type:-unknown}"
}

get_service_depends() {
    local service=$1
    eval "$(load_service_config "$service" | grep "^depends_on=")"
    echo "${depends_on:-}"
}

get_startup_cmd() {
    local service=$1
    eval "$(load_service_config "$service" | grep "^startup_cmd=")"
    echo "${startup_cmd:-}"
}

get_shutdown_cmd() {
    local service=$1
    eval "$(load_service_config "$service" | grep "^shutdown_cmd=")"
    echo "${shutdown_cmd:-}"
}

get_health_check() {
    local service=$1
    eval "$(load_service_config "$service" | grep "^health_check=")"
    echo "${health_check:-}"
}

###############################################################################
# INSTALLATION & UNINSTALLATION
###############################################################################

install_service() {
    local service=$1
    local skip_deps=${2:-false}

    log "Installing $service..."
    set_service_state "$service" "installing"

    # Check and install dependencies
    if [ "$skip_deps" != "true" ]; then
        local deps=$(get_service_depends "$service")
        if [ -n "$deps" ]; then
            for dep in ${deps//,/ }; do
                local dep_state=$(get_service_state "$dep")
                if [ "$dep_state" != "running" ]; then
                    info "Installing dependency: $dep"
                    if ! install_service "$dep" "false"; then
                        error "Failed to install dependency: $dep"
                        set_service_state "$service" "failed"
                        return 1
                    fi
                fi
            done
        fi
    fi

    local startup_cmd=$(get_startup_cmd "$service")
    if [ -z "$startup_cmd" ]; then
        error "No startup command defined for $service"
        set_service_state "$service" "failed"
        return 1
    fi

    # Execute startup
    if eval "$startup_cmd" >> "${LOGS_DIR}/${service}.log" 2>&1; then
        success "$service installed and started"
        set_service_state "$service" "running"
        return 0
    else
        error "Failed to install $service"
        set_service_state "$service" "failed"
        return 1
    fi
}

uninstall_service() {
    local service=$1
    local remove_data=${2:-false}

    log "Uninstalling $service..."
    set_service_state "$service" "uninstalling"

    # First stop the service
    if ! stop_service "$service" "true"; then
        warn "Could not stop service before uninstall"
    fi

    # Execute shutdown command
    local shutdown_cmd=$(get_shutdown_cmd "$service")
    if [ -n "$shutdown_cmd" ]; then
        eval "$shutdown_cmd" >> "${LOGS_DIR}/${service}.log" 2>&1 || true
    fi

    # Clean up data if requested
    if [ "$remove_data" = "true" ]; then
        cleanup_service_data "$service"
    fi

    success "$service uninstalled"
    set_service_state "$service" "not_installed"
    return 0
}

cleanup_service_data() {
    local service=$1

    case "$service" in
        postgres)
            [ -d "${PROJECT_ROOT}/.pgdata" ] && rm -rf "${PROJECT_ROOT}/.pgdata"
            info "Cleaned up PostgreSQL data"
            ;;
        redis)
            [ -d "${PROJECT_ROOT}/.redis_data" ] && rm -rf "${PROJECT_ROOT}/.redis_data"
            info "Cleaned up Redis data"
            ;;
        influxdb)
            [ -d "${PROJECT_ROOT}/.influx_data" ] && rm -rf "${PROJECT_ROOT}/.influx_data"
            info "Cleaned up InfluxDB data"
            ;;
    esac
}

###############################################################################
# START/STOP OPERATIONS
###############################################################################

start_service() {
    local service=$1

    log "Starting $service..."

    # Check dependencies
    local deps=$(get_service_depends "$service")
    if [ -n "$deps" ]; then
        for dep in ${deps//,/ }; do
            local dep_state=$(get_service_state "$dep")
            if [ "$dep_state" != "running" ]; then
                error "Unsatisfied dependency: $dep is not running"
                return 1
            fi
        done
    fi

    local startup_cmd=$(get_startup_cmd "$service")
    if [ -z "$startup_cmd" ]; then
        error "No startup command defined for $service"
        return 1
    fi

    # Execute startup
    if eval "$startup_cmd" >> "${LOGS_DIR}/${service}.log" 2>&1; then
        success "$service started"
        set_service_state "$service" "running"

        # Check health
        sleep 1
        if check_health "$service"; then
            success "Health check passed"
            return 0
        else
            warn "Health check failed"
            return 0  # Don't fail if health check fails
        fi
    else
        error "Failed to start $service"
        set_service_state "$service" "failed"
        return 1
    fi
}

stop_service() {
    local service=$1
    local graceful=${2:-true}

    log "Stopping $service..."

    local shutdown_cmd=$(get_shutdown_cmd "$service")
    if [ -z "$shutdown_cmd" ]; then
        error "No shutdown command defined for $service"
        return 1
    fi

    # Graceful shutdown
    if [ "$graceful" = "true" ]; then
        if timeout 30 bash -c "eval \"$shutdown_cmd\"" >> "${LOGS_DIR}/${service}.log" 2>&1; then
            success "$service stopped gracefully"
            set_service_state "$service" "stopped"
            return 0
        else
            warn "Graceful stop failed, force stopping..."
            return $(force_stop_service "$service")
        fi
    else
        force_stop_service "$service"
        return $?
    fi
}

force_stop_service() {
    local service=$1

    local service_type=$(get_service_type "$service")

    if [ "$service_type" = "docker" ]; then
        local container=$(python3 -c "import yaml; f=yaml.safe_load(open('${PROJECT_ROOT}/config/services.yml')); print(f.get('services', {}).get('$service', {}).get('container', '$service'))")
        docker-compose kill "$container" 2>/dev/null || true
    else
        pkill -9 -f "$service" 2>/dev/null || true
    fi

    success "$service force stopped"
    set_service_state "$service" "stopped"
    return 0
}

check_health() {
    local service=$1
    local health_check=$(get_health_check "$service")

    if [ -z "$health_check" ]; then
        return 0  # No health check defined
    fi

    if timeout 10 bash -c "$health_check" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

###############################################################################
# GRACEFUL SHUTDOWN
###############################################################################

graceful_shutdown() {
    log "════════════════════════════════════════════════════════════════════"
    log "Initiating graceful shutdown..."
    log "════════════════════════════════════════════════════════════════════"
    log ""

    # Get all running services in reverse order
    local all_services=$(python3 -c "import yaml; f=yaml.safe_load(open('${PROJECT_ROOT}/config/services.yml')); print(' '.join(reversed(list(f.get('services', {}).keys()))))" 2>/dev/null || echo "")

    for service in $all_services; do
        local state=$(get_service_state "$service")
        if [ "$state" = "running" ]; then
            stop_service "$service" "true" || true
            echo ""
        fi
    done

    success "Graceful shutdown complete"
}

###############################################################################
# STATUS DISPLAY
###############################################################################

show_status() {
    local service=${1:-}

    if [ -n "$service" ]; then
        show_service_status "$service"
    else
        show_all_status
    fi
}

show_service_status() {
    local service=$1

    local state=$(get_service_state "$service")
    local service_type=$(get_service_type "$service")
    local port=$(python3 -c "import yaml; f=yaml.safe_load(open('${PROJECT_ROOT}/config/services.yml')); print(f.get('services', {}).get('$service', {}).get('port', 'N/A'))")
    local deps=$(get_service_depends "$service")

    echo ""
    echo -e "${BOLD}Service: ${CYAN}${service}${NC}${BOLD}${NC}"
    echo -e "  State:  ${state}"
    echo -e "  Type:   ${service_type}"
    echo -e "  Port:   ${port}"
    [ -n "$deps" ] && echo -e "  Depends: ${deps}"
    echo ""
}

show_all_status() {
    log "System Status"
    log "════════════════════════════════════════════════════════════════════"
    echo ""

    python3 << 'EOF'
import json
import yaml

try:
    # Load state
    with open('$STATE_FILE', 'r') as f:
        state = json.load(f)
    
    # Load config
    with open('${PROJECT_ROOT}/config/services.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"{'Service':<20} {'State':<15} {'Type':<10} {'Port':<8} {'Dependencies':<30}")
    print("─" * 85)
    
    for service in config.get('services', {}).keys():
        svc_config = config['services'][service]
        svc_state = state.get('services', {}).get(service, {})
        
        service_state = svc_state.get('state', 'unknown')
        svc_type = svc_config.get('type', 'unknown')
        port = str(svc_config.get('port', 'N/A'))
        deps = ', '.join(svc_config.get('depends_on', []))
        
        print(f"{service:<20} {service_state:<15} {svc_type:<10} {port:<8} {deps:<30}")
    
except Exception as e:
    print(f"Error: {e}")
EOF

    echo ""
}

###############################################################################
# BATCH OPERATIONS
###############################################################################

install_all() {
    log "Installing all services..."
    log "════════════════════════════════════════════════════════════════════"
    echo ""

    python3 << 'EOF' | while read service; do
        install_service "$service" "false" || return 1
        echo ""
    done
EOF

    success "All services installed"
}

uninstall_all() {
    local remove_data=${1:-false}

    log "Uninstalling all services..."
    log "════════════════════════════════════════════════════════════════════"
    echo ""

    python3 << 'EOF' | tac | while read service; do
        uninstall_service "$service" "$remove_data" || true
        echo ""
    done
EOF

    success "All services uninstalled"
}

###############################################################################
# MAIN ENTRY POINT
###############################################################################

main() {
    init_state

    local command=${1:-help}
    shift || true

    case "$command" in
        install)
            if [ "$#" -eq 0 ]; then
                install_all
            else
                for service in "$@"; do
                    install_service "$service" "false" || exit 1
                done
            fi
            ;;
        uninstall)
            local remove_data=false
            [ "${1:-}" = "--remove-data" ] && { remove_data=true; shift; }
            if [ "$#" -eq 0 ]; then
                uninstall_all "$remove_data"
            else
                for service in "$@"; do
                    uninstall_service "$service" "$remove_data" || true
                done
            fi
            ;;
        start)
            for service in "$@"; do
                start_service "$service" || exit 1
            done
            ;;
        stop)
            if [ "$#" -eq 0 ]; then
                graceful_shutdown
            else
                for service in "$@"; do
                    stop_service "$service" "true" || exit 1
                done
            fi
            ;;
        graceful-stop)
            graceful_shutdown
            ;;
        status)
            show_status "$@"
            ;;
        shutdown)
            graceful_shutdown
            ;;
        help|*)
            cat << 'HELP'
Component Manager - Service lifecycle management

USAGE:
  component-manager <command> [options]

COMMANDS:
  install [SERVICE...]         Install and start services
  uninstall [SERVICE...]       Uninstall services
    --remove-data              Remove service data
  start <SERVICE...>           Start services
  stop [SERVICE...]            Stop services (graceful)
  graceful-stop                Graceful shutdown of all services
  status [SERVICE]             Show status of services
  help                         Show this help message

EXAMPLES:
  component-manager install database
  component-manager install                # Install all
  component-manager start database api
  component-manager stop proxy api
  component-manager graceful-stop
  component-manager status database

HELP
            ;;
    esac
}

# Run main if script is executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
