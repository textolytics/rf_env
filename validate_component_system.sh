#!/usr/bin/env bash
# Quick Start Validation Script
# Verifies all component management system files and configurations

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${BLUE}→${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Market Data Platform - Component Management System v2.0"
echo "Quick Start Validation"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check Python
log "Checking Python environment..."
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    success "Python $PYTHON_VERSION found"
else
    error "Python 3 not found"
    exit 1
fi

# Check Docker
log "Checking Docker..."
if command -v docker &>/dev/null && command -v docker-compose &>/dev/null; then
    success "Docker and Docker Compose installed"
else
    warn "Docker or Docker Compose not found (required for docker-based services)"
fi

# Check Bash
log "Checking Bash..."
if command -v bash &>/dev/null; then
    BASH_VERSION=$(bash --version | head -1)
    success "$BASH_VERSION"
else
    error "Bash not found"
    exit 1
fi

# Check Required Directories
echo ""
log "Checking project structure..."

required_dirs=(
    "bin"
    "config"
    "lib"
    "market_data_platform/cli"
    "logs"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$PROJECT_ROOT/$dir" ]; then
        success "Found: $dir"
    else
        warn "Missing: $dir (will be created)"
        mkdir -p "$PROJECT_ROOT/$dir"
    fi
done

# Check Required Files
echo ""
log "Checking component files..."

required_files=(
    "bin/mdp-cli"
    "bin/mdp-components"
    "bin/mdp-status"
    "bin/mdp-terminal"
    "config/services.yml"
    "market_data_platform/cli/component_manager.py"
    "market_data_platform/cli/rich_status.py"
    "market_data_platform/cli/terminal_ui.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        success "Found: $file"
    else
        error "Missing: $file"
    fi
done

# Check Python Packages
echo ""
log "Checking Python dependencies..."

python3 << 'PYTHON_EOF'
import sys

packages = ['yaml', 'rich', 'typer']
missing = []

for package in packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (MISSING)")
        missing.append(package)

if missing:
    print(f"\n  Install missing packages:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)
PYTHON_EOF

# Check Configuration
echo ""
log "Checking service configuration..."

if [ -f "$PROJECT_ROOT/config/services.yml" ]; then
    python3 << 'PYTHON_EOF'
import yaml
try:
    with open('config/services.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    services = config.get('services', {})
    components = config.get('components', {})
    
    print(f"  ✓ Services configured: {len(services)}")
    print(f"  ✓ Components defined: {len(components)}")
    
    # List components
    print("\n  Components:")
    for comp in components.keys():
        print(f"    - {comp}")
except Exception as e:
    print(f"  ✗ Configuration error: {e}")
PYTHON_EOF
else
    error "config/services.yml not found"
fi

# Check Scripts
echo ""
log "Checking script permissions..."

scripts=(
    "bin/mdp-cli"
    "bin/mdp-components"
    "bin/mdp-status"
    "bin/mdp-terminal"
    "lib/component_manager_enhanced.sh"
)

for script in "${scripts[@]}"; do
    if [ -f "$PROJECT_ROOT/$script" ]; then
        if [ -x "$PROJECT_ROOT/$script" ]; then
            success "Executable: $script"
        else
            warn "Not executable: $script"
            chmod +x "$PROJECT_ROOT/$script"
            success "Made executable: $script"
        fi
    fi
done

# Check Documentation
echo ""
log "Checking documentation..."

docs=(
    "COMPONENT_MANAGEMENT_SYSTEM.md"
    "DEPLOYMENT_TESTING_GUIDE.md"
    "COMPONENT_MANAGEMENT_COMPLETE.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$PROJECT_ROOT/$doc" ]; then
        success "Found: $doc"
    else
        warn "Missing: $doc"
    fi
done

# Final Summary
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
success "Component Management System Validation Complete"
echo ""
echo "Next Steps:"
echo "  1. Review configuration: cat config/services.yml"
echo "  2. Install services: ./bin/mdp-cli component install"
echo "  3. Check status: ./bin/mdp-cli component status"
echo "  4. View dashboard: ./bin/mdp-terminal"
echo ""
echo "Documentation:"
echo "  - ./bin/mdp-cli help"
echo "  - COMPONENT_MANAGEMENT_SYSTEM.md"
echo "  - DEPLOYMENT_TESTING_GUIDE.md"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
