# Terminal System - Quick Reference Card

## 🚀 QUICK START

```bash
cd /root/rf_env
python market_data_platform/cli/unified_terminal_launcher.py
# Select: 1 (Advanced Menu Terminal)
```

---

## ⌨️ KEYBOARD CONTROLS

| Key | Action |
|-----|--------|
| **↑** | Move up in menu |
| **↓** | Move down in menu |
| **→** | In some menus: Next option |
| **←** | In some menus: Previous option |
| **Enter** | Select/Execute current item |
| **q** | Go back or Quit |
| **1-9** | Jump to menu item number |

---

## 📋 MAIN MENU STRUCTURE

```
1.  📋 Components
    ├─ View Status      → Show all 7 components + status
    ├─ Start Component  → Start any component
    └─ Stop Component   → Stop any component

2.  🧪 Testing
    ├─ Run Tests        → 6 test types (All, Python, Go, Rust, RF, Integration)
    └─ Run Specific Test → 6 test suites

3.  ⚙️ Configuration
    ├─ View Config      → Display all 15 settings
    ├─ Edit Config      → Change: Theme, Refresh, Timeout, Ports, LogLevel
    └─ Reset Config     → Restore defaults

4.  🔑 Keywords
    └─ Show Keywords    → Browse 5 categories (27 total keywords)

5.  💾 Commands
    └─ Execute Command  → Health Check, Install, Build, Connect, Get Prices

6.  q - Back/Quit
```

---

## 🔧 COMPONENTS (7 Total)

1. **ZMQ Bus** (Port 5555) - Message routing
2. **Python Gateway** (Port 8001) - Python module
3. **Go Gateway** (Port 8002) - Gate.io connector
4. **Rust Gateway** (Port 8003) - Data processor
5. **Robot Framework** - Test automation
6. **Redis Cache** (Port 6379) - Data cache
7. **Postgres DB** (Port 5432) - Primary database

---

## 🧪 TESTS (12 Total)

**Test Types** (6):
- All Tests
- Python Tests
- Go Tests
- Rust Tests
- Robot Framework Tests
- Integration Tests

**Specific Tests** (6):
- Test Python Modules
- Test Go Connectivity
- Test Rust Processor
- Test RF Execution
- Test Data Pipeline
- Test Complete Flow

---

## ⚙️ CONFIGURATION (15 Settings)

```
theme                   = byobu
auto_start              = true
auto_refresh            = true
refresh_interval        = 5000 (ms)
execution_timeout       = 30 (s)
max_history             = 100 (lines)
zmq_host                = 127.0.0.1
zmq_port                = 5555
python_gateway_port     = 8001
go_gateway_port         = 8002
rust_gateway_port       = 8003
database                = postgresql://localhost/market_data
redis_url               = redis://localhost:6379
log_level               = INFO
environment             = development
```

---

## 🔑 KEYWORDS (27 Total in 5 Categories)

**Gateway Management (6)**: Connect, Disconnect, List, GetStatus, Stream, Stop

**Component Management (6)**: Start, Stop, Check, Restart, GetInfo, ShowAll

**Data Operations (5)**: FetchOHLC, ProcessData, Store, Query, Aggregate

**Configuration (5)**: SetValue, GetValue, Load, Save, Reset

**Testing (5)**: RunAll, RunPython, RunGo, RunRust, RunIntegration

---

## 💾 COMMANDS (5 Total)

1. **Health Check** - Verify all systems operational
2. **Install Dependencies** - Setup environment
3. **Build All** - Compile all modules
4. **Connect Gate.io** - Test external connectivity
5. **Get Market Prices** - Fetch current market data

---

## 📊 COMPONENT STATUS INDICATORS

| Symbol | Status | Meaning |
|--------|--------|---------|
| ▶ | RUNNING | Component is active |
| ⊡ | STOPPED | Component is inactive |
| ✗ | ERROR | Component has error |
| ? | UNKNOWN | Status unknown |
| ⟳ | STARTING | Component starting up |

---

## 🎨 COLOR SCHEME

| Color | Usage |
|-------|-------|
| 🔵 Blue | Headers, titles, sections |
| 🟢 Green | Menu items, success |
| 🟡 Yellow | Warnings, timeouts |
| 🔷 Cyan | Information, help |

---

## 📁 FILES

```
/root/rf_env/market_data_platform/cli/
├── unified_terminal_launcher.py       # ← START HERE
├── advanced_menu_terminal.py          # Main menu system (30.8 KB)
├── commander_terminal.py              # Two-panel interface (20.1 KB)
├── advanced_dashboard.py              # System monitoring (15.3 KB)
├── terminal_integration.py            # Utilities (16.7 KB)
└── test_components.py                 # Validation tests (9.8 KB)

~/.market_data_config.json              # Configuration file
```

---

## ✅ COMMON TASKS

### Check Component Status
1. Launch → Option 1
2. Press ↓ to "View Status"
3. Press Enter
4. Review each component

### Start/Stop Components
1. Go to "Start Component" or "Stop Component"
2. Press Enter
3. Select component with ↓
4. Press Enter to execute

### Run Tests
1. Go to "Run Tests"
2. Press Enter
3. Select test type
4. Press Enter to run

### Manage Configuration
- **View**: Go to "View Config"
- **Edit**: Go to "Edit Config" → select setting → modify
- **Reset**: Go to "Reset Config" (auto-saved)

### Browse Keywords
1. Go to "Show Keywords"
2. Press Enter
3. Select category (27 total in 5 categories)
4. View keywords for that category

---

## 🔍 STATUS INFORMATION

**Show detailed status:**
```bash
Components:  7 available (ZMQ, Python, Go, Rust, RF, Redis, Postgres)
Keywords:    27 total in 5 categories (Gateway, Component, Data, Config, Testing)
Settings:    15 configuration options (theme, ports, timeouts, URLs, etc.)
Tests:       12 test suites (6 types + 6 specific)
Commands:    5 system commands (health, install, build, connect, prices)
```

---

## 🚀 VALIDATION

**Verify all systems operational:**
```bash
python /root/rf_env/market_data_platform/cli/test_components.py
```

**Expected output:**
```
✓ PASS - File Structure
✓ PASS - Components (7 components)
✓ PASS - Configuration (15 settings)
✓ PASS - Keywords (27 keywords)
✓ PASS - Menu System (21 menu items)
✓ PASS - Unified Launcher

Results: 6/6 tests passed ✅
```

---

## 🎯 GETTING STARTED

1. **Launch the terminal**
   ```bash
   python unified_terminal_launcher.py
   ```

2. **Select Mode 1** (Advanced Menu Terminal)

3. **Navigate** with arrow keys (↑↓)

4. **Select** with Enter key

5. **Go Back** with 'q' key

6. **Explore** Components → Testing → Config → Keywords → Commands

---

## 📚 DOCUMENTATION

- **User Guide**: See [TERMINAL_SYSTEM_GUIDE.md](TERMINAL_SYSTEM_GUIDE.md)
- **Technical Details**: See [TERMINAL_IMPLEMENTATION_COMPLETE.md](TERMINAL_IMPLEMENTATION_COMPLETE.md)
- **This File**: Quick reference for common tasks

---

## ⚡ QUICK COMMANDS

```bash
# Launch unified launcher
python /root/rf_env/market_data_platform/cli/unified_terminal_launcher.py

# Direct to menu terminal
python /root/rf_env/market_data_platform/cli/advanced_menu_terminal.py

# Validate all systems
python /root/rf_env/market_data_platform/cli/test_components.py

# View configuration
cat ~/.market_data_config.json

# Edit configuration
nano ~/.market_data_config.json
```

---

## 💡 TIPS & TRICKS

- Use **↑↓** for navigation (faster than typing)
- Press **q** at any time to go back
- Type **number keys** to jump to menu items
- Status bar shows hints for current menu
- Output section shows last 100 lines of activity
- All operations are logged to output history

---

## 🆘 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| Terminal won't start | Ensure Linux/WSL2, Python 3.7+, curses available |
| Colors broken | Set `TERM=xterm-256color` |
| Component not responding | Try "Start Component" from menu |
| Config not saving | Check ~/.market_data_config.json permissions |
| Tests failing | Run "Health Check" or "Install Dependencies" |

---

## 📊 SYSTEM STATUS

```
Status: ✅ OPERATIONAL
Tests:  6/6 Passed ✅
Files:  5 core + 1 test file
Code:   97+ KB production code
Ready:  YES - Launch immediately!
```

---

**Version**: 1.0  
**Status**: Production Ready ✅  
**Last Updated**: 2024
| F2 | Refresh | All |
| F3 | System status | All |
| F4 | Execute | All |
| F6 | Notebooks | All |
| F9 | Exit | All |
| Ctrl+F | Search/Palette | All |
| Ctrl+E | Execute all | Tests |
| Ctrl+L | Clear log | All |
| Ctrl+H | History | All |
| ? | Help | All |
| q | Quit | All |

## Command Panel Items

```
📦 Setup Commands
├─ install all
├─ config show
└─ config set

🚀 Services
├─ start services
├─ stop services
├─ restart services
└─ health check

🔌 Connectivity
├─ connect gate.io
└─ connect oanda

📊 Data Operations
├─ price EURUSD
├─ price BTCUSD
└─ ohlc ETH_USDT

📈 Monitoring
├─ logs tail
└─ monitor zmq
```

## Tests & Tasks Panel

```
🧪 Pytest Tests (Auto-discovered)
├─ test_python_modules
├─ test_go_connectivity
├─ test_rust_modules
├─ test_cpp_integration
├─ test_zmq_routing
└─ test_performance

🤖 Robot Framework Tasks (Auto-discovered)
├─ Deployment Suite
├─ Gateway Tests
├─ Data Operations
└─ Integration Tests

📓 Notebook Cells (F6)
├─ Code cells (executable)
└─ Markdown cells (docs)
```

## Status Indicators

| Symbol | Meaning | Color |
|--------|---------|-------|
| ✓ | Success | 🟢 Green |
| ✗ | Failed | 🟡 Yellow |
| ◉ | Running | 🔵 Blue |
| ○ | Pending | 🔵 Cyan |
| ⚠ | Warning | 🟡 Yellow |
| ℹ | Info | 🔵 Cyan |

## Navigation Workflow

```
1. Launch terminal
   → python enhanced_terminal_launcher.py
   → Select mode (integrated recommended)

2. View panels
   → Left: Commands
   → Right: Tests & Tasks
   → Bottom: Dashboard

3. Navigate
   → Use arrow keys to select
   → Tab to switch panels

4. Execute
   → Press Enter or F4
   → Watch dashboard for status

5. Monitor
   → F3 for system status
   → Ctrl+H for history
   → F2 to refresh

6. Exit
   → F9 or q or Ctrl+Q
```

## Search & Filter Workflow

```
1. Press Ctrl+F
   → Opens command palette
   → Shows search box

2. Type query
   → "python" → Find Python tests
   → "gate" → Find Gate.io commands
   → "health" → Find health check

3. Arrow keys to select
   → Navigate filtered results

4. Press Enter
   → Execute selected item

5. Press Esc
   → Close search
```

## Panel Navigation

```
LEFT PANEL (Commands)          RIGHT PANEL (Tests/Tasks)
┌─────────────────────┐        ┌─────────────────────┐
│ ▶ install all       │        │ ▶ test_python       │
│   start services    │   ←→   │   test_go           │
│   stop services     │        │   Deployment        │
│   health check      │        │   Integration       │
│   connect gate.io   │        │                     │
└─────────────────────┘        └─────────────────────┘

   ↑ ↓ Navigate           ↑ ↓ Navigate
   ← Switch              → Switch
   Tab Toggle            Tab Toggle
   Enter Execute         Enter Execute
```

## Dashboard Reading Guide

```
EXECUTION DASHBOARD
┌────────────────────────────────────────┐
│ Status: ✓ health check in 1.23s        │ ← Last execution result
├────────────────────────────────────────┤
│ Recent Executions (Last 5):            │
│ ✓ connect gate.io       2.34s          │
│ ✓ health check          1.45s          │
│ ✓ install all           15.67s         │
│ ✗ test_mobile           timeout        │ ← Failures in yellow
│ ✓ test_go               8.92s          │
└────────────────────────────────────────┘

  ✓ = Success, ✗ = Failed, Time in seconds
```

## Common Tasks

### Execute a Command
```
1. Navigate left panel (← key)
2. Find command (↑ ↓ keys)
3. Press Enter
4. Check dashboard for result
```

### Run a Test
```
1. Navigate right panel (→ key)
2. Find test (↑ ↓ keys)
3. Press Enter (or F4)
4. Wait for completion
5. View result in dashboard
```

### Search for Command
```
1. Press Ctrl+F
2. Type command name
3. Select from results
4. Press Enter to execute
```

### Refresh All Items
```
1. Press F2
2. Terminal re-discovers items
3. Panels update automatically
```

### View System Status
```
1. Press F3
2. View CPU, Memory, Disk usage
3. See execution metrics
4. Press Esc or q to close
```

### Browse Notebooks
```
1. Press F6
2. Select notebook
3. View cells
4. Select cell
5. Press Enter to execute
```

### Clear History
```
1. Press Ctrl+L
2. Execution log cleared
3. Dashboard resets
```

### Show Help
```
1. Press F1 (or ?)
2. View keyboard shortcuts
3. View help sections
4. Press Esc to close
```

## Configuration File

**Location**: `~/.robotmcp_terminal.json`

**Example**:
```json
{
  "theme": "byobu",
  "auto_refresh": true,
  "refresh_interval": 5000,
  "execution_timeout": 30,
  "max_history": 100
}
```

## Troubleshooting Quick Tips

| Problem | Solution |
|---------|----------|
| Terminal not starting | Check terminal size (min 60x20) |
| Colors not showing | Set `export TERM=xterm-256color` |
| Tests not found | Press F2 to refresh |
| Command timeout | Increase timeout in config |
| Panels misaligned | Resize terminal window |

## Terminal Modes Explained

| Mode | Best For | Features |
|------|----------|----------|
| Commander | Daily use | Two-panel navigation, quick access |
| Dashboard | Monitoring | System stats, metrics, history |
| Integrated | Full-featured | All features + RF integration |

## Color Legend

| Color | Usage | Example |
|-------|-------|---------|
| 🔵 Blue | Headers, titles | Panel names, menu items |
| 🟢 Green | Success, executable | ✓ marks, runnable items |
| 🟡 Yellow | Warning, failed | ✗ marks, timeouts |
| 🔵 Cyan | Info, alternative | ℹ marks, alternatives |

## Performance Optimization

**Tips for smooth operation**:

1. **Use search** for large test suites (Ctrl+F)
2. **Group navigation** with Tab key
3. **Check dashboard** for bottlenecks (F3)
4. **Clear history** periodically (Ctrl+L)
5. **Increase timeout** for slow tests

## Keyboard Reference Poster

```
╔═══════════════════════════════════════════════════════════╗
║     MARKET DATA TERMINAL - KEYBOARD REFERENCE            ║
╠═══════════════════════════════════════════════════════════╣
║ NAVIGATION        │ EXECUTION       │ FUNCTION KEYS      ║
║ ↑↓  Navigate      │ Enter Execute   │ F1   Help          ║
║ ←→  Panels        │ F4    Execute   │ F2   Refresh       ║
║ Tab  Cycle        │ Ctrl+E All      │ F3   System        ║
║ Ctrl+F Search     │ Ctrl+X Stop     │ F6   Notebooks     ║
║                   │ Ctrl+L Clear    │ F9   Exit          ║
╚═══════════════════════════════════════════════════════════╝
```

## Emergency Exit

```bash
# If stuck in terminal
Press: Ctrl+C

# Force quit
Press: Ctrl+Q

# Last resort (from another terminal)
pkill -f enhanced_terminal_launcher
```

## Getting Help

```bash
# Show help in terminal
Press: F1

# View this guide
cat TERMINAL_USER_GUIDE.md

# Check Robot Framework status
robot --version

# Check Pytest
pytest --version

# View logs
tail -f /tmp/market_data.log
```

## Next Steps

1. **Start the terminal** - `python enhanced_terminal_launcher.py`
2. **Execute a command** - Navigate and press Enter
3. **Run a test** - Select test and press F4
4. **Check system** - Press F3
5. **Read full guide** - See `TERMINAL_USER_GUIDE.md`

---
**Quick Reference v1.0** | Market Data Platform
Last Updated: 2024 | For latest updates, see TERMINAL_USER_GUIDE.md
