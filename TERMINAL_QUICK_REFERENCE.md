# Terminal Quick Reference - Instant Command Guide

## Launch Commands

```bash
# Interactive mode selector
python -m market_data_platform.cli.enhanced_terminal_launcher

# Integrated mode (recommended)
python -m market_data_platform.cli.enhanced_terminal_launcher --mode integrated

# Standard Commander mode
python -m market_data_platform.cli.enhanced_terminal_launcher --mode commander

# Advanced Dashboard
python -m market_data_platform.cli.enhanced_terminal_launcher --mode dashboard

# Direct Commander
python market_data_platform/cli/commander_terminal.py
```

## Essential Keyboard Shortcuts

| Shortcut | Action | Mode |
|----------|--------|------|
| ↑ ↓ | Navigate items | All |
| ← → | Switch panels | All |
| Tab | Cycle panels | All |
| Enter | Execute item | All |
| F1 | Help | All |
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
