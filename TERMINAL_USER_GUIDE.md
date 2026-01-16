# Market Data Terminal - Complete User Guide

## Overview

The Market Data Terminal provides a **Midnight Commander / Bloomberg Terminal style interface** for managing Robot Framework tests, pytest tests, CLI commands, and system monitoring. It features:

- **Two-panel navigation** (Midnight Commander style)
- **Keyboard-driven interface** with arrow keys and function keys
- **Real-time execution dashboard** (Bloomberg Terminal style)
- **4-color Byobu theme** for modern terminal aesthetics
- **Full Robot Framework integration** for test discovery and execution
- **Multi-mode operation** (Standard, Dashboard, Integrated)

## Quick Start

### Launch the Terminal

```bash
# Interactive mode selector
python market_data_platform/cli/enhanced_terminal_launcher.py

# Direct launch (integrated mode with all features)
python market_data_platform/cli/enhanced_terminal_launcher.py --mode integrated

# Standard Commander mode
python market_data_platform/cli/enhanced_terminal_launcher.py --mode commander

# Advanced Dashboard mode
python market_data_platform/cli/enhanced_terminal_launcher.py --mode dashboard

# Specify workspace
python market_data_platform/cli/enhanced_terminal_launcher.py --workspace /path/to/workspace --mode integrated
```

### Alternative: Direct Commander Terminal

```bash
python market_data_platform/cli/commander_terminal.py
```

## Terminal Layout

### Main Interface (Integrated Mode)

```
┌──────────────────────────────────────────────────────────────────┐
│  MARKET DATA COMMANDER TERMINAL (F1:Help, F9:Exit)             │
├──────────────────────────┬───────────────────────────────────────┤
│ 🚀 COMMANDS              │ 🧪 TESTS & TASKS                      │
│ ▶ install all            │ ▶ test_python_modules                 │
│   start services         │   test_go_connectivity                │
│   stop services          │   Deployment Suite                    │
│   health check           │   Integration Tests                   │
│   restart services       │   RF: Gateways                        │
│   connect gate.io        │   Data Operations                     │
│   connect oanda          │   Performance Tests                   │
│   price EURUSD           │   ZMQ Routing                         │
│   price BTCUSD           │   API Testing                         │
│   ohlc ETH_USDT          │                                       │
│   config show            │                                       │
│   config set             │                                       │
├──────────────────────────┴───────────────────────────────────────┤
│ 📊 EXECUTION DASHBOARD                                          │
│ Status: ✓ health check completed in 1.23s                     │
│ Recent Executions:                                             │
│  ✓ connect gate.io       2.34s                                │
│  ✓ health check          1.45s                                │
│  ✓ install all           15.67s                               │
└──────────────────────────────────────────────────────────────────┘
 F1:Help  F2:Refresh  F3:System  F4:Execute  F5:Copy  F9:Exit
```

### Advanced Dashboard Mode

```
┌──────────────────────────────────────────────────────────────────┐
│ SYSTEM STATUS            │ EXECUTION METRICS                    │
│ CPU  [████░░░░░] 45.2%   │ Total:     125 executions            │
│ MEM  [██████░░░]  68.5%  │ Success:   118 (94.4%)               │
│ DISK [███████░░░] 73.2%  │ Failed:      7                       │
│                          │ Avg Time:   2.34s                    │
├──────────────────────────┼────────────────────────────────────────┤
│ RECENT EXECUTIONS        │ TEST SUMMARY                         │
│ ✓ test_python  2.34s     │ ✓ test_python   100.0%              │
│ ✓ test_go      3.45s     │ ⚠ test_go        85.0%              │
│ ✓ test_cpp     4.12s     │ ✓ test_rust     100.0%              │
│ ✗ test_mobile  timeout   │ ✓ test_zmq       95.0%              │
└──────────────────────────┴────────────────────────────────────────┘
```

## Keyboard Navigation

### Arrow Keys & Movement

| Key | Action |
|-----|--------|
| ↑ / Down Arrow | Navigate items in active panel (up/down) |
| ↓ | Navigate items in active panel (down) |
| ← / Left Arrow | Switch to left panel (commands) |
| → / Right Arrow | Switch to right panel (tests/tasks) |
| Tab | Cycle between panels |

### Execution & Selection

| Key | Action |
|-----|--------|
| Enter | Execute selected item |
| Ctrl+E | Execute all tests in current panel |
| Ctrl+X | Stop current execution |
| Ctrl+C | Cancel (also quits) |

### Searching & Filtering

| Key | Action |
|-----|--------|
| Ctrl+F | Open command palette / search |
| Ctrl+S | Search in current panel |
| / | Quick search (alternative) |
| Esc | Close search / Cancel search |

### Function Keys

| Key | Action |
|-----|--------|
| F1 | Show help & keyboard shortcuts |
| F2 | Refresh panels (re-discover items) |
| F3 | Show system status panel |
| F4 | Execute selected item (same as Enter) |
| F5 | Copy selected item name |
| F6 | Open notebook browser |
| F7 | New filter / search window |
| F8 | Edit selected item |
| F9 | Exit terminal |
| F10 | Menu (advanced options) |

### Control Keys

| Key | Action |
|-----|--------|
| Ctrl+L | Clear execution log |
| Ctrl+R | Refresh everything |
| Ctrl+D | Debug selected item |
| Ctrl+V | Paste |
| Ctrl+H | Show execution history |
| Ctrl+T | Toggle dashboard panels |

### Other

| Key | Action |
|-----|--------|
| ? | Show help |
| q | Quit terminal |
| Ctrl+Q | Force quit |

## Command Panel (Left Side)

The left panel displays available commands organized by category:

### Setup Commands
- **install all** - Install all dependencies
- **config show** - Show configuration
- **config set** - Set configuration

### Service Management
- **start services** - Start all services (Docker)
- **stop services** - Stop all services
- **restart services** - Restart services
- **health check** - Run health check

### Connectivity
- **connect gate.io** - Test Gate.io connection
- **connect oanda** - Test OANDA connection

### Data Operations
- **price EURUSD** - Get EUR/USD price
- **price BTCUSD** - Get BTC/USD price
- **ohlc ETH_USDT** - Get ETH/USDT OHLC

### Monitoring
- **logs tail** - View live logs
- **monitor zmq** - Monitor ZMQ bus

## Tests & Tasks Panel (Right Side)

The right panel displays:

### Pytest Tests
- Python module tests
- Go connectivity tests
- Rust framework tests
- C++ integration tests
- ZMQ routing tests
- Performance tests

### Robot Framework Tasks
- Deployment suite
- Gateway tests
- Data operations
- Integration tests

### Notebook Cells (in Notebook Browser mode)
- Code cells from Jupyter notebooks
- Markdown cells for documentation
- Execute cells directly from terminal

## Execution Dashboard (Bottom Section)

The execution dashboard tracks:

- **Status Line** - Shows last executed item and result
- **Recent Executions** - History of last 5-10 executed items
- **Execution Time** - Duration of each execution
- **Color Indicators**:
  - ✓ Green = Successful execution
  - ✗ Yellow = Failed execution
  - ◉ Blue = Currently executing

## Advanced Dashboard Mode

Launch with: `--mode dashboard`

### Panels

1. **System Status**
   - CPU usage with progress bar
   - Memory usage with progress bar
   - Disk usage with progress bar

2. **Execution Metrics**
   - Total executions
   - Success/failure counts
   - Success rate percentage
   - Average execution time

3. **Recent Executions**
   - Last 5-10 executed items
   - Status (✓/✗)
   - Execution duration
   - Timestamp

4. **Test Summary**
   - Per-test success rates
   - Status indicators
   - Trend information

## Color Scheme (Byobu 4-Color Theme)

| Color | Usage |
|-------|-------|
| 🔵 Blue (PRIMARY) | Headers, focus indicators, panel titles |
| 🟢 Green (SUCCESS) | Executable items, successful status, ✓ indicators |
| 🟡 Yellow (WARNING) | Important items, failed status, ✗ indicators, cautions |
| 🔵 Cyan (INFO) | Information text, alternative items, ℹ indicators |

## Configuration

Terminal settings are stored in `~/.robotmcp_terminal.json`

### Example Configuration

```json
{
  "theme": "byobu",
  "last_panel": "commands",
  "search_history": ["python", "test_go"],
  "favorite_commands": ["health check", "test_regression"],
  "auto_refresh": true,
  "refresh_interval": 5000,
  "execution_timeout": 30,
  "max_history": 100
}
```

### Load Custom Configuration

```bash
# Edit configuration
vim ~/.robotmcp_terminal.json

# Terminal will auto-load on next start
```

## Integration with Robot Framework

### Discover Robot Framework Tests

The terminal automatically discovers:
- All `.robot` files in workspace
- All test cases in Robot Framework suites
- All custom keywords from Python libraries
- All resource files

### Run Robot Framework Tests

1. Navigate to test in right panel
2. Press **Enter** or **F4**
3. Watch execution dashboard for status
4. View output in dashboard or separate window

### Example Robot Framework Test Names

- `Deployment Suite :: Deploy Application`
- `Gateways :: Test Gate.io Connectivity`
- `Data Operations :: Fetch OHLC Data`
- `Integration :: Multi-Language Test`

## Integration with Pytest

### Run Pytest Tests

1. Navigate to test in right panel
2. Press **Enter** or **F4**
3. Wait for test execution (max 60s timeout)
4. View result in dashboard

### Test Results

- ✓ = All assertions passed
- ✗ = Test failed
- Duration shown in seconds

## Notebook Integration

### Browse Jupyter Notebooks

1. Press **F6** (Open Notebook Browser)
2. Select notebook from list
3. View cells and metadata
4. Execute code cells directly
5. Press **Enter** to run selected cell

### Notebook Cell Types

- **Code** - Executable Python cells (green indicator)
- **Markdown** - Documentation cells (cyan indicator)

## Command Execution

### How Commands Execute

1. **Parse** - Terminal identifies command type
2. **Prepare** - Set working directory and environment
3. **Execute** - Run command with timeout
4. **Capture** - Collect stdout/stderr output
5. **Track** - Record execution time and status
6. **Display** - Show result in dashboard

### Execution Timeouts

- Regular commands: **30 seconds**
- Test suites: **60 seconds**
- Performance tests: **120 seconds**

### Execution Output

Output can be viewed:
1. In the dashboard status line
2. Full output in separate panel (F3)
3. In execution history (Ctrl+H)
4. Exported to file (F5)

## Search & Filtering

### Quick Search

1. Press **Ctrl+F** to open command palette
2. Type search query
3. Results filter in real-time
4. Arrow keys to select
5. Enter to execute

### Search Examples

```
"test_py"     -> Find: test_python_modules
"gate"        -> Find: connect gate.io, Gateway tests
"health"      -> Find: health check
```

## Troubleshooting

### Terminal Not Starting

```bash
# Check terminal size (min 60x20)
stty size

# Clear terminal
reset

# Try with compatibility mode
python enhanced_terminal_launcher.py --mode commander
```

### Colors Not Displaying

```bash
# Check terminal support
echo $TERM

# Set proper terminal
export TERM=xterm-256color

# Or use compatible mode
python enhanced_terminal_launcher.py --no-colors
```

### Tests Not Discovering

```bash
# Verify Robot Framework installed
robot --version

# Check pytest available
pytest --version

# Manually trigger refresh
# Press F2 in terminal
```

### Execution Timeout

```bash
# Increase timeout for slow tests
# Edit ~/.robotmcp_terminal.json
{
  "execution_timeout": 120
}
```

## Performance Tips

1. **Use Command Palette** (Ctrl+F) for large test suites
2. **Group Related Tests** by using search filtering
3. **Monitor Dashboard** to track resource usage
4. **Clear History** periodically (Ctrl+L)
5. **Refresh Selectively** rather than full refresh (F2)

## Keyboard Shortcut Reference Card

```
┌─ NAVIGATION ─────────────┐ ┌─ EXECUTION ──────────────┐
│ ↑↓ Navigate items        │ │ Enter / F4 Execute      │
│ ←→ Switch panels         │ │ Ctrl+E Run all tests    │
│ Tab Cycle panels         │ │ Ctrl+X Stop execution   │
│ Ctrl+F Search/palette    │ │ Ctrl+L Clear log        │
│ Ctrl+S Search in panel   │ │ Ctrl+H Show history     │
└──────────────────────────┘ └─────────────────────────┘

┌─ PANELS ──────────────────┐ ┌─ DASHBOARD ───────────────┐
│ F1 Help                   │ │ F3 System status         │
│ F2 Refresh all            │ │ F5 Copy item name        │
│ F6 Notebook browser       │ │ F9 Exit                  │
│ F8 Edit item              │ │ ? Help                   │
│ F10 Menu                  │ │ q Quit                   │
└──────────────────────────┘ └─────────────────────────┘
```

## Integration with CLI Tools

### Combined Usage

```bash
# 1. Start terminal
python market_data_platform/cli/enhanced_terminal_launcher.py

# 2. In terminal, execute command (F4)
# -> Command: connect gate.io

# 3. View results in dashboard
# -> Status: ✓ Connected to Gate.io

# 4. Run test (F4)
# -> Test: test_go_connectivity
# -> Result: ✓ All tests passed

# 5. Check system (F3)
# -> CPU: 45.2%, Memory: 68.5%

# 6. Exit (F9)
```

## Advanced Topics

### Custom Command Registration

Edit `terminal_integration.py`:

```python
def _load_default_commands(self):
    custom_command = {
        "my_command": {
            "description": "My custom command",
            "category": "custom",
            "command": "python my_script.py"
        }
    }
    self.register_command("my_command", custom_command)
```

### Extend Keyboard Bindings

Edit `commander_terminal.py` `handle_input()` method to add custom key handlers.

### Create Custom Dashboard Panels

Extend `AdvancedDashboard` class in `advanced_dashboard.py`:

```python
def render_custom_panel(self, window, y, x, width, height):
    # Your custom panel rendering
    pass
```

## Feedback & Support

For issues, suggestions, or improvements:

1. Check existing Robot Framework documentation
2. Review test results in dashboard
3. Check system status (F3)
4. View execution history (Ctrl+H)
5. Report issues with log output

## See Also

- [Robot Framework Documentation](https://robotframework.org)
- [Pytest Documentation](https://pytest.org)
- [Market Data Platform CLI Guide](README.md)
- [Testing Infrastructure Guide](../testing/TESTING_README.md)
